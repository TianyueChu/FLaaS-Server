from django.utils import timezone
from django.db.models import Q
from datetime import timedelta

import os
import io
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from api.models import Project, Round, DeviceTrainRequest, DeviceStatusResponse
from api import device_control as dc
from api import server_control as sc

from api.libs import consts
import torch

from kd.run.kd_pipeline import run_knowledge_distillation

PAST_DEVICE_STATUS_REPORTS_CONSIDERATION_MINS = 60


def __keep_last_per_device(responses, verbose=False):

    device_responses = {}

    for response in responses:

        # get username
        username = response.device.profile.user.username

        # if device for this user does not exist
        if username not in device_responses.keys():
            device_responses[username] = response

        # OR exists but its older
        elif response.create_date > device_responses[username].create_date:
            device_responses[username] = response

    # convert into a list and return
    return list(device_responses.values())


def __query_device_status_responses(project, verbose=False):

    # get all devices registered to this project
    registered_devices = [profile.device for profile in project.profiles.all()]

    # Past X minutes
    from_date_filter = timezone.now() - timedelta(minutes=PAST_DEVICE_STATUS_REPORTS_CONSIDERATION_MINS)

    # apply filters
    responses = DeviceStatusResponse.objects.filter(
        create_date__gt=from_date_filter,
        device__in=registered_devices)

    # build query based on settings in Project
    if project.power_plugged_only:

        # power plugged only
        responses = responses.filter(data__battery_status__power_plugged=True)

    else:

        # power plugged OR > battery_level_threshold
        battery_level_threshold = float(project.battery_level_threshold)
        responses = responses.filter(Q(data__battery_status__power_plugged=True) | Q(data__battery_status__level__gte=battery_level_threshold))

    # only keep the last response per device
    responses = __keep_last_per_device(responses, verbose)

    return responses


def __is_project_complete(project, verbose=False):

    completed_rounds = Round.objects.filter(
        project=project,
        status=Round.Status.COMPLETE)

    if len(completed_rounds) >= project.number_of_rounds:
        verbose and print("Project is complete (reached %d successfully completed rounds)" % len(completed_rounds))
        return True
    else:
        return False


def __check_for_round_completion(round, verbose=False):

    # get project
    project = round.project

    # check if devices are available
    if project.profiles.count() == 0:
        verbose and print("No devices are attached to this project.")
        return

    # compute responses and ratio
    try:
        device_train_responses = round.device_train_request.device_train_responses.count()
    except DeviceTrainRequest.DoesNotExist:
        device_train_responses = 0
    all_devices = project.profiles.count()
    trained_ratio = device_train_responses / all_devices

    # check if time has passed (OR all devices reported a model)
    if trained_ratio == 1.0 or round.start_training_date + timedelta(minutes=project.max_training_time) < timezone.now():

        # if enough devices reported models
        if trained_ratio >= float(project.valid_round_training_threshold):

            verbose and print("Reported models are enough (Ratio is %.2f). Round '%d' is complete." % (trained_ratio, round.round_number))

            # set round status to complete
            round.status = Round.Status.COMPLETE
            round.stop_training_date = timezone.now()
            round.save()

            # create next round
            next_round = __create_next_round(project, verbose)

            # aggregate model of last round into the next round
            sc.aggregate_model(round, next_round)

            # check if it is time to set project as complete
            if __is_project_complete(project, verbose):
                # set project status to complete
                project.status = Project.STATUS_CHOICES[2][0]
                project.save()

            else:
                # attempt training in new round
                __attempt_device_training(next_round, verbose)

        else:
            verbose and print("Time has pass. Round '%d' is invalid." % round.round_number)
            __invalidate_round(round, verbose)

    else:
        remaining_mins = (round.start_training_date + timedelta(minutes=project.max_training_time) - timezone.now()).seconds / 60
        verbose and print("Waiting for %d more minutes. Current Ratio is %.2f (%d out of %d)." % (int(remaining_mins), trained_ratio, device_train_responses, all_devices))


def __invalidate_round(round, verbose=False):

    round.status = Round.Status.INVALID
    round.stop_training_date = timezone.now()
    round.save()

    next_round = __create_next_round(round.project, verbose)

    # just copy the model
    sc.copy_model(round, next_round)

    # attempt training in new round
    __attempt_device_training(next_round, verbose)


def __create_next_round(project, verbose=False):

    next_round_number = project.current_round + 1

    verbose and print("Creating next round: %d" % (next_round_number))

    # increase counter
    project.current_round = next_round_number
    project.save()

    round = Round.objects.create(
        round_number=next_round_number,
        number_of_samples=project.number_of_samples,
        number_of_epochs=project.number_of_epochs,
        seed=project.seed,
        project=project
    )

    return round


def __attempt_device_training(round, verbose=False):

    # get project
    project = round.project

    if project.profiles.count() == 0:
        verbose and print("Attempted to train but no devices are attached to this project.")
        return

    # query responses based on the configuration on Project
    queried_responses = __query_device_status_responses(project, verbose)
    responses_ratio = len(queried_responses) / project.profiles.count()

    verbose and print("Ratio is %.2f (%d/%d)" % (responses_ratio, len(queried_responses), project.profiles.count()))

    # if ratio > threshold
    if responses_ratio >= float(project.responses_ratio_threshold):

        verbose and print("Sending train request:")

        # request device training
        devices = [response.device for response in queried_responses]
        dc.send_train_request(project, devices, verbose)

        # log these devices and set into training status
        round.requested_training_devices.set(devices)
        round.status = Round.Status.TRAINING
        round.start_training_date = timezone.now()
        round.save()

    else:
        verbose and print("Attempted to train but failed to reach the threshold.")


def tick(verbose=False):

    # get all active (In Progress) projects
    projects = Project.objects.filter(status=Project.STATUS_CHOICES[1][0])

    verbose and print("Found %d available projects." % len(projects))

    # ping / attempt to train devices in each active project
    for project in projects:

        verbose and print("Project '%d. %s':" % (project.id, project.title))

        # get last created round of that project
        last_round = project.rounds.last()

        # switch on status
        status = last_round.status
        if status == Round.Status.WAIT:
            verbose and print("Round status 'Wait': Evaluating device statuses.")

            if project.use_knowledge_distillation:
                print("Knowledge Distillation is enabled for this project.")
                # Step 1: Run KD training
                student_model = run_knowledge_distillation(batch_size=8, epochs=1)

                project_id = project.id
                round_id = 0  # for initial round

                # Step 2: Serialize the model into memory
                buffer = io.BytesIO()
                torch.save(student_model.state_dict(), buffer)
                buffer.seek(0)

                # Read model bytes once
                model_bytes = buffer.read()
                model_size_bytes = len(model_bytes)
                model_size_mb = model_size_bytes / (1024 * 1024)

                # Step 3: Construct the remote path
                remote_model_path = os.path.join(
                    consts.PROJECTS_PATH, str(project_id), str(round_id), consts.MODEL_WEIGHTS_FILENAME
                )

                # Step 4: Overwrite if exists
                print(f"Uploading the new student model as the global model: {remote_model_path}")
                if default_storage.exists(remote_model_path):
                    default_storage.delete(remote_model_path)

                # Step 5: Save the new model
                default_storage.save(remote_model_path, ContentFile(model_bytes))

                # Step 6: Log result
                print(f"KD-trained student model saved to: {remote_model_path}")
                print(f"Student model size: {model_size_bytes:,} bytes ({model_size_mb:.2f} MB)")

            __attempt_device_training(last_round, verbose)

        elif status == Round.Status.TRAINING:
            verbose and print("Round status 'Training': Checking for round completion.")
            __check_for_round_completion(last_round, verbose)

        elif status == Round.Status.COMPLETE:
            verbose and print("Round status 'Complete'.")
            # should never happen really

        elif status == Round.Status.INVALID:
            verbose and print("Round status 'Invalid'.")
            # should never happen really

        else:
            print("Unknown Round.status '%d':" % status)
            return
