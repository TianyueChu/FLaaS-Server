import os

from django.core.files.storage import default_storage

from api.mlmodel import MLModel
from api.mlreport import MLReport
# from api.produce_plots import plot
from api.libs.filemanagement import filecopy, delfolder
from api.libs import consts
from api.libs.helper_client import aggregate_via_helper

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time
import numpy as np


def aggregate_model(round, into_round):
    # get project
    project = round.project

    # init model with zeros (to append weights during aggregation)
    model = MLModel(consts.MODEL_SIZE, "zeros")

    # round level
    round_path = os.path.join(consts.PROJECTS_PATH, str(project.id), str(round.round_number))

    round_model_path = os.path.join(round_path, consts.MODEL_WEIGHTS_FILENAME)

    dp_type = project.DP_used

    delta = project.delta

    epsilon = project.epsilon

    use_split_learning = project.use_split_learning

    num_samples = project.number_of_samples

    fl_round = project.number_of_rounds

    # get all devices that reported data (weights)
    reported_devices = [response.device for response in round.device_train_request.device_train_responses.all()]

    # Check if helper aggregation is enabled
    if project.use_helper:
        print("Using helper")
        BASE_PORT = 8500
        MAX_WORKERS = 8
        HELPER_GROUP_SIZE = 5

        # Step 1: Collect model updates
        updates = []

        # Load weights from devices
        for device in reported_devices:
            file_path = os.path.join(round_path, str(device.id), consts.MODEL_WEIGHTS_FILENAME)
            with default_storage.open(file_path, "rb") as f:
                weights = np.frombuffer(f.read(), dtype=np.float32)
                updates.append(weights.tolist())  # Convert to JSON-serializable format

        # Divide updates into groups
        num_helpers = math.ceil(len(updates) / HELPER_GROUP_SIZE)
        grouped_updates = [
            updates[i * HELPER_GROUP_SIZE:(i + 1) * HELPER_GROUP_SIZE] for i in range(num_helpers)
        ]
        group_sizes = [len(group) for group in grouped_updates]

        print(f'group size: {len(grouped_updates[0][0])}')

        # Step 3: Launch helpers in parallel and collect partial aggregates
        print("Launching helpers in parallel...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            all_aggs = []
            for i, group in enumerate(grouped_updates):
                port = BASE_PORT + i
                start_time = time.time()
                print(f"Submitting group {i + 1}/{num_helpers} to helper on port {port}")
                future = executor.submit(aggregate_via_helper, group, use_split_learning, port)
                futures[future] = (i, start_time, group_sizes[i], port)

            for future in as_completed(futures):
                group_index, start_time, group_size, port = futures[future]
                end_time = time.time()
                elapsed = end_time - start_time

                try:
                    result = future.result()
                    agg = np.array(result, dtype=np.float32)
                    print(f"Helper {group_index + 1} (port {port}, size {group_size}) finished in {elapsed:.2f} seconds")
                    print(f'Aggregation shape: {agg.shape}')
                    all_aggs.append(agg)
                except Exception as e:
                    print(f"[ERROR] Helper {group_index + 1} (port {port}) failed after {elapsed:.2f}s: {e}")

            if not all_aggs:
                raise RuntimeError("No successful aggregation from helpers.")

            ## aggregate the results from all the helpers
            all_aggs = np.mean(all_aggs, axis=0).tolist()

            if dp_type == "Central DP":
                model.dp_accumulate_model_helper(all_aggs, round_model_path, clipping_norm=0.2, noise_type="gaussian")
            else:
                model.accumulate_model_helper(all_aggs)

        if dp_type == "Central DP":
            model.dp_aggregate(delta, epsilon, fl_round, clipping_norm=0.2, noise_type="gaussian")
        else:
            # aggregate weights
            model.aggregate()
        print("Helper-based aggregation complete.")

    else:
        for device in reported_devices:
            # find device weights
            file_path = os.path.join(round_path, str(device.id), consts.MODEL_WEIGHTS_FILENAME)

            # Use Split Learning
            if use_split_learning:
                model.use_split_learning(file_path, num_samples)

            # Use Central DP
            if dp_type == "Central DP":
                model.dp_accumulate_model(file_path, round_model_path, clipping_norm=0.2, noise_type="gaussian")
            # use Local DP or Do not use DP
            else:
                model.accumulate_model(file_path)

        if dp_type == "Central DP":
            model.dp_aggregate(delta, epsilon, fl_round, clipping_norm=0.2, noise_type="gaussian")
        else:
            # aggregate weights
            model.aggregate()

    # write model into round folder
    file_path = os.path.join(consts.PROJECTS_PATH, str(project.id), str(into_round.round_number), consts.MODEL_WEIGHTS_FILENAME)
    model.write(file_path)

    # TODO: Enable once server-based eval is implemented
    # # compute required list of result filenames (from number of apps)
    # result_filenames_list = []
    # if project.apps > 1:
    #     for i in range(project.apps):
    #         result_filenames_list.append("performance_app_%d_eval_results.csv" % (i + 1))
    # result_filenames_list.append("performance_app_0_eval_results.csv")  # 0 always exists as main model

    # # produce/update round plot
    # path = os.path.join(consts.PROJECTS_PATH, str(project.id))
    # plot(path, result_filenames_list, project.title, consts.RESULTS_FIGURE_FILENAME)


def copy_model(round, into_round):

    # get project
    project = round.project

    original_model = os.path.join(consts.PROJECTS_PATH, str(project.id), str(round.round_number), consts.MODEL_WEIGHTS_FILENAME)
    new_model = os.path.join(consts.PROJECTS_PATH, str(project.id), str(into_round.round_number), consts.MODEL_WEIGHTS_FILENAME)
    filecopy(original_model, new_model)


def delete_project(project):

    # project level
    path = os.path.join(consts.PROJECTS_PATH, str(project.id))

    # delete project folder
    delfolder(path)


def reset_project(project):

    delete_project(project)

    # project level
    path = os.path.join(consts.PROJECTS_PATH, str(project.id))

    # Copy the appropriate model into the project_path
    filecopy(
        os.path.join(consts.MODELS_PATH, project.model + '.bin'),
        os.path.join(path, '0', 'model_weights.bin'))

    # reset counters
    project.current_round = 0
    project.save()


# TODO: Refactor and enable this function
def __report_results(project, round):

    # round level
    path = os.path.join(consts.PROJECTS_PATH, str(project.id), str(round))

    # init empty list
    accuracy_list = []

    # list sessions (folders, not files)
    (sessions, _) = default_storage.listdir(path)
    for session in sessions:

        # build file path
        file_path = os.path.join(path, session, consts.EVAL_RESULTS_FILENAME)

        # report
        report = MLReport(file_path)
        accuracy = report.get_accuracy()
        accuracy_list.append(accuracy)

    np_list = np.array(accuracy_list)

    print("\t>> Round %d accuracy: %.4f (%.4f)" % (project.round_counter, np_list.mean(), np_list.std()))


def __delete_round_models(project, round):

    # round level
    path = os.path.join(consts.PROJECTS_PATH, str(project.id), str(round))

    # list sessions (folders, not files)
    (sessions, _) = default_storage.listdir(path)
    for session in sessions:

        # build session path
        session_path = os.path.join(path, session)

        # delete model if exists
        file_path = os.path.join(session_path, consts.MODEL_WEIGHTS_FILENAME)
        if default_storage.exists(file_path):
            default_storage.delete(file_path)
