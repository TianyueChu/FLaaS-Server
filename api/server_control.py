import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights

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
                time.sleep(0.1)  # optional to reduce contention

            for future in as_completed(futures):
                group_index, start_time, group_size, port = futures[future]
                end_time = time.time()
                elapsed = end_time - start_time
                try:
                    result = future.result()
                    agg = np.array(result, dtype=np.float32)
                    print(f"Helper {group_index + 1} (port {port}, size {group_size}) finished in {elapsed:.2f} seconds")
                    all_aggs.append(agg)
                except Exception as e:
                    print(f"[ERROR] Helper {group_index + 1} (port {port}) failed after {elapsed:.2f}s: {e}")

            if not all_aggs:
                print("No successful aggregation from helpers.")

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

    # --------------------------------------------------------------------------
    # New evaluation step: compute and record test accuracy
    # --------------------------------------------------------------------------

    pytorch_model = MobileNetV2_FC()
    load_fc_from_flat_vector(pytorch_model, model.weights)

    # pytorch_model = load_weights_to_model(pytorch_model, model.weights)

    # try:
        # accuracy = evaluate_test_accuracy(pytorch_model, batch_size=32, fraction=0.01)
    # except Exception:
        # If evaluation fails, continue without halting training.  The
        # error will already have been logged by ``evaluate_test_accuracy``.
        # pass

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

def evaluate_test_accuracy(model: torch.nn.Module, batch_size=64, fraction=1.0) -> float:
    """
    Evaluate the given PyTorch model on the CIFAR-10 test set.

    Args:
        model: A PyTorch model (e.g., MobileNetV2_FC) ready for inference.
        batch_size: Batch size for evaluation.
        fraction: Fraction of test dataset to use (0 < fraction ≤ 1).

    Returns:
        Test accuracy as a float between 0 and 1.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use transforms matching pretrained weights
    weights = models.MobileNet_V2_Weights.DEFAULT
    transform = weights.transforms()

    # transform = transforms.Compose([
    #    transforms.Resize((224, 224)),
    #    transforms.ToTensor(),
    #    transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225])
    # ])

    # Load dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if 0 < fraction < 1.0:
        subset_size = int(len(test_dataset) * fraction)
        test_dataset = torch.utils.data.Subset(test_dataset, range(subset_size))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Evaluating model...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy: {:.2f}%'.format(100 * correct / total))

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


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


class MobileNetV2_FC(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # MobileNetV2 backbone (exclude classifier head)
        # base_model = models.mobilenet_v2(pretrained=True)
        weights = MobileNet_V2_Weights.DEFAULT
        base_model = models.mobilenet_v2(weights=weights)
        self.features = base_model.features  # output: (B, 1280, 7, 7)
        self.fc = nn.Linear(1280 * 7 * 7, num_classes)

        # Initialize fc1 with Xavier and zero bias
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.features(x)        # (B, 1280, 7, 7)
        x = x.view(x.size(0), -1)   # flatten to (B, 62720)
        x = self.fc(x)              # → (B, 10)
        return x

    def flatten_weights(self) -> np.ndarray:
        return np.concatenate([
            self.fc.weight.detach().cpu().numpy().flatten(),
            self.fc.bias.detach().cpu().numpy().flatten()
        ])


def load_weights_to_model(pytorch_model: torch.nn.Module, flat_weights: np.ndarray):
    """
    Load a flat weight vector into only the classifier head of a PyTorch model.

    Args:
        pytorch_model: instance of MobileNetV2_CIFAR10
        flat_weights: 1D NumPy array containing only fc parameters, in order
    """
    assert isinstance(flat_weights, np.ndarray) and flat_weights.ndim == 1

    expected_size = pytorch_model.flatten_weights().size
    print(f"expected_size: {expected_size}")
    print(f"flat_weights.shape: {flat_weights.size}")

    pointer = 0
    new_state_dict = {}

    # Collect the layers you want to replace
    target_layers = ['fc.weight', 'fc.bias']
    state_dict = pytorch_model.state_dict()

    for name in target_layers:
        if name not in state_dict:
            raise KeyError(f"Layer {name} not found in model.")

        param = state_dict[name]
        numel = param.numel()

        if pointer + numel > flat_weights.size:
            raise ValueError(
                f"[ERROR] Not enough weights for {name} (needs {numel}, has {flat_weights.size - pointer})"
            )

        # Load and reshape
        weight_slice = flat_weights[pointer:pointer + numel]
        reshaped = torch.tensor(weight_slice.reshape(param.shape), dtype=param.dtype)
        new_state_dict[name] = reshaped
        pointer += numel

    if pointer != flat_weights.size:
        raise ValueError(f"[ERROR] Unused weights left: used {pointer}, total {flat_weights.size}")

    # Load only those layers
    pytorch_model.load_state_dict(new_state_dict, strict=False)
    print("[SUCCESS] weights loaded into model.")

    return pytorch_model

def load_fc_from_flat_vector(model: nn.Module, flat_w):
    # flat_w: 1D torch.Tensor or np.ndarray
    if isinstance(flat_w, np.ndarray):
        flat_w = torch.from_numpy(flat_w)
    flat_w = flat_w.float().clone()  # make sure float32

    in_f = model.fc.in_features  # 62720 for your head
    out_f = model.fc.out_features  # 10 for CIFAR-10
    expected = in_f * out_f + out_f
    assert flat_w.numel() == expected, f"Size mismatch: got {flat_w.numel()}, expected {expected}"

    W = flat_w[:in_f * out_f].view(out_f, in_f)  # PyTorch Linear: [out, in]
    b = flat_w[in_f * out_f:]

    # Load to model's device
    dev = next(model.parameters()).device
    model.fc.weight.data.copy_(W.to(dev))
    model.fc.bias.data.copy_(b.to(dev))

    print("Loaded FC:",
          f"W shape={tuple(model.fc.weight.shape)} std={model.fc.weight.std().item():.4f}",
          f"b shape={tuple(model.fc.bias.shape)} std={model.fc.bias.std().item():.4f}")
