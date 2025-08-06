import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader

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

    pytorch_model = MobileNetV2_PoolFC()
    print("Evaluating model...")
    pytorch_model = load_weights_to_model(pytorch_model, model.weights)

    try:
        accuracy = evaluate_test_accuracy(pytorch_model, batch_size=64, fraction=0.1)
        # Print accuracy for immediate feedback
        print(f"Aggregated model test accuracy: {accuracy:.4f}")
    except Exception:
        # If evaluation fails, continue without halting training.  The
        # error will already have been logged by ``evaluate_test_accuracy``.
        pass

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
    print("Evaluating model...")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Use transforms matching pretrained weights
    # weights = models.MobileNet_V2_Weights.DEFAULT
    # transform = weights.transforms()

    # Prepare CIFAR‑10 test loader
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2023, 0.1994, 0.2010])
    ])

    # Load dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if 0 < fraction < 1.0:
        subset_size = int(len(test_dataset) * fraction)
        test_dataset = torch.utils.data.Subset(test_dataset, range(subset_size))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader, 1):
            images, labels = images.to(device), labels.to(device)
            try:
                outputs = model(images)
            except Exception as e:
                print(f"[ERROR] During inference at batch {batch_idx}: {e}")
                continue
            _, predicted = outputs.max(1)

            batch_correct = (predicted == labels).sum().item()
            batch_total = labels.size(0)
            batch_accuracy = batch_correct / batch_total

            print(f"[Batch {batch_idx}] Accuracy: {batch_accuracy:.4f}")

            correct += batch_correct
            total += batch_total

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
        base_model = models.mobilenet_v2(pretrained=True)
        self.features = base_model.features  # output: (B, 1280, 7, 7)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)        # (B, 1280, 7, 7)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)   # flatten to (B, 62720)
        x = self.fc(x)              # → (B, 10)
        return x

    def flatten_weights(self) -> np.ndarray:
        return np.concatenate([
            self.fc.weight.detach().cpu().numpy().flatten(),
            self.fc.bias.detach().cpu().numpy().flatten()
        ])


class MobileNetV2_ExpandedFC(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # Load pretrained chenyaofo MobileNetV2 for CIFAR-10
        base = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_mobilenetv2_x1_4",
            pretrained=pretrained
        )

        self.backbone = nn.Sequential(*list(base.children())[:-1])  # Remove the original classifier

        # New classifier: 28672 → 62720 → 10
        self.fc1 = nn.Linear(28672, 62720)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(62720, num_classes)
        # Initialize fc1 with Xavier and zero bias
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = self.backbone(x)           # (B, 28672)
        x = x.view(x.size(0), -1)      # Flatten
        x = self.fc1(x)                # → (B, 62720)
        x = self.relu(x)
        x = self.fc2(x)                # → (B, 10)
        return x

    def flatten_weights(self) -> np.ndarray:
        return np.concatenate([
            self.fc2.weight.detach().cpu().numpy().flatten(),
            self.fc2.bias.detach().cpu().numpy().flatten()
        ])

class MobileNetV2_PoolFC(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # Load pretrained chenyaofo MobileNetV2 for CIFAR-10
        base = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_mobilenetv2_x1_4",
            pretrained=pretrained
        )

        # Use backbone (exclude original classifier)
        self.features = base.features

        # Determine flattened feature size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)  # CIFAR-10 image size
            dummy_output = self.features(dummy_input)
            flattened_dim = dummy_output.view(1, -1).shape[1]
            print(f"{dummy_output.shape}->{flattened_dim}")

        # Custom classifier
        self.fc1 = nn.Linear(flattened_dim, 62720)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(62720, num_classes)

        # Initialize fc1 with Xavier and zero bias
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = self.features(x)          # → [B, C, H, W]
        x = torch.flatten(x, 1)       # → [B, C*H*W]
        x = self.fc1(x)               # → [B, 62720]
        x = self.relu(x)
        x = self.fc2(x)               # → [B, num_classes]
        return x

    def flatten_weights(self) -> np.ndarray:
        return np.concatenate([
            self.fc2.weight.detach().cpu().numpy().flatten(),
            self.fc2.bias.detach().cpu().numpy().flatten()
        ])


def load_weights_to_model(pytorch_model: torch.nn.Module, flat_weights: np.ndarray):
    """
    Load a flat weight vector into only the classifier head (fc2) of a PyTorch model.

    Args:
        pytorch_model: instance of MobileNetV2_CIFAR10
        flat_weights: 1D NumPy array containing only fc2 parameters, in order
    """
    assert isinstance(flat_weights, np.ndarray) and flat_weights.ndim == 1

    expected_size = pytorch_model.flatten_weights().size
    print(f"expected_size: {expected_size}")
    print(f"flat_weights.shape: {flat_weights.size}")

    pointer = 0
    new_state_dict = {}

    # Collect the layers you want to replace
    target_layers = ['fc2.weight', 'fc2.bias']
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