import numpy as np

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from opacus.accountants.utils import get_noise_multiplier

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

class MLModel:

    def __init__(self, size, state, initial_privacy_budget=1.0):
        """
        Initialize the ML model with the given size and privacy budget.

        Parameters:
        - size (int): The size of the model's weights vector.
        - state (str): The initial state of the weights. Can be "random" or "zeros".
        - initial_privacy_budget (float): Initial amount of privacy budget.
        """

        self.devices = 0
        self.initial_privacy_budget = initial_privacy_budget
        self.remaining_privacy_budget = initial_privacy_budget
        self.size = size

        if state == "random":
            self.weights = np.random.rand(size).astype(np.float32)
        elif state == "zeros":
            self.weights = np.zeros(size, dtype=np.float32)
        else:
            raise Exception("Unknown state:" + state)

    def accumulate_model(self, file_path):
        try:
            with default_storage.open(file_path) as data:
                weights = np.frombuffer(data.read(), dtype=np.float32)
                if len(weights) == len(self.weights):
                    self.weights += weights
                    self.devices += 1
                else:
                    print("Ignoring weights with incorrect length: %d" % len(weights))
        except (OSError, IOError) as ex:
            print("Something went wrong while accumulating model at '%s'" % file_path)
            print(ex)

    def accumulate_model_helper(self, agg: np.ndarray) -> None:
        try:
            if len(agg) == len(self.weights):
                self.weights += agg
                self.devices += 1
            else:
                print(f"Ignoring weights with incorrect length: {len(agg)} (expected {len(self.weights)})")
        except Exception as ex:
            print(f"Error in accumulate_model_helper: {ex}")

    def dp_accumulate_model(self, file_path: str, model_file_path: str, clipping_norm: float,
                            noise_type: str, ) -> None:
        try:
            with default_storage.open(file_path, "rb") as f1:
                local_weights_flat = np.frombuffer(f1.read(), dtype=np.float32)

            with default_storage.open(model_file_path, "rb") as f2:
                global_weights_all = np.frombuffer(f2.read(), dtype=np.float32)

            if global_weights_all.shape[0] > self.size:
                print("Loaded full model weights, but size exceeds expected limit — resetting to zeros.")
                global_weights_all = np.zeros(self.size, dtype=np.float32)

            global_weights_flat = np.nan_to_num(global_weights_all, nan=0.0)

            local_weights = local_weights_flat.copy()
            global_weights = global_weights_flat.copy()

            inspect_weights("local_weights (before clip)", local_weights)
            inspect_weights("global_weights", global_weights)

            clipped_weights = compute_clip_model_update(local_weights, global_weights, clipping_norm, noise_type)

            if len(clipped_weights) != len(self.weights):
                print(f"Ignoring weights with incorrect length: {len(clipped_weights)}")
                return

            self.weights += clipped_weights
            self.devices += 1

        except (OSError, IOError) as ex:
            print(f"Error while accumulating model from '{file_path}': {ex}")

    def dp_accumulate_model_helper(self, agg: np.ndarray, model_file_path: str,
                                   clipping_norm: float, noise_type: str) -> None:
        try:
            with default_storage.open(model_file_path, "rb") as f:
                global_weights_all = np.frombuffer(f.read(), dtype=np.float32)

            if global_weights_all.shape[0] > self.size:
                print("Loaded full model weights, but size exceeds expected limit — resetting to zeros.")
                global_weights_all = np.zeros(self.size, dtype=np.float32)

            global_weights = np.nan_to_num(global_weights_all, nan=0.0)
            local_weights = np.array(agg, dtype=np.float32)

            inspect_weights("local_weights (before clip)", local_weights)
            inspect_weights("global_weights", global_weights)

            clipped_weights = compute_clip_model_update(local_weights, global_weights, clipping_norm, noise_type)

            if len(clipped_weights) != len(self.weights):
                print(f"Ignoring weights with incorrect length: {len(clipped_weights)}")
                return

            self.weights += clipped_weights
            self.devices += 1

        except (OSError, IOError) as ex:
            print(f"Error while accumulating model from helper: {ex}")


    def use_split_learning(self, file_path, num_samples):
        try:
            with default_storage.open(file_path) as data:
                weights = np.frombuffer(data.read(), dtype=np.float32)
                ## hard-coded here
                # number of samples is manually set in the device side, also the model size and number of class
                len_bottleneck = num_samples * 62720
                bottleneck =  weights[:len_bottleneck].reshape((num_samples, 62720))
                labels = weights[len_bottleneck:].reshape((num_samples, 10))
                labels_indices = np.argmax(labels, axis=1)  # Convert one-hot to class indices

                bottleneck_tensor = torch.tensor(bottleneck.reshape((-1, 7, 7, 1280)), dtype=torch.float32)
                labels_tensor = torch.tensor(labels_indices, dtype=torch.long)

                model = TrainHead()
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                num_epochs = 10

                for epoch in range(num_epochs):
                    model.train()
                    optimizer.zero_grad()

                    outputs = model(bottleneck_tensor)
                    loss = criterion(outputs, labels_tensor)

                    loss.backward()
                    optimizer.step()

                    print(f"SL Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

                    preds = torch.argmax(outputs, dim=1)
                    accuracy = (preds == labels_tensor).float().mean().item()
                    print(f"SL training accuracy: {accuracy:.2%}")

                # Save the trained weights
                fc_weight = model.fc.weight.data.numpy()  # shape: (10, 62720)
                fc_bias = model.fc.bias.data.numpy()  # shape: (10,)

                # Transpose weight to match client layout: [62720, 10]
                fc_weight_tflite = fc_weight.T  # shape: (62720, 10)

                # Flatten both
                flat_weights = fc_weight_tflite.flatten()  # length: 627200
                flat_bias = fc_bias.flatten()  # length: 10

                # Concatenate
                combined = np.concatenate([flat_weights, flat_bias]).astype(np.float32)  # length: 627210

                # Save to file using default_storage
                byte_data = combined.tobytes()
                default_storage.save(file_path, ContentFile(byte_data))
                print(f"Weights saved to {file_path} ({len(byte_data)} bytes)")

        except (OSError, IOError) as ex:
            print(ex)

    def aggregate(self):
        if self.devices > 1:
            self.weights /= self.devices
        self.devices = 0

    def dp_aggregate(self, epsilon: float, delta: float, fl_rounds: int, clipping_norm: float,
                     noise_type: str = 'gaussian') -> None:
        if self.devices > 1:
            self.weights /= self.devices

        noise_multiplier = privacy_accountant(epsilon, delta, fl_rounds, sampling_frac=1, noise_type=noise_type)
        print("Noise multiplier:", noise_multiplier)

        self.weights = add_noise_to_params(
            self.weights,
            noise_multiplier,
            clipping_norm,
            self.devices,
            noise_type, )

        print(
            f"aggregate_fit: central DP noise with {compute_stdv(noise_multiplier, clipping_norm, self.devices):.4f} stdev added")

        self.weights = np.nan_to_num(self.weights, nan=0.0, posinf=0.0, neginf=0.0)
        # Estimate from one of the client's original weights
        expected_std = 0.00034 # or estimate dynamically
        expected_mean = 0.0  # or estimate from previous global model

        agg = self.weights
        agg = (agg - np.mean(agg)) / np.std(agg)  # Normalize to 0 mean, 1 std
        agg = agg * expected_std + expected_mean  # Rescale
        self.weights = agg.astype(np.float32)

        inspect_weights("aggregated weights", self.weights)

        self.devices = 0

    def write(self, filename):
        content = ContentFile(self.weights.astype('float32').tobytes())
        default_storage.save(filename, content)


def compute_clip_model_update(
        local_model_weights: np.ndarray,
        global_model_weights: np.ndarray,
        clipping_norm: float,
        noise_type: str,
) -> np.ndarray:
    """Compute update = current - previous, clip it, then apply to previous in-place."""
    model_update = local_model_weights - global_model_weights
    clip_inputs_inplace(model_update, clipping_norm, noise_type)
    for i in range(len(local_model_weights)):
        local_model_weights[i] = global_model_weights[i] + model_update[i]
    return local_model_weights


def clip_inputs_inplace(model_update: np.ndarray, clipping_norm: float, noise_type: str) -> None:
    """Scale model update if its total L1/L2 norm exceeds the clipping norm."""
    if noise_type == 'gaussian':
        lp_norm = 2
    elif noise_type == 'laplace':
        lp_norm = 1
    else:
        raise ValueError(f'noise_type {noise_type} not supported!')

    norm = float(np.linalg.norm(np.nan_to_num(model_update.ravel(), nan=0.0, posinf=0.0, neginf=0.0), ord=lp_norm))
    print("Original norm:", norm)
    scaling_factor = min(1.0, clipping_norm / norm) if norm > 0 else 1.0
    print("Scaling factor:", scaling_factor)
    model_update *= scaling_factor


def add_noise_to_params(
        model_params: np.ndarray,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
        noise_type: str,
):
    """Add noise to model parameters."""
    add_noise_inplace(
        model_params,
        compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients),
        noise_type,
    )
    return model_params


def add_noise_inplace(input_array: np.ndarray, std_dev: float, noise_type: str) -> None:
    """Add noise to the entire parameter vector."""
    if input_array.dtype != np.float32:
        input_array = input_array.astype(np.float32)

    if noise_type == 'gaussian':
        # noise = np.random.normal(0, std_dev, input_array.shape).astype(input_array.dtype)
        noise = np.random.normal(0, std_dev, input_array.shape).astype(np.float64)
    elif noise_type == 'laplace':
        # noise = np.random.laplace(0, std_dev, input_array.shape).astype(input_array.dtype)
        noise = np.random.laplace(0, std_dev, input_array.shape).astype(np.float64)
    else:
        raise ValueError(f'noise_type {noise_type} not supported!')
    input_array += noise
    input_array = input_array.astype(np.float32)


def compute_stdv(
        noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    Paper: https://arxiv.org/abs/1710.06963
    """
    if num_sampled_clients == 0:
        print("[WARNING] compute_stdv: number of sampled clients is 0. Returning stdv = 0.0")
        return 0.0
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)


def privacy_accountant(target_epsilon: float, target_delta: float, fl_rounds: int, sampling_frac: float,
                       noise_type: str = 'gaussian', max_alpha: int = 128) -> float:
    """
    Privacy accounting to change from (epsilon, delta) to noise sigma, wraps Opacus for ADP.
    Args:
        target_epsilon: DP epsilon
        target_delta: DP delta, should be 0 for pure DP (laplace noise).
        fl_rounds: total number of FL rounds
        sampling_frac: probability of including a given client in an FL round (Poisson sampling with constant probability); this does not affect pure DP (delta=0), which should have sampling_frac == 1 in accounting.
        noise_type: type of noise to be used, supports 'gaussian' or 'laplace'; gaussian noise uses RDP accounting from Opacus, Laplace assumes pure DP
        max_alpha: maximum value of alpha to be used in RDP accounting; try increasing this is if encountering errors from RDP accountant, larger value can be more accurate, but also slower
    Returns:
        noise_sigma: DP noise parameter
    """
    assert target_epsilon > 0, f'target epsilon must be > 0, got {target_epsilon}'
    assert 0 <= target_delta <= 1, f'target delta must be in [0,1], got {target_delta}'
    assert fl_rounds > 0, f'FL rounds must be > 0, got {fl_rounds}'
    assert 0 < sampling_frac <= 1, f'sampling frac must be in [0,1], got {sampling_frac}'

    if noise_type == 'gaussian':
        # use RDP accountant from Opacus with Poisson subsampling on the clients
        # assume sensitivity = 1 and rescale in dp_module when doing clipping
        alphas = [1 + x / 100.0 for x in range(1, 100)] + [1.5 + x / 10.0 for x in range(0, 50)] + list(range(12, max_alpha))
        noise_sigma = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=target_delta, steps=fl_rounds,
                                           sample_rate=sampling_frac, accountant='rdp', alphas=alphas)

    elif noise_type == 'laplace':
        assert target_delta == 0, f'Got delta != 0 with laplace noise. Pure DP with laplace noise has delta == 0!'
        assert sampling_frac == 1., f'Got sampling fraction != 1 with laplace noise. Pure DP with laplace noise does NOT benefit from subsampling!'
        # noise_sigma for laplace: for each step, scale = sensitivity/(eps/n_comps); here assume sensitivity = 1 and rescale in dp_module when doing clipping
        noise_sigma = fl_rounds / target_epsilon
    else:
        raise ValueError(f'noise_type {noise_type} not supported!')

    return noise_sigma


def inspect_weights(name: str, weights: np.ndarray):
    print(f"--- Inspecting: {name} ---")
    print(f"Shape: {weights.shape}")
    print(f"Type: {type(weights)}")
    print(f"Dtype: {weights.dtype}")
    print(f"Any NaN? {np.isnan(weights).any()}")
    print(f"Any Inf? {np.isinf(weights).any()}")
    print(f"Max value: {np.max(weights)}")
    print(f"Min value: {np.min(weights)}")
    print(f"Mean: {np.mean(weights)}")
    print(f"Std dev: {np.std(weights)}")
    print("-" * 40)


class TrainHead(nn.Module):
    def __init__(self, input_channels=1280, bottleneck_size=7, num_classes=10):
        super(TrainHead, self).__init__()
        self.flatten_dim = input_channels * bottleneck_size * bottleneck_size  # 62720
        self.fc = nn.Linear(self.flatten_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        # Xavier uniform for weights, zeros for bias
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # Expecting input shape: [B, 7, 7, 1280] → permute to [B, 1280, 7, 7]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), -1)   # Flatten to [B, 62720]
        return self.fc(x)