import numpy as np

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from opacus.accountants.utils import get_noise_multiplier

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

        if state == "random":
            self.weights = np.random.rand(size)
        elif state == "zeros":
            self.weights = np.zeros(size)
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

    def dp_accumulate_model(self, file_path: str, model_file_path: str, clipping_norm: float, noise_type: str,) -> None:
        try:
            with default_storage.open(file_path, "rb") as f1:
                local_weights_flat = np.frombuffer(f1.read(), dtype=np.float32)

            with default_storage.open(model_file_path, "rb") as f2:
                global_weights_flat = np.frombuffer(f2.read(), dtype=np.float32)

            print("Local weights (flat):", local_weights_flat[:5])  # Print only first few values
            print("Global weights (flat):", global_weights_flat[:5])

            if len(local_weights_flat) != len(global_weights_flat):
                print("Mismatch between local and global weights.")
                return

            local_weights = [local_weights_flat.copy()]
            global_weights = [global_weights_flat.copy()]

            print("Calling compute_clip_model_update...")
            compute_clip_model_update(local_weights, global_weights, clipping_norm, noise_type)
            print("Finished compute_clip_model_update.")
            print("Clipped weights:", local_weights[0][:5])  # Again, show a small slice

            clipped_weights = local_weights[0]

            if len(clipped_weights) != len(self.weights):
                print(f"Ignoring weights with incorrect length: {len(clipped_weights)}")
                return

            self.weights += clipped_weights
            self.devices += 1

        except (OSError, IOError) as ex:
            print(f"Error while accumulating model from '{file_path}': {ex}")


    def aggregate(self):
        if self.devices > 1:
            self.weights /= self.devices
        self.devices = 0

    def dp_aggregate(self, epsilon:float, delta:float, fl_rounds:int, clipping_norm: float,
                     noise_type:str='gaussian') -> None:
        if self.devices > 1:
            self.weights /= self.devices

        noise_multiplier = privacy_accountant(epsilon, delta, fl_rounds, sampling_frac=1, noise_type=noise_type,)
        self.weights = add_noise_to_params(
            self.weights,
            noise_multiplier,
            clipping_norm,
            self.devices,
            noise_type,)
        print(f"aggregate_fit: central DP noise with {compute_stdv(noise_multiplier, clipping_norm, self.devices):.4f} stdev added")

        self.devices = 0

    def write(self, filename):
        content = ContentFile(self.weights.astype('float32').tobytes())
        default_storage.save(filename, content)


def compute_clip_model_update(
    local_model_weights: list[np.ndarray],
    global_model_weights: list[np.ndarray],
    clipping_norm: float,
    noise_type: str,
) -> None:
    """Compute update = current - previous, clip it, then apply to previous in-place."""
    model_update = [curr - prev for curr, prev in zip(local_model_weights, global_model_weights)]
    print("Model Updates:",model_update)
    clip_inputs_inplace(model_update, clipping_norm, noise_type)
    for i in range(len(local_model_weights)):
        local_model_weights[i] = global_model_weights[i] + model_update[i]

def clip_inputs_inplace(model_update: list[np.ndarray], clipping_norm: float, noise_type: str) -> None:
    """Scale model update if its total L2 norm exceeds the clipping norm."""
    if noise_type == 'gaussian':
        lp_norm = 2
    elif noise_type == 'laplace':
        lp_norm  = 1
    else:
        raise ValueError(f'noise_type {noise_type} not supported!')

    norm = get_norm(model_update, lp_norm)
    scaling_factor = min(1.0, clipping_norm / norm) if norm > 0 else 1.0
    print("Original norm:", norm)
    print("Scaling factor:", scaling_factor)
    print("Original model update[0]:", model_update[0][:5])
    for i in range(len(model_update)):
        model_update[i] *= scaling_factor
    print("Scaled model update[0]:", model_update[0][:5])


def get_norm(model_update: list[np.ndarray], lp_norm:int) -> float:
    """Compute total L1/L2 norm across all arrays."""
    return float((sum(np.linalg.norm(update.ravel())**lp_norm for update in model_update))**(1/lp_norm))

def add_noise_to_params(
    model_params: np.ndarray,
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
    noise_type: str,
) :
    """Add noise to model parameters."""
    add_noise_inplace(
        model_params,
        compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients),
        noise_type,
    )
    return model_params

def add_noise_inplace(input_array: np.ndarray, std_dev: float, noise_type: str) -> None:
    """Add noise to the entire parameter vector."""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, std_dev, input_array.shape).astype(input_array.dtype)
    elif noise_type == 'laplace':
        noise = np.random.laplace(0, std_dev, input_array.shape).astype(input_array.dtype)
    else:
        raise ValueError(f'noise_type {noise_type} not supported!')
    input_array += noise


def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    Paper: https://arxiv.org/abs/1710.06963
    """
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
    assert 0 <= target_delta < 1, f'target delta must be in [0,1], got {target_delta}'
    assert fl_rounds > 0, f'FL rounds must be > 0, got {fl_rounds}'
    assert 0 < sampling_frac <= 1, f'sampling frac must be in [0,1], got {sampling_frac}'

    if noise_type == 'gaussian':
        # use RDP accountant from Opacus with Poisson subsampling on the clients
        # assume sensitivity = 1 and rescale in dp_module when doing clipping
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, max_alpha))
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
