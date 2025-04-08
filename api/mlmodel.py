import numpy as np

from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


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

    def dp_accumulate_model(self, file_path: str, model_file_path: str, clipping_norm: float) -> None:
        try:
            with default_storage.open(file_path, "rb") as f1:
                local_weights_flat = np.frombuffer(f1.read(), dtype=np.float32)

            with default_storage.open(model_file_path, "rb") as f2:
                global_weights_flat = np.frombuffer(f2.read(), dtype=np.float32)

            if len(local_weights_flat) != len(global_weights_flat):
                print("Mismatch between current and previous weights.")
                return

            # Convert to NDArrays (lists of arrays), here treating whole model as single layer
            local_weights = [local_weights_flat.copy()]
            global_weights = [global_weights_flat.copy()]

            compute_clip_model_update(local_weights, global_weights, clipping_norm)

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

    def write(self, filename):
        content = ContentFile(self.weights.astype('float32'))
        default_storage.save(filename, content)


def compute_clip_model_update(
    local_model_weights: list[np.ndarray],
    global_model_weights: list[np.ndarray],
    clipping_norm: float,
) -> None:
    """Compute update = current - previous, clip it, then apply to previous in-place."""
    model_update = [curr - prev for curr, prev in zip(local_model_weights, global_model_weights)]
    print("Model Updates:",model_update)
    clip_inputs_inplace(model_update, clipping_norm)
    for i in range(len(local_model_weights)):
        local_model_weights[i] = global_model_weights[i] + model_update[i]

def clip_inputs_inplace(model_update: list[np.ndarray], clipping_norm: float) -> None:
    """Scale model update if its total L2 norm exceeds the clipping norm."""
    norm = get_norm(model_update)
    scaling_factor = min(1.0, clipping_norm / norm) if norm > 0 else 1.0
    for i in range(len(model_update)):
        model_update[i] *= scaling_factor

def get_norm(model_update: list[np.ndarray]) -> float:
    """Compute total L2 norm across all arrays."""
    return float(np.sqrt(sum(np.linalg.norm(update.ravel())**2 for update in model_update)))

