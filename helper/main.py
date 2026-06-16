from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os, resource, pathlib

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:
    _PROC = None
app = FastAPI()

class AggregationRequest(BaseModel):
    updates: List[List[float]]  # List of updates from clients (each is a list of floats)
    use_split_learning: bool
    dataset: str = 'CIFAR10'

@app.post("/aggregate")
async def aggregate(request: AggregationRequest):
    _mem_snapshot("before-parse")
    updates = np.array(request.updates, dtype=np.float32)  # shape: (num_clients, model_size)
    _mem_snapshot("after-parse")
    _cg_peak()

    logger.info(f"number of the client: {len(updates)}")

    if not request.use_split_learning:
        # Simple averaging
        aggregated = np.mean(updates, axis=0).tolist()
    else:
        # Split Learning aggregation
        logger.info(f"using split learning: {request.use_split_learning} (dataset={request.dataset})")
        sl_outputs = []
        for update in updates:
            logger.info(f"size of the update: {update.size}")
            processed = use_split_learning(update, dataset=request.dataset)
            if processed is not None:
                logger.info(f"size of the SL result: {processed.size}")
                sl_outputs.append(processed)
            else:
                logger.warning("Split learning result is None — skipping")

        if not sl_outputs:
            logger.warning("No valid updates received for split learning.")
            return {"error": "No valid updates received for split learning."}
        else:
            logger.info("valid updates received for split learning")
        aggregated = np.mean(np.array(sl_outputs), axis=0).tolist()

    _mem_snapshot("before-return")
    _cg_peak()

    return {"aggregated": aggregated}


def use_split_learning(update, num_samples=150, dataset='CIFAR10'):
    weights = np.array(update, dtype=np.float32)

    # Dataset-specific bottleneck size and number of classes:
    #   CIFAR10: bottleneck = 1280*7*7 = 62720, num_classes = 10
    #   WuW:     bottleneck = 576,             num_classes = 2
    if dataset == 'WuW':
        bottleneck_size = 576
        num_classes = 2
        bottleneck_shape = (-1, 1, 576)
        spatial_input = False
    else:
        bottleneck_size = 62720
        num_classes = 10
        bottleneck_shape = (-1, 7, 7, 1280)
        spatial_input = True

    len_bottleneck = num_samples * bottleneck_size

    try:
        bottleneck = weights[:len_bottleneck].reshape((num_samples, bottleneck_size))
        labels = weights[len_bottleneck:].reshape((num_samples, num_classes))
    except ValueError:
        logger.warning("Shape mismatch in split learning update, skipping")
        return None

    labels_indices = np.argmax(labels, axis=1)  # Convert one-hot to class indices

    bottleneck_tensor = torch.tensor(bottleneck.reshape(bottleneck_shape), dtype=torch.float32)
    labels_tensor = torch.tensor(labels_indices, dtype=torch.long)

    model = TrainHead(flatten_dim=bottleneck_size, num_classes=num_classes,
                      spatial_input=spatial_input)
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

    # Save the trained weights
    fc_weight = model.fc.weight.data.numpy()  # shape: (num_classes, bottleneck_size)
    fc_bias = model.fc.bias.data.numpy()      # shape: (num_classes,)
    fc_weight_tflite = fc_weight.T            # shape: (bottleneck_size, num_classes)

    flat_weights = fc_weight_tflite.flatten()
    flat_bias = fc_bias.flatten()

    combined = np.concatenate([flat_weights, flat_bias]).astype(np.float32)
    return combined


class TrainHead(nn.Module):
    def __init__(self, input_channels=1280, bottleneck_size=7, num_classes=10,
                 flatten_dim=None, spatial_input=True):
        super(TrainHead, self).__init__()
        if flatten_dim is None:
            flatten_dim = input_channels * bottleneck_size * bottleneck_size  # 62720
        self.flatten_dim = flatten_dim
        self.spatial_input = spatial_input
        self.fc = nn.Linear(self.flatten_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        if self.spatial_input:
            x = x.permute(0, 3, 1, 2)         # [B, 7, 7, 1280] → [B, 1280, 7, 7]
        x = x.reshape(x.size(0), -1)          # Flatten to [B, flatten_dim]
        return self.fc(x)


def _mem_snapshot(tag: str = ""):
    # ru_maxrss: on Linux it's in KiB; in Docker you'll be on Linux.
    ru = resource.getrusage(resource.RUSAGE_SELF)
    maxrss_mb = ru.ru_maxrss / 1024.0
    rss_mb = _PROC.memory_info().rss / (1024**2) if _PROC else -1
    print(f"[MEMPROC]{'['+tag+']' if tag else ''} rss_mb={rss_mb:.1f} maxrss_mb={maxrss_mb:.1f}")

def _cg_peak():
    # cgroup v2: /sys/fs/cgroup/memory.peak ; v1: memory.max_usage_in_bytes
    p2 = pathlib.Path("/sys/fs/cgroup/memory.peak")
    p1 = pathlib.Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes")
    try:
        if p2.exists():
            val = int(p2.read_text().strip() or "0")
        elif p1.exists():
            val = int(p1.read_text().strip() or "0")
        else:
            return
        print(f"[MEMCG] peak_mb={val/1024/1024:.1f}")
    except Exception:
        pass