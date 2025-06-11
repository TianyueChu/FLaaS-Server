from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


app = FastAPI()

class AggregationRequest(BaseModel):
    updates: List[List[float]]  # List of updates from clients (each is a list of floats)
    use_split_learning: bool

@app.post("/aggregate")
async def aggregate(request: AggregationRequest):
    updates = np.array(request.updates, dtype=np.float32)  # shape: (num_clients, model_size)
    print(f"size of the updates: {len(updates)}", flush=True)

    if not request.use_split_learning:
        # Simple averaging
        aggregated = np.mean(updates, axis=0).tolist()
    else:
        # Split Learning aggregation
        sl_outputs = []
        for update in updates:
            print(f"size of the update: {update.size}", flush=True)
            processed = use_split_learning(update)
            if processed is not None:
                print(f"size of the SL result: {processed.size}", flush=True)
                sl_outputs.append(processed)
            else:
                print("Split learning result is None — skipping", flush=True)

        if not sl_outputs:
            print("No valid updates received for split learning.", flush=True)
            return {"error": "No valid updates received for split learning."}

        aggregated = np.mean(np.array(sl_outputs), axis=0).tolist()

    return aggregated



def use_split_learning(update, num_samples=150):
    weights = np.array(update, dtype=np.float32)
    len_bottleneck = num_samples * 62720

    try:
        bottleneck = weights[:len_bottleneck].reshape((num_samples, 62720))
        labels = weights[len_bottleneck:].reshape((num_samples, 10))
    except ValueError:
        print("Shape mismatch in split learning update, skipping")
        return None

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

    # Save the trained weights
    fc_weight = model.fc.weight.data.numpy()  # shape: (10, 62720)
    fc_bias = model.fc.bias.data.numpy()      # shape: (10,)
    fc_weight_tflite = fc_weight.T            # shape: (62720, 10)

    flat_weights = fc_weight_tflite.flatten()  # length: 627200
    flat_bias = fc_bias.flatten()              # length: 10

    combined = np.concatenate([flat_weights, flat_bias]).astype(np.float32)  # length: 627210
    return combined


class TrainHead(nn.Module):
    def __init__(self, input_channels=1280, bottleneck_size=7, num_classes=10):
        super(TrainHead, self).__init__()
        self.flatten_dim = input_channels * bottleneck_size * bottleneck_size  # 62720
        self.fc = nn.Linear(self.flatten_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)         # [B, 7, 7, 1280] → [B, 1280, 7, 7]
        x = x.reshape(x.size(0), -1)      # Flatten to [B, 62720]
        return self.fc(x)
