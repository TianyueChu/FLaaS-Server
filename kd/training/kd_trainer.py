import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Tuple

def KDTrainer(
    teacher_model: torch.nn.Module,
    student_model: torch.nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    teacher_percentage: float = 0.5,
    temperature: float = 4.0,
    device: Optional[str] = None
) -> Tuple[float, float, float]:
    """
    Performs one epoch of KD training and returns:
    - training loss
    - training accuracy
    - test accuracy
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_model = teacher_model.to(device)
    student_model = student_model.to(device)

    teacher_model.eval()
    student_model.train()

    print("Starting KD training epoch...")
    print(f"Using device: {device}")
    print(f"Teacher percentage: {teacher_percentage}, Temperature: {temperature}")

    start_time = time.time()

    epoch_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Get teacher predictions (soft targets)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
            soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

        # Get student predictions
        student_logits = student_model(inputs)

        # Compute losses
        ce_loss = criterion(student_logits, targets)
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            soft_teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        total_loss = teacher_percentage * kd_loss + (1. - teacher_percentage) * ce_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track loss and accuracy
        epoch_loss += total_loss.item() * inputs.size(0)
        _, predicted = torch.max(student_logits, 1)
        correct_train += (predicted == targets).sum().item()
        total_train += targets.size(0)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(trainloader):
            print(f"Batch {batch_idx + 1}/{len(trainloader)} - Loss: {total_loss.item():.4f}")

    avg_loss = epoch_loss / len(trainloader.dataset)
    train_acc = correct_train / total_train * 100

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nEpoch completed in {elapsed_time:.2f} seconds. Avg Train Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    # Evaluate test accuracy
    student_model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == targets).sum().item()
            total_test += targets.size(0)

    test_acc = correct_test / total_test * 100
    print(f"Test Accuracy: {test_acc:.2f}%\n")

    return avg_loss, train_acc, test_acc

