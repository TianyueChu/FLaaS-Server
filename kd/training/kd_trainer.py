import torch
import torch.nn.functional as F
from typing import Optional


def knowledge_distillation_train(teacher_model: torch.nn.Module,
                                 student_model: torch.nn.Module,
                                 trainloader: torch.utils.data.DataLoader,
                                 criterion: torch.nn.Module,
                                 optimizer: torch.optim.Optimizer,
                                 teacher_percentage: float = 0.5,
                                 # defines the weight of the teacher's predictions vs the dataset's labels
                                 temperature: float = 2,  # defines the softness of the softmax temperature
                                 device: Optional[str] = None):
    '''
    this function performs a single epoch of training with knowledge distillation
    '''
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model = student_model.to(device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    student_model.train()

    epoch_loss = 0.0

    for inputs, targets in trainloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        # the teacher is not trained so there is no need to keep the gradients
        with torch.no_grad():
            teacher_targets = teacher_model(inputs).to(device)
            teacher_targets = F.softmax(teacher_targets / temperature, dim=1)
            teacher_targets = teacher_targets.to(device)

        # we perform a typical torch training loop, but with the teacher's predictions as targets as well
        optimizer.zero_grad()
        outputs = student_model(inputs)
        # we calculate the loss with the targets from the dataset predictions
        absolute_loss = criterion(outputs, targets)
        # as well as the loss with the teacher's predictions
        teacher_loss = criterion(outputs, teacher_targets)
        # and we perform a weighted sum of the two losses
        total_loss = teacher_loss * teacher_percentage + (1 - teacher_percentage) * absolute_loss
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss.item() * inputs.size(0)

        epoch_loss = epoch_loss / len(trainloader.dataset)

    return epoch_loss
