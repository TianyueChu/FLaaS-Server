import torch
import torch.nn as nn
import torch.optim as optim


from kd.models.teacher import load_teacher_model
from kd.models.student import StudentModel
from kd.data.cifar10 import load_cifar10_dataloaders
from kd.training.kd_trainer import KDTrainer


def run_knowledge_distillation(batch_size=8, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    teacher = load_teacher_model()
    student = StudentModel()

    # Print model sizes
    def model_size(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Teacher model size: {model_size(teacher):,} parameters")
    print(f"Student model size: {model_size(student):,} parameters")

    # Load data
    train_loader, test_loader = load_cifar10_dataloaders(batch_size=batch_size)

    # Optimizer and loss
    optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(0, epochs):
        temperature = get_temperature(epoch, epochs)
        print(f"[Epoch {epoch}] Temperature: {temperature:.2f}")
        alpha = get_alpha(epoch,epochs)

        loss, train_acc, test_acc = KDTrainer(
            teacher_model=teacher,
            student_model=student,
            trainloader=train_loader,
            testloader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            teacher_percentage=alpha,
            temperature=temperature,
            device=device
        )
        print(f"[Epoch {epoch + 1}] Loss: {loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    print(f"Final student test accuracy after KD: {test_acc:.2f}%")

    # Save model weights as global_weights.bin
    torch.save(student.state_dict(), "global_weights.bin")

    return student

def get_temperature(epoch: int, total_epochs: int, start_T: float = 4.0, end_T: float = 1.0) -> float:
    if total_epochs <= 1:
        return end_T  # Avoid division by zero
    return start_T - (start_T - end_T) * ((epoch) / (total_epochs))

def get_alpha(epoch, total_epochs, max_alpha=0.7, min_alpha=0.3):
    return max(min_alpha, max_alpha - (epoch / total_epochs) * (max_alpha - min_alpha))

