import torch
from kd.models.teacher import load_teacher_model
from kd.models.student import StudentModel
from kd.data.cifar10 import load_cifar10_dataloaders
from kd.training.kd_trainer import kd_trainer

def run_knowledge_distillation(batch_size=8, epochs=5, alpha=0.5, temperature=4.0):
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

    # Train with KD
    # loss and acc
    epoch_loss = kd_trainer(
          teacher,
          student,
          trainloader=train_loader,
          criterion=nn.CrossEntropyLoss(),
          optimizer=optimizer,
          teacher_percentage=1,
          temperature=1)



    print(f"Final student accuracy after KD: {student_acc:.2f}%")
    return student_acc

