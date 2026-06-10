import os
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def load_teacher_model():
    teacher = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_vgg19_bn",
        pretrained=True
    )

    # Save temporarily
    torch.save(teacher.state_dict(), "teacher_model.pth")
    size_bytes = os.path.getsize("teacher_model.pth")
    size_mb = size_bytes / (1024 ** 2)

    print(f"Teacher model size: {size_mb:.2f} MB")

    return teacher


def load_teacher_model_wakeword():
    teacher = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_vgg19_bn",
        pretrained=True
    )

    # Change the final classifier from 10 classes to 2 classes
    in_features = teacher.classifier[-1].in_features
    teacher.classifier[-1] = nn.Linear(in_features, 2)

    # Save temporarily
    torch.save(teacher.state_dict(), "teacher_model_wuw.pth")
    size_bytes = os.path.getsize("teacher_model_wuw.pth")
    size_mb = size_bytes / (1024 ** 2)

    print(f"Teacher model size: {size_mb:.2f} MB")

    return teacher