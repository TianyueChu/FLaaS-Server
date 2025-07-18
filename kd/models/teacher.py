import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_teacher_model():
    teacher = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_vgg19_bn",
        pretrained=True
    )
    return teacher
