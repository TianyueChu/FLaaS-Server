import torch.nn as nn
from torchvision.models import mobilenet_v2


class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Bottleneck feature extractor from MobileNetV2
        self.backbone = mobilenet_v2(pretrained=False).features

        # Classification head (matches train_head.tflite: 62720 → 10)
        self.classifier = nn.Linear(7 * 7 * 1280, 10)

    def forward(self, x):
        x = self.backbone(x)  # [B, 1280, 7, 7]
        x = x.view(x.size(0), -1)  # Flatten to [B, 62720]
        x = self.classifier(x)  # Final logits: [B, 10]
        return x
