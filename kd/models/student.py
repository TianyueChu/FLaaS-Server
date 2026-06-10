import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_small

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.backbone = mobilenet_v2(pretrained=False).features  # Takes 32x32, gives [B, 1280, 1, 1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Guarantees output is [B, 1280, 1, 1]
        self.classifier = nn.Linear(1280, 10)      # Final classifier

    def forward(self, x):
        x = self.backbone(x)           # -> [B, 1280, H, W] (H=W=1 for 32x32 input)
        x = self.pool(x)               # -> [B, 1280, 1, 1]
        x = x.view(x.size(0), -1)      # -> [B, 1280]
        x = self.classifier(x)         # -> [B, 10]
        return x


class StudentModel_wuw(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small()
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(
            in_features,
            2
        )

    def forward(self, x):
        return self.model(x)