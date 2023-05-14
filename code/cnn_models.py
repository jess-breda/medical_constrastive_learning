### Models for contrastive pre-training and classification
# Written by Jess Breda and Claira Fucetola

import torch.nn as nn
from efficientnet_pytorch import EfficientNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

MODEL_SAVE_PATH = "C:\\Users\\jbred\\github\\medical_constrastive_learning\\models\\"


class MnistModel(nn.Module):
    def __init__(self, mode="contrastive", freeze_features=False):
        super(MnistModel, self).__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Classification projection head
        self.classification_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(256, 15),
            nn.Softmax(dim=1),
        )

        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
        else:
            for param in self.features.parameters():
                param.requires_grad = True

        self.mode = mode
        print(f"model initialized in {self.mode} mode")

    def forward(self, x):
        h = self.features(x)
        if self.mode == "contrastive":
            z = self.contrastive_head(h)
        else:
            z = self.classification_head(h)
        return h, z


class MnistEfficentNetModel(nn.Module):
    def __init__(self, mode="contrastive", freeze_features=False):
        super(MnistModel, self).__init__()

        # transfer learn on efficient net
        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")

        # Classification projection head
        self.classification_head = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(p=1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 15),
            nn.Softmax(dim=1),
        )

        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.8),
            nn.Linear(256, 128),
        )

        # freeze the feature extraction layers if freeze is True
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # freeze some of the layers and backprop through others
            for i, param in enumerate(self.backbone.parameters()):
                if i < 150:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        self.mode = mode
        print(f"model initialized in {self.mode} mode")

    def forward(self, x):
        h = self.backbone(x)
        if self.mode == "constrastive":
            z = self.contrastive_head(h)
        else:
            z = self.classification_head(h)
        return h, z
