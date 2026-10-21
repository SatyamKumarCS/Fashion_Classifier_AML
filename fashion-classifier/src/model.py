"""Small two-block CNN; input float32 (N,1,28,28) in [0,1]."""
import torch.nn as nn

MODEL_CONFIG = {'in_channels': 1, 'channels': [32, 64], 'hidden_units': [128, 64],
                'dropout': 0.4, 'num_classes': 10, 'input_shape': [1, 28, 28]}


class SmallCNN(nn.Module):
    """Adapt the reference CNN with learned spatial filters and raw logits."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.4), nn.Linear(64, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
