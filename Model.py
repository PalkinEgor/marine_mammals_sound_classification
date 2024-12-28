import torch
from torchvision.models import resnet50


class CustomResNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                            padding=(3, 3), bias=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
