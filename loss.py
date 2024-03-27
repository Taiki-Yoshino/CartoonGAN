import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

class ContentLoss(nn.Module):
    def __init__(self, omega=10):
        super(ContentLoss, self).__init__()
        self.base_loss = nn.L1Loss()
        self.omega = omega
        vgg = vgg16(pretrained=True)
        perception_layers = list(vgg.features)[:25]
        self.perception = nn.Sequential(*perception_layers).eval()
        for param in self.perception.parameters():
            param.requires_grad = False

    def forward(self, source, target):
        source_features = self.perception(source)
        target_features = self.perception(target)
        return self.omega * self.base_loss(source_features, target_features)