import torchvision
from torch import nn


class MaxViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.net = torchvision.models.maxvit_t(torchvision.models.MaxVit_T_Weights.DEFAULT if pretrained else None)

        # Get the input dimension of last layer
        kernel_count = self.net.classifier.in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        self.net.classifier = nn.Sequential(nn.Linear(kernel_count, num_classes), nn.Sigmoid())

    def forward(self, inputs):
        """
        Forward the network with the inputs
        """
        return self.net(inputs)


class DenseNet121(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Load the DenseNet121 from ImageNet
        self.net = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.DEFAULT if pretrained else None)

        # Get the input dimension of last layer
        kernel_count = self.net.classifier.in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        self.net.classifier = nn.Sequential(nn.Linear(kernel_count, num_classes), nn.Sigmoid())

    def forward(self, inputs):
        """
        Forward the netword with the inputs
        """
        return self.net(inputs)


class EfficientNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Load the DenseNet121 from ImageNet
        self.net = torchvision.models.efficientnet_v2_s(
            weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)

        # Get the input dimension of last layer
        kernel_count = self.net.classifier[1].in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        self.net.classifier = nn.Sequential(
            nn.Linear(kernel_count, num_classes), nn.Sigmoid())

    def forward(self, inputs):
        """
        Forward the netword with the inputs
        """
        return self.net(inputs)
