import torchvision
from torch import nn


class DenseNet121(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Load the DenseNet121 from ImageNet
        self.net = torchvision.models.densenet121(pretrained=pretrained)

        # Get the input dimension of last layer
        kernel_count = self.net.classifier.in_features

        # Replace last layer with new layer that have num_classes nodes, after that apply Sigmoid to the output
        self.net.classifier = nn.Sequential(nn.Linear(kernel_count, num_classes), nn.Sigmoid())

    def forward(self, inputs):
        """
        Forward the netword with the inputs
        """
        return self.net(inputs)
