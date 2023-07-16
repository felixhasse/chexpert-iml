import torchvision
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class VIT_L_16(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Load the DenseNet121 from ImageNet
        self.net = torchvision.models.vit_l_16(
            weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1 if pretrained else None)

        self.net.heads.head = nn.Linear(self.net.heads.head.in_features, num_classes)


    def get_transforms():
        return torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms

    def forward(self, inputs):
        """
        Forward the netword with the inputs
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


class DeepLabV3ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Note: Using pretrained weights currently doesn't work because a different output shape is used
        pretrained = False
        self.net = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained
            else None)

        self.net.classifier = DeepLabHead(2048, num_classes)

    def forward(self, inputs):
        return self.net(inputs)
