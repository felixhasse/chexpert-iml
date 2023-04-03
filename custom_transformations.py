import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps


class CombinedLoss:
    def __call__(self, logits, labels):
        bce = torch.nn.BCEWithLogitsLoss()(logits, labels)
        soft_dice = SoftDiceLossV1()(logits, labels)
        return (soft_dice + bce) / 2


# Soft Dice Loss for binary segmentation
# Taken from https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
# v1: pytorch autograd
class SoftDiceLossV1(torch.nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class HistogramEqualization(torch.nn.Module):
    def __call__(self, img):
        new_img = ImageOps.equalize(img)
        return new_img


class GaussianNoise(torch.nn.Module):

    def __init__(self, mean=0.0, std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor):
        noise = torch.randn(*tensor.size()) * self.std + self.mean
        return tensor + noise


class SegmentationAugmentation(torch.nn.Module):
    """
    Applies the data augmentation pipeline by Chamveha et al. (2020)
    Probabilites for random operations were not given, so we use default values here
    Excludes Histogram Equalization as it is applied seperately
    TODO: Add gaussian noise
    """

    def __call__(self, img):
        transform_list = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=8),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=5)]), p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([GaussianNoise(mean=0.0, std=0.1)]), p=0.3),
            transforms.ToPILImage()
        ]
        transform = transforms.Compose(transform_list)
        return transform(img)


class ExtendedSegmentationAugmentation(torch.nn.Module):

    def __call__(self, img):
        transform_list = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.RandomPerspective(),
            transforms.RandomAutocontrast(),
            transforms.RandomInvert(),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=5)]), p=0.5),
            transforms.ToPILImage()
        ]
        transform = transforms.Compose(transform_list)
        return transform(img)
