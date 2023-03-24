import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps


class HistogramEqualization(torch.nn.Module):
    def __call__(self, img):
        new_img = ImageOps.equalize(img)
        return new_img


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
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=5)]), p=0.5),
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
