import torch
from torchvision import transforms
from PIL import ImageOps


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


class LungSegmentationAugmentation(torch.nn.Module):
    """
    Applies the data augmentation pipeline by Chamveha et al. (2020)
    Probabilites for random operations were not given, so we use default values here
    Excludes Histogram Equalization as it is applied seperately
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


class HeartSegmentationAugmentation(torch.nn.Module):
    """
    Applies the data augmentation pipeline by Chamveha et al. (2020)
    Probabilites for random operations were not given, so we use default values here
    Excludes Histogram Equalization as it is applied seperately
    """

    def __call__(self, img):
        transform_list = [
            transforms.ToTensor(),
            transforms.RandomRotation(degrees=8),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=5)]), p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([GaussianNoise(mean=0.0, std=0.1)]), p=0.3),
            transforms.ToPILImage()
        ]
        transform = transforms.Compose(transform_list)
        return transform(img)


class AlternativeSegmentationAugmentation(torch.nn.Module):

    def __call__(self, img):
        transform_list = [
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=8, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.ToPILImage()
        ]

        transform = transforms.Compose(transform_list)
        return transform(img)


class CombinedHeartSegmentationAugmentation(torch.nn.Module):
    """
    Applies the data augmentation pipeline by Chamveha et al. (2020)
    Probabilites for random operations were not given, so we use default values here
    Excludes Histogram Equalization as it is applied seperately
    """

    def __call__(self, img):
        transform_list = [
            transforms.ToTensor(),
            transforms.RandomAffine(degrees=8, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=5)]), p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([GaussianNoise(mean=0.0, std=0.1)]), p=0.3),
            transforms.ToPILImage()
        ]
        transform = transforms.Compose(transform_list)
        return transform(img)


class CombinedLungSegmentationAugmentation(torch.nn.Module):
    """
    Applies the data augmentation pipeline by Chamveha et al. (2020)
    Probabilites for random operations were not given, so we use default values here
    Excludes Histogram Equalization as it is applied seperately
    """

    def __call__(self, img):
        transform_list = [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=8, translate=(0.2, 0.2), scale=(0.7, 1.3)),
            transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=5)]), p=0.3),
            transforms.RandomApply(torch.nn.ModuleList([GaussianNoise(mean=0.0, std=0.1)]), p=0.3),
            transforms.ToPILImage()
        ]
        transform = transforms.Compose(transform_list)
        return transform(img)
