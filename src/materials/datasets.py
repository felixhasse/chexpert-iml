import csv
import glob
from os import path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .constants import *
from torchvision import transforms

from .segmentation_inference import ctr_from_tensor
from .util import generate_bb, calculate_ctr


def scoring_function(ctr, label):
    difficulty = 0
    if label == -1:
        difficulty = 2
    elif 0.8 > ctr > 0.2:
        if round(ctr) != label:
            difficulty = 2
        else:
            if ctr > 0.6 or ctr < 0.4:
                difficulty = 0
            else:
                difficulty = 1
    return difficulty


class CheXpertDataset(Dataset):
    """
    PyTorch dataset class to load the CheXpert dataset

    Args:
        data_path (str): Path to the CSV file containing the dataset
        uncertainty_policy (UncertaintyPolicy): Enum value representing the way to handle uncertain labels
        transform (callable, optional): Optional data transformation to be applied to the images. Default is None.

    Attributes:
        image_paths (List[str]): List of image paths
        labels (List[List[int]]): List of labels for each image
        transform (callable): Data transformation to be applied to the images
    """

    def __init__(self, data_path: str, uncertainty_policy: str, transform: callable = None, lung_mask_path: str = None,
                 heart_mask_path: str = None, crop_images: bool = False, curriculum_learning: bool = False):

        image_paths = []
        labels = []
        lung_mask_paths = []
        heart_mask_paths = []
        difficulties = []

        nn_class_count = 1
        with open(data_path, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # skip the header
            for line in csv_reader:
                image_path = line[0]
                npline = np.array(line)
                idx = 7
                label = npline[idx]
                if not label:
                    label = 0.0
                label = float(label)
                if npline[3] == "Frontal" and label != -1:  # Only include frontal images
                    image_paths.append(f"{DATASET_PATH}/{image_path}")
                    labels.append(label)
                    if lung_mask_path is not None:
                        lung_mask_paths.append(lung_mask_path + image_path.split("small")[-1])
                    if heart_mask_path is not None:
                        heart_mask_paths.append(heart_mask_path + image_path.split("small")[-1])
        self.difficulties = []
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.lung_mask_paths = lung_mask_paths
        self.heart_mask_paths = heart_mask_paths
        self.crop_images = crop_images
        self.curriculum_learning = curriculum_learning
        self.uncertainty_policy = uncertainty_policy
        if curriculum_learning:
            self.generate_difficulties()

    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns:
            int: The number of samples in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Return the image and label for a given index

        Args:
            index (int): The index of the sample to return

        Returns:
            Tuple[PIL.Image.Image, torch.Tensor]: A tuple containing the image and label for the given index
        """

        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        if self.crop_images:
            # Resize the image to match mask dimensions
            image = transforms.Resize((512, 512))(image)
            mask = transforms.ToTensor()(Image.open(self.lung_mask_paths[index]).convert("1"))
            bbox = generate_bb(mask)
            image.crop(bbox)

        label = self.labels[index]
        if label == -1:
            if self.uncertainty_policy == "zeros":
                label = 0
            else:
                label = 1

        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor([label], dtype=torch.float)

    def get_path(self, index):
        return self.image_paths[index]

    def get_ctr(self, index):
        lung_mask = transforms.ToTensor()(Image.open(self.lung_mask_paths[index]).convert("1"))
        heart_mask = transforms.ToTensor()(Image.open(self.heart_mask_paths[index]).convert("1"))
        ctr = calculate_ctr(heart_mask=heart_mask, lung_mask=lung_mask)
        return ctr

    def generate_difficulties(self):
        for i in range(len(self.image_paths)):
            image_path = self.image_paths[i]
            label = self.labels[i]
            lung_mask = transforms.ToTensor()(Image.open(self.lung_mask_paths[i]).convert("1"))
            heart_mask = transforms.ToTensor()(Image.open(self.heart_mask_paths[i]).convert("1"))
            ctr = calculate_ctr(heart_mask=heart_mask, lung_mask=lung_mask)
            self.difficulties.append(scoring_function(ctr, label))


class MontgomeryDataset(Dataset):

    def __init__(self, image_folder: str, left_mask_folder: str, right_mask_folder, transform: callable = None):
        image_paths = sorted(glob.glob(path.join(image_folder, "*.png")))
        left_mask_paths = []
        right_mask_paths = []

        for image_path in image_paths:
            image_name = image_path.split("/")[-1]
            left_mask_paths.append(path.join(left_mask_folder, image_name))
            right_mask_paths.append(path.join(right_mask_folder, image_name))

        self.image_paths = image_paths
        self.left_mask_paths = left_mask_paths
        self.right_mask_paths = right_mask_paths
        self.transform = transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        left_mask_path = self.left_mask_paths[index]
        right_mask_path = self.right_mask_paths[index]
        left_mask = Image.open(left_mask_path).convert('1')
        right_mask = Image.open(right_mask_path).convert('1')

        if self.transform is not None:
            image = self.transform(image)
            left_mask = self.transform(left_mask)
            right_mask = self.transform(right_mask)

        mask = left_mask.logical_or(right_mask)  # This only works if self.transform includes conversion to tensor
        # mask = mask.long()  # Convert true, false to 1, 0

        return image, mask

    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns:
            int: The number of samples in the dataset
        """
        return len(self.image_paths)


class SegmentationDataset(Dataset):
    def __init__(self, image_folder: str, mask_folder: str, image_transform: callable = None,
                 mask_transform: callable = None):
        image_paths = sorted(glob.glob(path.join(image_folder, "*.png")))
        mask_paths = []

        for image_path in image_paths:
            image_name = image_path.split("/")[-1]
            mask_paths.append(path.join(mask_folder, image_name))

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.mask_transform = mask_transform
        self.image_transform = image_transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)

        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).convert('1')

        if isinstance(mask, torch.Tensor):
            mask = transforms.ToPILImage()(mask)

        # Applying the same seed ensures that image and mask transforms are the same
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        if self.image_transform is not None:
            image = self.image_transform(image)

        torch.manual_seed(seed)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns:
            int: The number of samples in the dataset
        """
        return len(self.image_paths)
