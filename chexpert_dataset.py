import csv
from enum import Enum, auto
from os import path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from config import *


class UncertaintyPolicy(Enum):
    """
    Enum class to represent the different ways to handle uncertainty labels in the CheXpert dataset

    Attributes:
        ONES: Consider uncertainty labels as 1's
        ZEROS: Consider uncertainty labels as 0's
        IGNORE: Ignore uncertainty labels
    """
    ONES = auto()
    ZEROS = auto()
    IGNORE = auto()


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

    def __init__(self, data_path: str, uncertainty_policy: UncertaintyPolicy, transform: callable = None):

        image_paths = []
        labels = []

        nn_class_count = 1
        with open(data_path, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # skip the header
            for line in csv_reader:
                image_path = line[0]
                npline = np.array(line)
                idx = [7]
                label = list(npline[idx])
                for i in range(nn_class_count):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if uncertainty_policy == UncertaintyPolicy.ONES:  # All U-Ones
                                label[i] = 1
                            elif uncertainty_policy == UncertaintyPolicy.ZEROS:
                                label[i] = 0  # All U-Zeroes
                            # TODO: Include IGNORE policy
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                if npline[3] == "Frontal":
                    image_paths.append(path.join(DATASET_PATH, image_path))
                labels.append(label)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

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
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)
