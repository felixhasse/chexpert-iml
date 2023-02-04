import csv
from enum import Enum, auto

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class UncertaintyPolicy(Enum):
    ONES = auto()
    ZEROS = auto()
    IGNORE = auto()


class ChestXrayDataset(Dataset):

    def __init__(self, data_path: str, uncertainty_policy: UncertaintyPolicy, transform=None):
        """
        Init Dataset
        """
        # Get all image paths and image labels from dataframe

        image_paths = []
        labels = []

        nn_class_count = 14
        with open(data_path, "r") as file:
            csv_reader = csv.reader(file)
            next(csv_reader, None)  # skip the header
            for line in csv_reader:
                image_path = line[0]
                npline = np.array(line)
                idx = [7, 10, 11, 13, 15]
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

                image_paths.append('./' + image_path)
                labels.append(label)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """
        Read image at index and convert to torch Tensor
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

