from .datasets import CheXpertDataset
import random
from torch.utils.data import Subset


def get_subsets(dataset: CheXpertDataset, balance_sets: bool = True, keep_easier_samples = False):
    # Get dataset indices corresponding to each difficulty score
    hard_indices = [i for i, difficulty in enumerate(dataset.difficulties) if difficulty == 2]
    medium_indices = [i for i, difficulty in enumerate(dataset.difficulties) if difficulty == 1]
    easy_indices = [i for i, difficulty in enumerate(dataset.difficulties) if difficulty == 0]

    # Get counts of positive, negative and uncertain labels in the dataset
    positive_count = sum(1 for _, label in dataset if label.item() == 1.0)
    negative_count = sum(1 for _, label in dataset if label.item() == 0.0)
    uncertain_count = sum(1 for _, label in dataset if label.item() == -1.0)

    # Calculate ratio of positive to negative samples in the dataset
    positive_ratio = positive_count / (negative_count + positive_count)

    if balance_sets:
        easy_indices, medium_indices, hard_indices = balance_examples(easy_indices, medium_indices, hard_indices,
                                                                      dataset, positive_ratio)

    easy_subset = Subset(dataset, easy_indices)
    if keep_easier_samples:
        medium_subset = Subset(dataset, easy_indices + medium_indices)
        hard_subset = Subset(dataset, easy_indices + medium_indices + hard_indices)

    else:
        medium_subset = Subset(dataset, medium_indices)
        hard_subset = Subset(dataset, hard_indices)

    return easy_subset, medium_subset, hard_subset


def balance_examples(easy_indices: list, medium_indices: list, hard_indices, dataset: CheXpertDataset,
                     positive_ratio: float):
    easy_positives = [i for i in easy_indices if dataset[i][1].item() == 1.0]
    easy_negatives = [i for i in easy_indices if dataset[i][1].item() == 0.0]
    medium_positives = [i for i in medium_indices if dataset[i][1].item() == 1.0]
    medium_negatives = [i for i in medium_indices if dataset[i][1].item() == 0.0]

    random.shuffle(easy_positives)
    random.shuffle(easy_negatives)
    random.shuffle(medium_positives)
    random.shuffle(medium_negatives)

    easy_ratio = len(easy_positives) / (len(easy_positives) + len(easy_negatives))

    while not is_optimal(positive_ratio, len(easy_positives), len(easy_negatives)):
        if easy_ratio > positive_ratio:
            index = easy_positives.pop(0)
            medium_positives.append(index)

        else:
            index = easy_negatives.pop(0)
            medium_negatives.append(index)
        easy_ratio = len(easy_positives) / (len(easy_positives) + len(easy_negatives))

    medium_ratio = len(medium_positives) / (len(medium_negatives) + len(medium_positives))

    while not is_optimal(positive_ratio, len(medium_positives), len(medium_negatives)):
        if medium_ratio > positive_ratio:
            index = medium_positives.pop(0)
            hard_indices.append(index)

        else:
            index = medium_negatives.pop(0)
            hard_indices.append(index)
        medium_ratio = len(medium_positives) / (len(medium_negatives) + len(medium_positives))

    new_easy_indices = easy_positives + easy_negatives
    new_medium_indices = medium_positives + medium_negatives

    return new_easy_indices, new_medium_indices, hard_indices


def is_optimal(dataset_positive_ratio: float, len_positives: int, len_negatives: int):
    subset_positive_ratio = len_positives / (len_positives + len_negatives)
    if subset_positive_ratio > dataset_positive_ratio:
        if abs(subset_positive_ratio - dataset_positive_ratio) > abs(((len_positives - 1) / (
                len_positives + len_negatives - 1) - dataset_positive_ratio)):
            return False
        else:
            return True
    else:
        if abs(subset_positive_ratio - dataset_positive_ratio) > abs((len_positives / (
                len_positives + len_negatives - 1)) - dataset_positive_ratio):
            return True
        else:
            return False
