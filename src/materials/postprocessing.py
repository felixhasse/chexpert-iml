import torch
import numpy as np
from scipy import ndimage


def process_lung_mask(mask: torch.Tensor):
    mask = closing(mask)
    mask = largest_connected_components(mask, n=2)
    return mask


def process_heart_mask(mask: torch.Tensor):
    mask = closing(mask)
    mask = largest_connected_components(mask, n=1)
    return mask


def closing(mask: torch.Tensor):
    mask = mask.squeeze().cpu().numpy()

    # Both dilation and erosion use a 3x3 kernel by default
    struct = np.ones((10, 10))
    # Perform dilation
    mask = ndimage.binary_closing(mask, iterations=1, structure=struct)
    # Convert the result to a PyTorch tensor
    return torch.from_numpy(mask.astype(np.float32))


def largest_connected_components(mask: torch.Tensor, n: int = 1):
    np_array = mask.cpu().numpy()

    # Label connected components
    labeled_array, num_features = ndimage.label(np_array)

    # Count the number of elements in each connected component
    component_sizes = np.bincount(labeled_array.ravel())

    # Find the largest n connected components (excluding the area not belonging to a mask)
    largest_n_component_labels = np.argsort(component_sizes[1:])[-n:] + 1

    # Create a boolean mask to extract the largest n connected components
    mask = np.isin(labeled_array, largest_n_component_labels)

    # Convert the result back to a PyTorch tensor if needed
    largest_n_components = torch.from_numpy(mask.astype(np.float32))

    return largest_n_components
