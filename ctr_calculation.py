import torch


def calculate_diameter(mask: torch.Tensor):
    # Creates a new 1d-tensor that represents the columns in the image
    # Is True for each index if the corresponding column is part of the mask (contains white pixels), 0 otherwise
    column_in_mask = torch.any(mask, dim=1).squeeze()

    # Find left and right edge of the mask
    left_edge_index = torch.nonzero(column_in_mask).squeeze()[0].item()
    right_edge_index = torch.nonzero(column_in_mask).squeeze()[-1].item()

    diameter = right_edge_index - left_edge_index

    return diameter


def calculate_ctr(heart_mask: torch.Tensor, lung_mask: torch.Tensor):
    cardiac_diameter = calculate_diameter(heart_mask)
    thoracic_diameter = calculate_diameter(lung_mask)

    ctr = cardiac_diameter / thoracic_diameter

    return ctr
