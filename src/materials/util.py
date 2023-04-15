import torch
from segmentation.models import unet

from .models import DeepLabV3ResNet50


def calculate_diameter(mask: torch.Tensor):
    # Creates a new 1d-tensor that represents the columns in the image
    # Is True for each index if the corresponding column is part of the mask (contains white pixels), 0 otherwise
    column_in_mask = torch.any(mask, dim=1).squeeze()

    # Find left and right edge of the mask
    nonzero_column_indexes = torch.nonzero(column_in_mask).squeeze()
    # TODO: Find a better way to handle nonexistent masks
    if nonzero_column_indexes.numel() <= 1:
        return len(column_in_mask)
    left_edge_index = nonzero_column_indexes[0].item()
    right_edge_index = nonzero_column_indexes[-1].item()

    diameter = right_edge_index - left_edge_index

    return diameter


def calculate_ctr(heart_mask: torch.Tensor, lung_mask: torch.Tensor):
    cardiac_diameter = calculate_diameter(heart_mask)
    thoracic_diameter = calculate_diameter(lung_mask)

    ctr = cardiac_diameter / thoracic_diameter

    return ctr


def generate_bb(mask: torch.Tensor):
    # Creates a new 1d-tensor that represents the columns/rows in the image
    # Is True for each index if the corresponding column/row is part of the mask (contains white pixels), 0 otherwise
    column_in_mask = torch.any(mask, dim=1).squeeze()
    row_in_mask = torch.any(mask, dim=-1).squeeze()

    nonzero_column_indexes = torch.nonzero(column_in_mask).squeeze()
    nonzero_row_indexes = torch.nonzero(column_in_mask).squeeze()
    
    if nonzero_column_indexes.numel() <= 1 or nonzero_row_indexes.numel() <= 1:
        return 0, 0, len(column_in_mask), len(row_in_mask)
    
    # Find the edges of the bounding box
    left = nonzero_column_indexes[0].item()
    right = nonzero_column_indexes[-1].item()
    upper = nonzero_row_indexes[0].item()
    lower = nonzero_row_indexes[-1].item()

    return left, upper, right, lower


def load_segmentation_model(model_path: str, is_deeplab: bool, device: str):
    if is_deeplab:
        model = DeepLabV3ResNet50(num_classes=1, pretrained=False)
    else:
        model = unet.unet_vgg16(n_classes=1, batch_size=8)
    state_dict = torch.load(model_path, map_location=torch.device(device))["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
