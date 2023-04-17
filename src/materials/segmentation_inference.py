import collections
import torch
from torch.utils.data import DataLoader
from .postprocessing import *

from .util import calculate_ctr
from .models import DeepLabV3ResNet50

device = "cpu"


def infer_from_tensor(tensor: torch.Tensor, model: torch.nn, device: str):
    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        if type(output) == collections.OrderedDict:
              output = output["out"]
        output = torch.sigmoid(output)
        output = output.squeeze()
        output = torch.round(output)
        return output


def ctr_from_tensor(tensor: torch.Tensor, heart_segmentation_model: torch.nn, lung_segmentation_model: torch.nn):
    # heart_model = load_model(heart_segmentation_path)
    # lung_model = load_model(lung_segmentation_path)
    with torch.no_grad():
        heart_output = heart_segmentation_model(tensor)
        if type(heart_output) == collections.OrderedDict:
              heart_output = heart_output["out"]
        heart_output = torch.sigmoid(heart_output)
        heart_output = heart_output.squeeze()
        heart_output = torch.round(heart_output)
        heart_output = process_heart_mask(heart_output)

        lung_output = lung_segmentation_model(tensor)
        if type(lung_output) == collections.OrderedDict:
              lung_output = lung_output["out"]
        lung_output = torch.sigmoid(lung_output)
        lung_output = lung_output.squeeze()
        lung_output = torch.round(lung_output)
        lung_output = process_lung_mask(lung_output)

        ctr = calculate_ctr(heart_mask=heart_output, lung_mask=lung_output)
        return ctr


# def bb_from_tensor(tensor: torch.Tensor, lung_segmentation_model: torch.nn):

def load_model(model_path: str):
    model = DeepLabV3ResNet50(num_classes=1, pretrained=False)
    state_dict = torch.load(model_path, map_location=torch.device(device))["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
