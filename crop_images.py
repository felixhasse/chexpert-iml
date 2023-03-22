import argparse
import datetime
import json
import time
import math

from fastprogress import master_bar
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import *
from constants import *
from models import *
from chexpert_trainer import *
from util import load_segmentation_model

parser = argparse.ArgumentParser(
    prog='Train cropped model',
    description='Trains a DenseNet121 model on CheXpert images cropped to their lung bounding boxes')

parser.add_argument(
    '--prefix', '-p',
    type=str,
    default="",
    help="Add a prefix to the name of the model you're training"
)

args = parser.parse_args()

with open(CROPPED_CONFIG_PATH, "r") as file:
    config = json.load(file)

segmentation_transformations = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
])
lung_model_path = config["lung_model_path"]
heart_model_path = config["heart_model_path"]
dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/train.csv",
                          uncertainty_policy=config["policy"], transform=segmentation_transformations)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

print("Dataset loaded")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

heart_model = load_segmentation_model(model_path=heart_model_path, device=device)
lung_model = load_segmentation_model(model_path=lung_model_path, device=device)

print(f"Starting segmenting and cropping images on device {device}")




