import argparse
import json
import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from materials.custom_transformations import HistogramEqualization
from materials.datasets import CheXpertDataset
from src.materials.segmentation_inference import infer_from_tensor
from materials.util import load_segmentation_model
from src.materials.postprocessing import *

parser = argparse.ArgumentParser(
    prog='Segment CheXpert',
    description='Generate heart and lung masks for the CheXpert dataset', )

parser.add_argument(
    '--name', '-n',
    type=str,
    default="masks",
    help="Add a name for the directory the masks are saved in"
)
args = parser.parse_args()

mask_dir = os.path.join("data/chexpert_masks", args.name)
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
heart_dir = os.path.join(mask_dir, "heart")
lung_dir = os.path.join(mask_dir, "lung")

for directory in [mask_dir, heart_dir, lung_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

with open("chexpert_segentation_config.json", "r") as file:
    config = json.load(file)
print(heart_dir)
transformation_list = [
    transforms.Resize((256, 256)),
    HistogramEqualization(),
    transforms.ToTensor(),
]
image_transformation = transforms.Compose(transformation_list)

train_dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/train.csv",
                                uncertainty_policy="zeros", transform=image_transformation)

valid_dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/valid.csv",
                                uncertainty_policy="zeros", transform=image_transformation)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config["batch_size"], shuffle=True)

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"

heart_model = load_segmentation_model(config["heart_model_path"], device)
lung_model = load_segmentation_model(config["lung_model_path"], device)
print(f"Starting segmentation on device {device}")

for dataloader in (train_dataloader, valid_dataloader):
    num_images = len(dataloader)
    for step, (image, label) in enumerate(dataloader):
        print(f"Segmenting image {step + 1} of {num_images}")
        image_path = dataloader.dataset.get_path(step)
        directory = image_path.split("view")[0].split("small")[-1]
        image_name = image_path.split("/")[-1]

        os.makedirs(heart_dir + directory, exist_ok=True)
        os.makedirs(lung_dir + directory, exist_ok=True)

        heart_pred = infer_from_tensor(image, heart_model, device)
        lung_pred = infer_from_tensor(image, lung_model, device)
        heart_pred = process_heart_mask(heart_pred)
        lung_pred = process_lung_mask(lung_pred)
        heart_mask = transforms.ToPILImage()(heart_pred).convert("1")
        lung_mask = transforms.ToPILImage()(lung_pred).convert("1")
        heart_mask.save(os.path.join(heart_dir + directory, image_name))
        lung_mask.save(os.path.join(lung_dir + directory, image_name))

print("Done")
