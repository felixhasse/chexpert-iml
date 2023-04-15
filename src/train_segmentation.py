import argparse
import datetime
import json
import time
import math

import torchvision
from fastprogress import master_bar
from torch import optim
from torch.utils.data import DataLoader
from src.materials.datasets import *
from src.materials.constants import *
from src.materials.segmentation_trainer import *
from torch.utils.tensorboard import SummaryWriter
from src.materials.custom_transformations import *
from segmentation.models import unet

from src.materials.loss_functions import CombinedLoss
from src.materials.models import DeepLabV3ResNet50

parser = argparse.ArgumentParser(
    prog='Train segmentation',
    description='Trains a segmentation model for the lungs or heart on the JSRT dataset', )

parser.add_argument(
    '--target', '-t',
    type=str,
    default="lung",
    choices=["heart", "lung"],
    help="Specify if you want to train the lung or heart segmentation model"
)
parser.add_argument(
    '--prefix', '-p',
    type=str,
    default="",
    help="Add a prefix to the name of the model you're training"
)

args = parser.parse_args()

# Heart segmentation model will be trained when true, lung segmentation otherwise
train_heart = args.target == "heart"

with open(SEGMENTATION_CONFIG_PATH, "r") as file:
    config = json.load(file)

data_augmentation = None
aug_prefix = "no_aug"
if config["augmentation"] == "default":
    data_augmentation = HeartSegmentationAugmentation() if train_heart else LungSegmentationAugmentation()
    aug_prefix = "default_aug"

elif config["augmentation"] == "alternative":
    data_augmentation = AlternativeSegmentationAugmentation()
    aug_prefix = "alt_aug"

elif config["augmentation"] == "combined":
    data_augmentation = CombinedHeartSegmentationAugmentation() if train_heart else CombinedLungSegmentationAugmentation()
    aug_prefix = "combined_aug"

now = datetime.datetime.now()
model_name = f"{args.prefix + '_' if args.prefix else ''}{'DeepLabV3_' if config['use_DeepLabV3'] else 'UNet_VGG16_'}{aug_prefix}_lr={config['lr']}_batch={config['batch_size']}_{now.day}." \
             f"{now.month}_{now.hour}:{now.minute}"

directory_name = "heart_segmentation" if train_heart else "lung_segmentation"
model_path = f"models/{directory_name}/{model_name}.pth"

writer = SummaryWriter(log_dir=f"runs/{directory_name}/{model_name}")
with open(f"runs/{directory_name}/{model_name}/config.json", "w") as file:
    json.dump(config, file)

for key in config:
    writer.add_text(tag=key, text_string=str(config[key]))

# Define list of image transformations
transformation_list = [transforms.Resize((config["image_size"], config["image_size"]))]
transformation_list += [data_augmentation] if data_augmentation is not None else []
transformation_list += [HistogramEqualization(), transforms.ToTensor()]

test_image_transformation_list = [
    transforms.Resize((config["image_size"], config["image_size"])),
    HistogramEqualization(),
    transforms.ToTensor(),
]
mask_transformation_list = [transforms.Resize((config["image_size"], config["image_size"]))]
mask_transformation_list += [data_augmentation] if data_augmentation is not None else []
mask_transformation_list += [transforms.ToTensor()]

test_mask_transformation_list = [
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
]
# Do we need normalization here?
if config["pretrained"]:
    transformation_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), )

image_transformation = transforms.Compose(transformation_list)
test_image_transformation = transforms.Compose(test_image_transformation_list)
test_mask_transformation = transforms.Compose(test_mask_transformation_list)
mask_transformation = transforms.Compose(
    mask_transformation_list)

print("Start loading dataset")

train_dataset = SegmentationDataset(
    image_folder=path.join(SEGMENTATION_DATASET_PATH, "heart" if train_heart else "lung", "images"),
    mask_folder=path.join(SEGMENTATION_DATASET_PATH, "heart" if train_heart else "lung", "masks"),
    image_transform=image_transformation, mask_transform=mask_transformation)

test_dataset = SegmentationDataset(
    image_folder=path.join(SEGMENTATION_DATASET_PATH, "heart" if train_heart else "lung", "images"),
    mask_folder=path.join(SEGMENTATION_DATASET_PATH, "heart" if train_heart else "lung", "masks"),
    image_transform=test_image_transformation,
    mask_transform=test_mask_transformation)

train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                 [math.floor(len(train_dataset) * config["train_test_split"]),
                                                  math.ceil(
                                                      len(train_dataset) * (1 - config["train_test_split"]))],
                                                 generator=torch.Generator().manual_seed(42))

_, test_dataset = torch.utils.data.random_split(test_dataset,
                                                [math.floor(len(test_dataset) * config["train_test_split"]),
                                                 math.ceil(
                                                     len(test_dataset) * (1 - config["train_test_split"]))],
                                                generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)

print("Dataset loaded")

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
print(f"Starting training on device {device}")

model = DeepLabV3ResNet50(num_classes=1, pretrained=False).to(device) if config["use_DeepLabV3"] else unet.unet_vgg16(
    n_classes=1, batch_size=config["batch_size"]).to(device)

# Loss function
loss_function = CombinedLoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"],
                       weight_decay=config["weight_decay"])

# Best IoU value during training
training_losses = []
validation_losses = []
IoUs = []

# Config progress bar
mb = master_bar(range(config["epochs"]))
mb.names = ['Training loss', 'Validation loss']
x = []
start_time = time.time()

# Training each epoch
for epoch in mb:
    x.append(epoch)

# Training
train_loss = epoch_training(epoch, model, train_dataloader, device, loss_function, optimizer, mb)
writer.add_scalar("Train loss", train_loss, epoch)
mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))
training_losses.append(train_loss)

# Evaluating
val_loss, IoU = evaluate(epoch, model, test_dataloader, device, loss_function, mb)
writer.add_scalar("Validation loss", val_loss, epoch)
writer.flush()
writer.add_scalar("Mean IoU", IoU, epoch)
writer.flush()
mb.write('Finish validation epoch {} with loss {:.4f}'.format(epoch, val_loss))
validation_losses.append(val_loss)
IoUs.append(IoU)

# Update training chart
mb.update_graph([[x, training_losses], [x, validation_losses]], [0, epoch + 1], [0, 1])

torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            }, model_path)

writer.close()
