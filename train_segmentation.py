import argparse
import datetime
import json
import time
import math

from fastprogress import master_bar
from torch import optim
from torch.utils.data import DataLoader
from datasets import *
from constants import *
from models import *
from segmentation_trainer import *
from torch.utils.tensorboard import SummaryWriter
from custom_transformations import *

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

now = datetime.datetime.now()
model_name = f"{args.prefix + '_' if args.prefix else ''}lr={config['lr']}_batch={config['batch_size']}_{now.day}." \
             f"{now.month}_{now.hour}:{now.minute}"

directory_name = "heart_segmentation" if train_heart else "lung_segmentation"
model_path = f"models/{directory_name}/{model_name}.pth"

writer = SummaryWriter(log_dir=f"runs/{directory_name}/{model_name}")
with open(f"runs/{directory_name}/{model_name}/config.json", "w") as file:
    json.dump(config, file)

for key in config:
    writer.add_text(tag=key, text_string=str(config[key]))

# Define list of image transformations
transformation_list = [
    transforms.Resize((config["image_size"], config["image_size"])),
    SegmentationAugmentation(),
    HistogramEqualization(),
    transforms.ToTensor(),
]
mask_transformation_list = [
    transforms.Resize((config["image_size"], config["image_size"])),
    SegmentationAugmentation(),
    transforms.ToTensor(),
]

# Do we need normalization here?
if config["pretrained"]:
    transformation_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), )

image_transformation = transforms.Compose(transformation_list)
mask_transformation = transforms.Compose(
    mask_transformation_list)

print("Start loading dataset")

dataset = JSRTDataset(image_folder="data/JSRT/png_images",
                      mask_folder=f"data/JSRT/masks/{'heart' if train_heart else 'both_lungs'}",
                      image_transform=image_transformation, mask_transform=mask_transformation)

train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                            [math.floor(len(dataset) * config["train_test_split"]),
                                                             math.ceil(
                                                                 len(dataset) * (1 - config["train_test_split"]))])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=True)

print("Dataset loaded")

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
print(f"Starting training on device {device}")

model = DeepLabV3ResNet50(num_classes=1).to(device)

# Loss function
loss_function = nn.BCEWithLogitsLoss()

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
