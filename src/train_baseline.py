import argparse
import datetime
import json
import time
import math

from fastprogress import master_bar
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from .materials.datasets import *
from .materials.constants import *
from .materials.models import *
from .materials.chexpert_trainer import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(
    prog='Train segmentation',
    description='Trains a classification model on CheXpert', )
parser.add_argument(
    '--prefix', '-p',
    type=str,
    default="",
    help="Add a prefix to the name of the model you're training"
)

parser.add_argument(
    '--crop', '-c',
    action="store_true",
    help="Train on cropped images"
)

parser.add_argument(
    '--curriculum',
    action="store_true",
    help="Train on cropped images"
)

args = parser.parse_args()

with open(BASELINE_CONFIG_PATH, "r") as file:
    config = json.load(file)

directory = "crop" if args.crop else "baseline"

now = datetime.datetime.now()
model_name = f"{args.prefix + '_' if args.prefix else ''}{'crop_' if args.crop else ''}lr={config['lr']}_batch={config['batch_size']}_epochs={config['epochs']}_{now.day}." \
             f"{now.month}_{now.hour}:{now.minute}"

model_path = f"models/{directory}/{model_name}.pth"

writer = SummaryWriter(log_dir=f"runs/{directory}/{model_name}")
with open(f"runs/{directory}/{model_name}/config.json", "w") as file:
    json.dump(config, file)

for key in config:
    writer.add_text(tag=key, text_string=str(config[key]))

# Define list of image transformations
transformation_list = [
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
]

if config["pretrained"]:
    transformation_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), )

image_transformation = transforms.Compose(transformation_list)

train_dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/train.csv",
                                uncertainty_policy=config["policy"], transform=image_transformation,
                                lung_mask_path=config["mask_path"] if config["mask_path"] else None, crop_images=args.crop)

train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                 [math.floor(len(train_dataset) * config["train_data_size"]),
                                                  math.ceil(len(train_dataset) * (1 - config["train_data_size"]))])

test_dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/valid.csv",
                               uncertainty_policy=config["policy"], transform=image_transformation, 
                               lung_mask_path=config["mask_path"] if config["mask_path"] else None, crop_images=args.crop)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=8)

print("Dataset loaded")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Starting training on device {device}")

model = DenseNet121(num_classes=1, pretrained=config["pretrained"]).to(device)

# Loss function
loss_function = nn.BCELoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"],
                       weight_decay=config["weight_decay"])

# Best AUROC value during training
training_losses = []
validation_losses = []
validation_score = []

# Config progress bar
mb = master_bar(range(config["epochs"]))
mb.names = ['Training loss', 'Validation loss', 'Validation AUROC']
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
    val_loss, score = evaluate(epoch, model, test_dataloader, device, loss_function, mb)
    writer.add_scalar("Validation loss", val_loss, epoch)
    writer.add_scalar("ROCAUC", score, epoch)
    writer.flush()
    mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, score))
    validation_losses.append(val_loss)
    validation_score.append(score)

    # Update training chart
    mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0, epoch + 1], [0, 1])

    mb.write(f"AUROC: {score}")
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "score": score,
                "epoch": epoch,
                }, model_path)

writer.close()
