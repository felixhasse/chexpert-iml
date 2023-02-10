import json
import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from chexpert_dataset import *
from constants import *
from models import DenseNet121
from chexpert_trainer import *
from torch.utils.tensorboard import SummaryWriter
from uuid import uuid4

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

timestamp = round(time.time())

model_path = f"models/baseline/{timestamp}"


writer = SummaryWriter(logdir="runs/timestamp")

# Define list of image transformations
transformation_list = [
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
]

image_transformation = transforms.Compose(transformation_list)


train_dataset = CheXpertDataset(data_path="./data/CheXpert-v1.0-small/train.csv",
                          uncertainty_policy=config["policy"], transform=image_transformation)

test_dataset = CheXpertDataset(data_path="./data/CheXpert-v1.0-small/valid.csv",
                          uncertainty_policy=config["policy"], transform=image_transformation)
                          
train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=True)

print("Dataset loaded")

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f"Starting training on device {device}")

model = DenseNet121(num_classes=1).to(device)

# Loss function
loss_function = nn.BCELoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"], weight_decay=config["weight_decay"])

# Best AUROC value during training
best_score = 0
training_losses = []
validation_losses = []
validation_score = []

# Config progress bar
mb = master_bar(range(config["epochs"]))
mb.names = ['Training loss', 'Validation loss', 'Validation AUROC']
x = []

nonimproved_epoch = 0
start_time = time.time()

# Training each epoch
for epoch in mb:
    mb.main_bar.comment = f'Best AUROC score: {best_score}'
    x.append(epoch)

    # Training
    train_loss = epoch_training(epoch, model, train_dataloader, device, loss_function, optimizer, mb)
    writer.add_scalar("Train loss", train_loss, epoch)
    mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))
    training_losses.append(train_loss)

    # Evaluating
    val_loss, new_score = evaluate(epoch, model, test_dataloader, device, loss_function, mb)
    writer.add_scalar("Validation loss", val_loss, epoch)
    writer.add_scalar("ROCAUC", new_score, epoch)
    mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, new_score))
    validation_losses.append(val_loss)
    validation_score.append(new_score)

    # Update training chart
    mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0, epoch + 1], [0, 1])

    # Save model
    if best_score < new_score:
        mb.write(f"Improve AUROC from {best_score} to {new_score}")
        best_score = new_score
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                    "epoch": epoch,
                    }, model_path)
    
writer.flush()
writer.close()


