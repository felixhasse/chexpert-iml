import time
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from chexpert_dataset import *
from constants import *
from config import *
from models import DenseNet121
from chexpert_trainer import *

# Define list of image transformations
transformation_list = [
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
]

image_transformation = transforms.Compose(transformation_list)

dataset = CheXpertDataset(data_path="./data/CheXpert-v1.0-small/train.csv",
                          uncertainty_policy=UncertaintyPolicy.ONES, transform=image_transformation)

# Split full dataset into train and test data using a manual seed to ensure reproducibility
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1],
                                                            generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Data Loaded")
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(device)
model = DenseNet121(num_classes=1).to(device)

# Loss function
loss_criteria = nn.BCELoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

# Learning rate will be reduced automatically during training
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LEARNING_RATE_SCHEDULE_FACTOR,
                                                    patience=LEARNING_RATE_SCHEDULE_PATIENCE, mode='max', verbose=True)

# Best AUROC value during training
best_score = 0
model_path = "densenet.pth"
training_losses = []
validation_losses = []
validation_score = []

# Config progress bar
mb = master_bar(range(MAX_EPOCHS))
mb.names = ['Training loss', 'Validation loss', 'Validation AUROC']
x = []

nonimproved_epoch = 0
start_time = time.time()

print("Starting training")
# Training each epoch
for epoch in mb:
    mb.main_bar.comment = f'Best AUROC score: {best_score}'
    x.append(epoch)

    # Training
    train_loss = epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb)
    mb.write('Finish training epoch {} with loss {:.4f}'.format(epoch, train_loss))
    training_losses.append(train_loss)

    # Evaluating
    val_loss, new_score = evaluate(epoch, model, test_dataloader, device, loss_criteria, mb)
    mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, new_score))
    validation_losses.append(val_loss)
    validation_score.append(new_score)

    # Update learning rate
    lr_scheduler.step(new_score)

    # Update training chart
    mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0, epoch + 1], [0, 1])

    # Save model
    if best_score < new_score:
        mb.write(f"Improve AUROC from {best_score} to {new_score}")
        best_score = new_score
        nonimproved_epoch = 0
        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_score": best_score,
                    "epoch": epoch,
                    "lr_scheduler": lr_scheduler.state_dict()}, model_path)
    else:
        nonimproved_epoch += 1
    if nonimproved_epoch > 10:
        break
        print("Early stopping")
    if time.time() - start_time > 3600 * 8:
        break
        print("Out of time")
