import argparse
import datetime
import json
import time
import math
import optuna

from fastprogress import master_bar
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from .materials.datasets import *
from .materials.constants import *
from .materials.models import *
from .materials.chexpert_trainer import *
from torch.utils.tensorboard import SummaryWriter

def objective(trial):

    with open(BASELINE_CONFIG_PATH, "r") as file:
        config = json.load(file)

        # Define list of image transformations
        transformation_list = [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
        ]

        if config["pretrained"]:
            transformation_list.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD), )

        image_transformation = transforms.Compose(transformation_list)

        full_dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/train.csv",
                                        uncertainty_policy=config["policy"], transform=image_transformation,
                                        lung_mask_path=config["mask_path"] if config["mask_path"] else None, crop_images=False)

        train_dataset, validation_dataset = torch.utils.data.random_split(full_dataset,
                                                        [math.floor(len(full_dataset) * 0.9),
                                                        math.ceil(len(full_dataset) * 0.1)], generator=torch.Generator().manual_seed(42))

        train_dataset, _ = torch.utils.data.random_split(train_dataset,
                                                        [math.floor(len(train_dataset) * config["train_data_size"]),
                                                        math.ceil(len(train_dataset) * (1 - config["train_data_size"]))], generator=torch.Generator().manual_seed(42))

        test_dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/valid.csv",
                                    uncertainty_policy=config["policy"], transform=image_transformation, 
                                    lung_mask_path=config["mask_path"] if config["mask_path"] else None, crop_images=False)


        directory =  "baseline/hyperparameters"

        now = datetime.datetime.now()

        lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
        batch_size = trial.suggest_int("batch_size", 8, 512, log=True)

        model_name = f"{time.time()}_lr={lr}_bs={batch_size}"

        writer = SummaryWriter(log_dir=f"runs/{directory}/{model_name}")
        with open(f"runs/{directory}/{model_name}/config.json", "w") as file:
            json.dump(config, file)

        for key in config:
            writer.add_text(tag=key, text_string=str(config[key]))

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        validation_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        print("Dataset loaded")

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        print(f"Starting training on device {device}")
        print(torch.cuda.device_count())


        model = DenseNet121(num_classes=1, pretrained=config["pretrained"])
        # model = nn.DataParallel(model)
        model = model.to(device)

        # Loss function
        loss_function = nn.BCELoss()

        # Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=tuple(config["betas"]), eps=config["eps"],
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
            val_loss, score = evaluate(epoch, model, validation_dataloader, device, loss_function, mb)
            writer.add_scalar("Validation loss", val_loss, epoch)
            writer.add_scalar("ROCAUC", score, epoch)
            writer.flush()
            mb.write('Finish validation epoch {} with loss {:.4f} and score {:.4f}'.format(epoch, val_loss, score))
            validation_losses.append(val_loss)
            validation_score.append(score)

            # Update training chart
            mb.update_graph([[x, training_losses], [x, validation_losses], [x, validation_score]], [0, epoch + 1], [0, 1])

            mb.write(f"AUROC: {score}")

        # Evaluating
        test_loss, score = evaluate(epoch, model, test_dataloader, device, loss_function, mb)
        return score;

        writer.close()

if __name__ == "__main__":

    # Create a new Optuna study.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # Run for 20 trials

    for trial in study.trials:
        print(trial.value, trial.params)

    # Best parameters
    print(study.best_params)
    print(study.trials_dataframe)