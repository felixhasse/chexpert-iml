import torch
from fastprogress import progress_bar


def epoch_training(epoch, model, train_dataloader, device, loss_criteria, optimizer, mb):
    """
    Epoch training

    Paramteters
    -----------
    epoch: int
      epoch number
    model: torch Module
      model to train
    train_dataloader: Dataset
      data loader for training
    device: str
      "cpu" or "cuda"
    loss_criteria: loss function
      loss function used for training
    optimizer: torch optimizer
      optimizer used for training
    mb: master bar of fastprogress
      progress to log

    Returns
    -------
    float
      training loss
    """
    # Switch model to training mode
    model.train()
    training_loss = 0  # Storing sum of training losses

    # For each batch
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=2, warmup=6, active=10),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:

        for batch, (images, labels) in enumerate(progress_bar(train_dataloader, parent=mb)):
            # Move X, Y  to device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Clear previous gradient
            optimizer.zero_grad()

            # Feed forward the model
            print(type(labels))
            pred = model(images)
            print(type(pred))

            loss = loss_criteria(pred, labels)

            # Back propagation
            loss.backward()

            # Update parameters
            optimizer.step()

            # Update training loss after each batch
            training_loss += loss.item()

            mb.child.comment = f'Training loss {training_loss / (batch + 1)}'
            prof.step()

    del images, labels, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # return training loss
    return training_loss / len(train_dataloader)


def evaluate(epoch, model, val_loader, device, loss_criteria, mb):
    """
    Validate model on validation dataset

    Parameters
    ----------
    epoch: int
        epoch number
    model: torch Module
        model used for validation
    val_loader: Dataset
        data loader of validation set
    device: str
        "cuda" or "cpu"
    loss_criteria: loss function
      loss function used for training
    mb: master bar of fastprogress
      progress to log

    Returns
    -------
    float
        loss on validation set
    float
        metric score on validation set
    """

    # Switch model to evaluation mode
    model.eval()

    val_loss = 0  # Total loss of model on validation set
    out_pred = torch.FloatTensor().to(device)  # Tensor stores prediction values
    out_gt = torch.FloatTensor().to(device)  # Tensor stores groundtruth values

    with torch.no_grad():  # Turn off gradient
        # For each batch
        for step, (images, labels) in enumerate(progress_bar(val_loader, parent=mb)):
            # Move images, labels to device (GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Update groundtruth values
            out_gt = torch.cat((out_gt, labels), 0)

            # Feed forward the model
            ps = model(images)
            loss = loss_criteria(ps, labels)

            # Update prediction values
            out_pred = torch.cat((out_pred, ps), 0)

            # Update validation loss after each batch
            val_loss += loss
            mb.child.comment = f'Validation loss {val_loss / (step + 1)}'

    # Clear memory
    del images, labels, loss
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # return validation loss, and metric score
    return val_loss / len(val_loader) #, np.array(multi_label_auroc(out_gt, out_pred)).mean()
