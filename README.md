# Implementing Informed Machine Learning Approaches for Cardiomegaly Detection

## About

This is the repository for my Bachelor's thesis which implements and evaluates different Informed Machine Learning (IML)
approaches for using prior
knowledge in the detection of cardiomegaly.

### Approaches used

Four different approaches for detecting cardiomegaly are used in this project:

1. A baseline DenseNet121 trained on the CheXpert dataset
2. Automated cardiothoracic ratio (CTR) calculation using lung and heart segmentation
2. Training on images cropped to lung bounding boxes
3. Curriculum Learning using CTR as a proxy for disease severity

## Setup

### Downloading the required data

#### Chexpert Dataset

The CheXpert dataset can be downloaded
[here](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2).
For training we used a downsampled dataset which as of 22.03.2023 is not publically available anymore. However, using
the full size dataset should not change results as all images are resized to 256x256 by default.

#### JSRT Dataset

The JSRT dataset can be downloaded [here](http://db.jsrt.or.jp/eng.php). The corresponding masks and landmarks can be
found [here](https://zenodo.org/record/7056076#.Y_0GEy8w1QI).

#### Folder Structure

TODO

### Setting up the Python Environment

TODO

## Training the models

### General

By default the name of the trained model will include the learning rate, batch size and a timestamp. You can add a
prefix to the default name using the --prefix flag in the command line or specify your own name using the --name flag.
Tensorboard can be used to track the progress and results of training.
TODO: Add table for command line arguments

### Baseline model

You can train the baseline model by simply running the train_classification.py script. Hyperparameters can be adjusted
in the classification_config.json file.

### Train segmentation models

For the IML approaches to work you first need to train the heart and lung segmentation models.
Before training them, make sure to run the preprocessing.py script once. Then you can train the model by executing
train_segmentation.py. The --target flag accepts "heart" or "lung" and specifies which segmentation model should be
trained. For using the IML approaches you need to train both the heart and lung models.

### CTR calculation

TODO

### Generate masks for the CheXpert dataset

By running the segment_chexpert.py script you can generate heart and lung masks for the CheXpert dataset.
Using these you can train the other IML approaches. This can take quite long so it's recommended to only run this once
you have trained segmentation models with satisfactory performance. TODO: Add more details

### Training on images cropped to bounding boxes

TODO

### Curriculum Learning

TODO

### Pretrained models

You can find pretrained models as well as pregenerated masks for the CheXpert dataset [here]().
TODO: Add more details

## Evaluating the results

In the src/notebooks folder there are various notebooks that allow you to evaluate and compare the results of the
different approaches. TODO: Add more details

## References

TODO

