# Implementing Informed Machine Learning Approaches for Cardiomegaly Detection

## About

This is the repository for my Bachelor's thesis which implements and evaluates different Informed Machine Learning (IML)
approaches for using prior
knowledge in the detection of cardiomegaly.

### Approaches used

Four different approaches for detecting cardiomegaly are used in this project:

1. A baseline DenseNet121 trained on the CheXpert dataset
2. Automated cardiothoracic ratio (CTR) calculation using lung and heart segmentation
3. Training on images cropped to lung bounding boxes
4. Curriculum Learning using CTR as a proxy for disease severity

## Setup

### Downloading the required data

#### Chexpert Dataset

The CheXpert dataset can be downloaded
[here](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2).
For training we used a downsampled dataset which as of 26.04.2023 is not publicly available anymore. However, using
the full size dataset should not significantly change results as all images are resized before training by default.


The JSRT dataset can be downloaded [here](http://db.jsrt.or.jp/eng.php). The corresponding masks and landmarks can be
found [here](https://zenodo.org/record/7056076#.Y_0GEy8w1QI).

#### Folder Structure

Create a folder in the project root directory called data and move the files so that it matches the following folder structure:

```
data 
│
└───CheXpert-v1.0-small
│   │   train.csv
│   │   valid.csv
│   │
│   └───train
│   │
│   └───valid
│   
└───JSRT
│   │
│   └───images
│   │
|   └───landmarks
│
└───MontgomerySet
    │
    └───CXR_png
    │
    └───ManualMask

```

### Setting up the Python Environment

You can set up a python virtual environment using the 'requirements.txt' file in the project directory by running
```
python3.10 -m venv "env_name" -r requirements.txt.
``` 
If you want to train models on a GPU using CUDA, you need to install the CUDA version of PyTorch using:
```
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

## Training the models

### Running the python scripts
All python scripts need to be run from the root folder as a module, for example:
```
python -m src.train_baseline
```
This is neccessary for relative imports to work correctly.

### Preprocessing

Before training the IML approaches, make sure to run 
```
python -m src.preprocessing
```
This converts images and masks to the needed format and creates a folder data/segmentation_dataset, which is used as the basis for training the segmentation models.

### General
Training the models takes quite long, so we recommend to use a GPU for training. However, even on powerful GPUs, training may take several hours to finish.
By default the name of the trained model will include some hyperparameters. You can add a
prefix to the default name using the --prefix flag in the command line or specify your own name using the --name flag.
Tensorboard can be used to track the progress and results of training.
TODO: Add table for command line arguments

### Tracking Training Progress and Results

The training can be tracked by running the command
```
tensorboard --logdir=runs
```
The results can then be seen in the Tensorboard dashboard by opening localhost:6006 in the browser.

### Baseline model

You can train the baseline model by simply running the train_baseline.py script. Hyperparameters can be adjusted
in the baseline_config.json file.

### Train segmentation models

For the IML approaches to work you first need to train the heart and lung segmentation models. To do that, run the train_segmentation script. Hyperparameters can be adjusted in segmentation_config.json

The --target flag accepts "heart" or "lung" and specifies which segmentation model should be
trained. For using the IML approaches you need to train both the heart and lung models.

### CTR calculation

After training the segmentation models, you can use it to calculate the CTR on the CheXpert validation set and evaluate the results. Change the paths in ctr_evaluation_config.json to the paths of the models you want to use. Then you can run the ctr_evaluation script. Results will be saved to /runs/ctr_evaluation/<name_of_models>

### Generate masks for the CheXpert dataset

By running the segment_chexpert.py script you can generate heart and lung masks for the CheXpert dataset.
Using these you can train the other IML approaches. Generating the masks takes several hours, so it's recommended to only run this once
you have trained segmentation models with satisfactory performance. TODO: Add more details

### Curriculum Learning

You can train a model using curriculum learning by running the train_curriculum_learning script. Hyperparameters can be adjusted in curriculum_learning_config.json

### Training on images cropped to bounding boxes

To crop images to lung bounding boxes for training, you can add the --crop flag when running train_baseline or train_curriculum_learning. For it to work correctly, you need to specify the directory with the lung masks in the respective config file for the baseline model or for curriculum learning

### Pretrained models

You can find pretrained models as well as pregenerated masks for the CheXpert dataset [here]().
TODO: Add more details

## References

TODO

