import json

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from .materials.datasets import CheXpertDataset
from .materials.custom_transformations import HistogramEqualization
from .materials.constants import *
from .materials.util import *
from .materials.metrics import *
from .materials.segmentation_inference import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

with open(CLASSIFICATION_EVALUATION_CONFIG_PATH, "r") as file:
    config = json.load(file)

# Initialize tensorboard
directory_name = config['model_path'].split('/')[-1].split('.')[0]
model_type = config['model_path'].split('/')[-2]
writer = SummaryWriter(log_dir=f"runs/evaluation/{model_type}/{directory_name}")

# Specify device for inference
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# Load model
model = load_densenet121(config['model_path'], device)

# Load CheXpert Dataset
transformation_list = [
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
]

image_transformation = transforms.Compose(transformation_list)

test_dataset = CheXpertDataset(data_path="data/CheXpert-v1.0-small/valid.csv",
                               uncertainty_policy="zeros", transform=image_transformation)

test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# Run prediction
ground_truth = torch.FloatTensor()
network_output = []
prediction = torch.FloatTensor()
results = []
for step, (image, label) in enumerate(test_dataloader):
    ground_truth = torch.cat((ground_truth, label), 0)
    image = image.to(device)
    output_for_image = model(image)
    network_output.append(output_for_image)
    prediction_for_image = torch.ones(1) if output_for_image > 0.5 else torch.zeros(1)
    prediction = torch.cat((prediction, prediction_for_image), 0)
    result_dict = {"image_path": test_dataloader.dataset.get_path(step), "ground_truth": label.item(),
                   "output": output_for_image.item(), "prediction": prediction_for_image.item()}
    results.append(result_dict)

# Calculate metrics
metrics_calculator = EvaluationMetricsCalculator(ground_truth=ground_truth, prediction=prediction)
sensitivity = metrics_calculator.sensitivity()
specificity = metrics_calculator.specificity()
precision = metrics_calculator.precision()
accuracy = metrics_calculator.accuracy()
f1_score = metrics_calculator.f1_score()
fpr, tpr, thresholds = roc_curve(ground_truth.numpy(), torch.tensor(network_output).numpy())
roc_auc = auc(fpr, tpr)

metrics_dict = {"sensitivity": sensitivity, "specificity": specificity, "precision": precision, "accuracy": accuracy,
                "f1_score": f1_score, "roc_auc": roc_auc}

# Add to tensorboard
for key in metrics_dict:
    writer.add_text(tag=key, text_string=str(metrics_dict[key]))

# Save results
with open(f"runs/evaluation/{model_type}/{directory_name}/results.json", "w") as file:
    json.dump(results, file)

with open(f"runs/evaluation/{model_type}/{directory_name}/metrics.json", "w") as file:
    json.dump(metrics_dict, file)

# Draw ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.savefig(f"runs/evaluation/{model_type}/{directory_name}/roc.png")

writer.close()
