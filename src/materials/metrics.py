import torch
from sklearn.metrics import roc_auc_score


class EvaluationMetricsCalculator:
    def __init__(self, ground_truth: torch.Tensor, prediction: torch.Tensor):
        ground_truth = ground_truth.to("cpu")
        prediction = prediction.to("cpu")
        self.true_positives = torch.sum((prediction == 1) & (ground_truth == 1)).item()
        self.true_negatives = torch.sum((prediction == 0) & (ground_truth == 0)).item()
        self.false_positives = torch.sum((prediction == 1) & (ground_truth == 0)).item()
        self.false_negatives = torch.sum((prediction == 0) & (ground_truth == 1)).item()

    def sensitivity(self):
        return self.true_positives / (self.true_positives + self.false_negatives)

    def specificity(self):
        return self.true_negatives / (self.true_negatives + self.false_positives)

    def precision(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    def accuracy(self):
        return (self.true_positives + self.true_negatives) / (
                self.true_positives + self.true_negatives + self.false_positives + self.true_negatives)

    def f1_score(self):
        return 2 * ((self.precision() * self.sensitivity()) / (self.precision() + self.sensitivity()))


def iou(ground_truth: torch.Tensor, prediction: torch.Tensor):
    ious = []
    for i in range(len(ground_truth)):
        ground_truth[i] = ground_truth[i] > 0.5
        prediction[i] = prediction[i] > 0.5
        intersection = torch.logical_and(ground_truth[i], prediction[i])
        union = torch.logical_or(ground_truth[i], prediction[i])
        ious.append(torch.sum(intersection) / torch.sum(union))

    return sum(ious) / len(ious)


def auroc(ground_truth: torch.Tensor, prediction: torch.Tensor):
    """ Calculate AUROC for each class

    Parameters
    ----------
    ground_truth: torch.Tensor
        groundtruth
    prediction: torch.Tensor
        prediction

    Returns
    -------
    list
        F1 of each class
    """
    auroc = []
    gt_np = ground_truth.to("cpu").numpy()
    pred_np = torch.sigmoid(prediction).to("cpu").numpy()
    assert gt_np.shape == pred_np.shape, "ground truth and prediction tensors should have the same shape"
    for i in range(gt_np.shape[1]):
        auroc.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return auroc
