import torch


class CombinedLoss:
    def __call__(self, logits, labels):
        bce = torch.nn.BCEWithLogitsLoss()(logits, labels)
        soft_dice = SoftDiceLossV1()(logits, labels)
        return soft_dice + bce


# Soft Dice Loss for binary segmentation
# Taken from https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
# v1: pytorch autograd
class SoftDiceLossV1(torch.nn.Module):
    """
    soft-dice loss, useful in binary segmentation
    """

    def __init__(self,
                 p=1,
                 smooth=1):
        super(SoftDiceLossV1, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        """
        inputs:
            logits: tensor of shape (N, H, W, ...)
            label: tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        """
        probs = torch.sigmoid(logits)
        numer = (probs * labels).sum()
        denor = (probs.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss
