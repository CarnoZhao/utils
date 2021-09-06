import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim = self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim = self.dim))

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.log_softmax(dim = -1)
        # target = target.softmax(dim = -1)
        return torch.mean(torch.sum(-target * pred, dim = -1))