import torch
import torch
import numpy as np
device = torch.device("cuda")

def mixup_data(x, y, alpha=1, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_feature(x, alpha=1, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, index, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class MixUpLoss(torch.nn.Module):
    def __init__(self, criterion):
        super(MixUpLoss, self).__init__()
        self.criterion = criterion

    def forward(self, pred, y, index = None, lam = None, mix = False):
        if mix:
            y_a, y_b = y, y[index]
            return mixup_criterion(self.criterion, pred, y_a, y_b, lam)
        else:
            return self.criterion(pred, y)