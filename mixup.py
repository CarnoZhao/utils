import torch
import torch
import numpy as np
device = torch.device("cuda")

def mixup_data(x, y, alpha=1, use_cuda=True, x2 = None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    if x2 is None:
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    else:
        mixed_x2 = lam * x2 + (1 - lam) * x2[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, mixed_x2, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
