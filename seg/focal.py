import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma = 2, alpha = None, ignore_index = 255):
        super().__init__(weight = alpha, ignore_index = ignore_index)
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss)

class DiceLoss(nn.Module):
    def __init__(self, num_classes = 2):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = 1e-6

    def forward(self, output, target):
        output = output.argmax(1)
        dice_target = target.contiguous().view(1, -1).float()
        dice_output = output.contiguous().view(1, -1)
        loss = 0
        for c in range(self.num_classes):
            dice_output_c = dice_output == c
            dice_target_c = dice_target == c
            intersection = torch.sum(dice_output_c * dice_target_c, dim=1)
            union = torch.sum(dice_output_c, dim=1) + torch.sum(dice_target_c, dim=1) + self.eps
            loss += (1 - (2 * intersection + self.eps) / union).mean()
        return loss / self.num_classes

def tversky_loss(true, logits, alpha = 0.7, beta = 0.3, gamma = 4/3, eps=1e-7, no_back = False):
    """Computes the Tversky loss [1].
    Args:
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.
    Returns:
        tversky_loss: the Tversky loss.
    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
    References:
        [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true[true == 255] = num_classes + 1
        true_1_hot = torch.eye(num_classes + 2)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()[:, :num_classes+1]
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true[true == 255] = num_classes
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()[:, :num_classes]
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension() + 1))
    intersection = torch.sum(probas * true_1_hot, dims)
    fps = torch.sum(probas * (1 - true_1_hot), dims)
    fns = torch.sum((1 - probas) * true_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps))
    if no_back:
        tversky_loss = tversky_loss[1:]
    return ((1 - tversky_loss) ** gamma).mean()

import torch
a = torch.rand(2, 7, 4, 4)
b = torch.randint(0, 8, (2, 4, 4))

tversky_loss(b, a, 0.3, 0.7)

class FocalTverskyLoss(nn.Module):
    def __init__(self, num_classes = 1, no_back = False):
        super(FocalTverskyLoss, self).__init__()
        self.num_classes = num_classes
        self.no_back = no_back

    def forward(self, inputs, targets):
        loss = tversky_loss(targets, inputs, no_back=self.no_back)
        return loss


# def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
#               Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
#       labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
#       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#       per_image: compute the loss per image instead of per batch
#       ignore: void class labels
#     """
#     if per_image:
#         loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
#                           for prob, lab in zip(probas, labels))
#     else:
#         loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
#     return loss


# def lovasz_softmax_flat(probas, labels, classes='present'):
#     """
#     Multi-class Lovasz-Softmax loss
#       probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
#       labels: [P] Tensor, ground truth labels (between 0 and C - 1)
#       classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
#     """
#     if probas.numel() == 0:
#         # only void pixels, the gradients should be 0
#         return probas * 0.
#     C = probas.size(1)
#     losses = []
#     class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
#     for c in class_to_sum:
#         fg = (labels == c).float() # foreground for class c
#         if (classes is 'present' and fg.sum() == 0):
#             continue
#         if C == 1:
#             if len(classes) > 1:
#                 raise ValueError('Sigmoid output possible only with 1 class')
#             class_pred = probas[:, 0]
#         else:
#             class_pred = probas[:, c]
#         errors = (Variable(fg) - class_pred).abs()
#         errors_sorted, perm = torch.sort(errors, 0, descending=True)
#         perm = perm.data
#         fg_sorted = fg[perm]
#         losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
#     return mean(losses)
