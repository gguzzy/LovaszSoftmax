"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds: torch.Tensor, labels: torch.Tensor, EMPTY=1., ignore=None, per_image=True) -> float:
    """
    Updated IoU for foreground class to handle mean calculation robustly.
    """
    preds, labels = preds.view(-1), labels.view(-1)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        iou = torch.tensor(EMPTY) if not union else intersection.float() / union.float()
        ious.append(iou)

    ious_tensor = torch.stack(ious)
    valid_ious = ious_tensor[torch.isfinite(ious_tensor)]  # Exclude potential NaN/Inf values
    mean_iou = torch.nanmean(valid_ious) if valid_ious.numel() > 0 else torch.tensor(EMPTY)

    return 100 * mean_iou.item()


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class, adapted to use torch_mean for calculating the mean.
    """
    ious = []
    for c in range(C):
        if c == ignore:
            continue
        intersection = ((labels == c) & (preds == c)).sum(dim=0)
        union = ((labels == c) | ((preds == c) & (labels != ignore))).sum(dim=0)
        iou = torch.where(union == 0, torch.tensor([EMPTY], device=union.device), intersection.float() / union.float())
        ious.append(iou)

    # Here we use torch_mean to calculate the mean IoU across all classes
    ious_tensor = torch.stack(ious)  # Stack all IoU values to create a tensor
    mean_iou = torch_mean(ious_tensor, ignore_nan=True)  # Calculate mean IoU, ignoring NaN values
    return 100 * mean_iou.item()  # Convert to percentage and scalar

# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Updated Binary Lovasz hinge loss to use torch.nanmean for handling NaN values or empty inputs.
    """
    losses = []
    if per_image:
        for log, lab in zip(logits, labels):
            log = log.unsqueeze(0)
            lab = lab.unsqueeze(0)
            vlogits, vlabels = flatten_binary_scores(log, lab, ignore)
            loss = lovasz_hinge_flat(vlogits, vlabels)
            losses.append(loss)
    else:
        vlogits, vlabels = flatten_binary_scores(logits, labels, ignore)
        loss = lovasz_hinge_flat(vlogits, vlabels)
        losses.append(loss)

    losses_tensor = torch.stack(losses)
    # Use torch.nanmean to compute the mean loss, ignoring NaN values.
    return torch.nanmean(losses_tensor)

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        # Instead of directly returning the mean, use torch.nanmean to ignore NaN values.
        # This is beneficial if there's a concern that the loss calculation could produce NaNs.
        # Note: torch.nanmean is available in PyTorch 1.8 and later.
        return torch.nanmean(loss)


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Updated Multi-class Lovasz-Softmax loss to use torch_mean for handling NaN values or empty inputs.
    """
    def compute_loss_for_single_image(prob, lab):
        vprobas, vlabels = flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore)
        return lovasz_softmax_flat(vprobas, vlabels, classes=classes)

    if per_image:
        losses = torch.stack([compute_loss_for_single_image(prob, lab) for prob, lab in zip(probas, labels)])
        loss = torch_mean(losses, ignore_nan=True)  # Use torch_mean to handle potential NaN values gracefully
    else:
        vprobas, vlabels = flatten_probas(probas, labels, ignore)
        loss = lovasz_softmax_flat(vprobas, vlabels, classes=classes)
    return loss

def lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes='present') -> torch.Tensor:
    if probas.numel() == 0:
        # Directly return if probas is empty to avoid unnecessary computation
        return torch.tensor(0., device=probas.device)
    C = probas.size(1)  # Number of classes

    # Determine classes to consider based on the 'classes' parameter
    if classes == 'present':
        class_to_sum = labels.unique()
    elif classes == 'all':
        class_to_sum = torch.arange(C, device=probas.device)
    else:
        class_to_sum = torch.tensor(classes, device=probas.device)

    losses = torch.empty(len(class_to_sum), device=probas.device)

    for i, c in enumerate(class_to_sum):
        fg = (labels == c).float()  # Foreground mask for class c
        if fg.sum() == 0 and classes == 'present':
            continue  # Skip if class c is not present
        class_pred = probas[:, c]  # Predictions for class c
        errors = (fg - class_pred).abs()  # Absolute errors
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)  # Calculate gradient for sorted errors
        losses[i] = torch.dot(errors_sorted, grad)  # Compute the dot product

    # Compute the mean of losses while ensuring no division by zero or NaN issues
    valid_losses = losses[losses.isfinite()]  # Filter out any potential NaN/Inf in losses
    if valid_losses.numel() == 0:
        return torch.tensor(0., device=probas.device)  # Return 0 if all losses are NaN/Inf
    return valid_losses.mean()  # Return the mean of valid losses


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions and labels in the batch.
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # Reshape to [B*H*W, C]
    labels = labels.view(-1)

    if ignore is not None:
        valid = (labels != ignore)
        vprobas = probas[valid]
        vlabels = labels[valid]
        return vprobas, vlabels
    return probas, labels


def xloss(logits: torch.Tensor, labels: torch.Tensor, ignore=None) -> torch.Tensor:
    """
    Computes the cross-entropy loss while ignoring the specified label.
    """
    return F.cross_entropy(logits, labels, ignore_index=ignore)

# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x: torch.Tensor) -> torch.Tensor:
    """
    Checks if the input tensor contains any NaN values.
    """
    return torch.isnan(x)


# def mean(l, ignore_nan=False, empty=0):
#     """
#     nanmean compatible with generators.
#     """
#     l = iter(l)
#     if ignore_nan:
#         # l = filter(lambda x: x == x, l)  # filters out NaN values
#         l = ifilterfalse(isnan, l)
#     try:
#         n = 1
#         acc = next(l)
#     except StopIteration:
#         if empty == 'raise':
#             raise ValueError('Empty mean')
#         return empty
#     for n, v in enumerate(l, 2):
#         acc += v
#     if n == 1:
#         return acc
#     return acc / n


def torch_mean(l, ignore_nan=False, empty=0.0):
    """
    Computes the mean of a list or tensor, with options to ignore NaN values and handle empty inputs.
    - l: input list or tensor.
    - ignore_nan: if True, NaN values are ignored.
    - empty: value to return if the input is empty or all NaN (when ignore_nan is True).
    """
    if isinstance(l, list):
        l = torch.tensor(l, dtype=torch.float32)

    if l.numel() == 0:  # Check if the tensor is empty
        return torch.tensor(empty, device=l.device)

    if ignore_nan:
        l = l[torch.isfinite(l)]  # Filter out NaN and Inf values

    if l.numel() == 0:
        return torch.tensor(empty, device=l.device)

    return torch.mean(l)