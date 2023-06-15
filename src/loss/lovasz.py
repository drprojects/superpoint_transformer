import torch
from torch.nn.modules.loss import _Loss


__all__ = ['LovaszLoss']


class LovaszLoss(_Loss):
    """Multi-class Lovasz-Softmax loss.

    Re-implementation of:
        Lovasz-Softmax and Jaccard hinge loss in PyTorch
        Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
        Credit: https://github.com/bermanmaxim/LovaszSoftmax

    :param logits: [P, C] Tensor
        Point-wise prediction logits. Typically real numbers produced
        by the last layer of a classifier
    :param labels: [P] Tensor
        Point-wise ground truth labels (between 0 and C - 1)
    :param normalization:
        Normalization method used to convert input logits into
        probabilities
    :param class_to_sum: str or List(int) or Tensor
        Indicates which class to compute the Lovasz loss on. 'all' will
        sum the loss for all classes, 'present' will apply to classes
        which appear in the batch at hand. If a list of int is passed,
        these will be interpreted as the indices of the classes to
        consider
    :param reduction: str
        Reduction to apply to the loss. 'None' will return the
        non-aggregated, point-wise loss. 'sum' will sum the point-wise
        losses. NB: for the specific case of the Lovasz loss, the
        reduction should be the sum() and not the mean(). The complexity
        of the loss computation is such that it applying point-wise
        weights before the reduction is likely to break the loss (ie
        it is hard to define a segment-wise Lovasz loss on
        histograms...)
    :param ignore_index: int
        Class index to ignore
    :param weight: Tensor
        Class weights. Although this functionality is computationally
        sound, it has no theoretical guarantees regarding the loss
        landscape or convergence properties
    """

    def __init__(
            self, normalization='softmax', class_to_sum='present',
            reduction='sum', ignore_index=-1, weight=None):
        super().__init__(reduction=reduction)
        self.ignore_index = ignore_index
        self.normalization = normalization
        self.class_to_sum = class_to_sum
        self.weight = weight

    def forward(self, input, target):
        return lovasz(
            input, target, normalization=self.normalization,
            class_to_sum=self.class_to_sum, reduction=self.reduction,
            ignore_index=self.ignore_index, weight=self.weight)


def lovasz(
        logits, labels, normalization='softmax', class_to_sum='present',
        reduction='sum', ignore_index=-1, weight=None):
    """Multi-class Lovasz-Softmax loss.

    Re-implementation of:
        Lovasz-Softmax and Jaccard hinge loss in PyTorch
        Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
        Credit: https://github.com/bermanmaxim/LovaszSoftmax

    :param logits: [P, C] Tensor
        Point-wise prediction logits. Typically real numbers produced
        by the last layer of a classifier
    :param labels: [P] Tensor
        Point-wise ground truth labels (between 0 and C - 1)
    :param normalization:
        Normalization method used to convert input logits into
        probabilities
    :param class_to_sum: str or List(int) or Tensor
        Indicates which class to compute the Lovasz loss on. 'all' will
        sum the loss for all classes, 'present' will apply to classes
        which appear in the batch at hand. If a list of int is passed,
        these will be interpreted as the indices of the classes to
        consider
    :param reduction: str
        Reduction to apply to the loss. 'None' will return the
        non-aggregated, point-wise loss. 'sum' will sum the point-wise
        losses. NB: for the specific case of the Lovasz loss, the
        reduction should be the sum() and not the mean(). The complexity
        of the loss computation is such that it applying point-wise
        weights before the reduction is likely to break the loss (ie
        it is hard to define a segment-wise Lovasz loss on
        histograms...)
    :param ignore_index: int
        Class index to ignore
    :param weight: Tensor
        Class weights. Although this functionality is computationally
        sound, it has no theoretical guarantees regarding the loss
        landscape or convergence properties
    """
    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    assert not labels.is_floating_point()
    assert logits.shape[1] > 1

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Exclude the 0-point edge case
    if logits.numel() == 0:
        return logits * 0.

    # Initialize class weights to 1s if not provided
    class_weight = torch.ones_like(logits[0]) if weight is None else weight

    # Remove the unnecessary data based on ignore_index
    point_mask = labels != ignore_index
    logits = logits[point_mask]
    labels = labels[point_mask]
    if 0 <= ignore_index < logits.shape[1]:
        class_mask = [c != ignore_index for c in range(logits.shape[1])]
        logits = logits[:, class_mask]
        class_weight = class_weight[class_mask]

    # Initialize some shared parameters
    device = logits.device
    num_classes = logits.shape[1]

    # Again, exclude the 0-point situation, in case the point_mask
    # removed the only points we initially had
    if logits.numel() == 0:
        return logits * 0.

    # Convert logits to probabilities
    if normalization == 'softmax':
        probas = logits.float().softmax(dim=1)
    elif logits.ge(0).all():
        probas = logits.float() / logits.sum(dim=1).view(-1, 1)
    else:
        raise ValueError('logits must all be positive')

    # One-hot encode the labels and compute the class-wise errors, for
    # each point
    fg = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    errors = (fg - probas).abs()

    # If required, mask out classes that are not present or that are
    # explicitly excluded from class_to_sum
    if class_to_sum == 'all':
        class_mask = torch.ones(num_classes, device=device, dtype=torch.bool)
    elif class_to_sum == 'present':
        class_mask = fg.sum(dim=0) > 0
    else:
        class_mask = torch.zeros(num_classes, device=device, dtype=torch.bool)
        class_mask[class_to_sum] = True
    fg = fg[:, class_mask]
    errors = errors[:, class_mask]
    class_weight = class_weight[class_mask]

    # Sort by descending order of error, for each class
    errors, perm = errors.sort(dim=0, descending=True)
    fg = torch.gather(fg, 0, perm)

    # Compute the final loss
    loss = (errors * lovasz_gradient(fg))
    loss = loss * class_weight.view(1, -1)
    if reduction == 'sum':
        return loss.mean(dim=1).sum()
    else:
        inv_perm = perm.argsort(dim=0)
        return loss.gather(0, inv_perm).mean(dim=1)


def lovasz_gradient(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.
    """
    gts = gt_sorted.sum(dim=0).view(1, -1)
    intersection = gts - gt_sorted.float().cumsum(dim=0)
    union = gts + (1 - gt_sorted).float().cumsum(dim=0)
    jaccard = 1. - intersection / union
    if gt_sorted.shape[0] > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard
