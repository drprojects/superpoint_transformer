import torch


__all__ = ['loss_with_sample_weights', 'loss_with_target_histogram']


def loss_with_sample_weights(criterion, preds, y, weights):
    assert weights.dim() == 1
    assert preds.shape[0] == y.shape[0] == weights.shape[0]

    reduction_backup = criterion.reduction
    criterion.reduction = 'none'

    weights = weights.float() / weights.sum()

    loss = criterion(preds, y)
    loss = loss.sum(dim=1) if loss.dim() > 1 else loss
    loss = (loss * weights).sum()

    criterion.reduction = reduction_backup

    return loss


def loss_with_target_histogram(criterion, preds, y_hist):
    assert preds.dim() == 2
    assert y_hist.dim() == 2
    assert preds.shape[0] == y_hist.shape[0]

    y_mask = y_hist != 0
    logits_flat = preds.repeat_interleave(y_mask.sum(dim=1), dim=0)
    y_flat = torch.where(y_mask)[1]
    weights = y_hist[y_mask]

    loss = loss_with_sample_weights(
        criterion, logits_flat, y_flat, weights)

    return loss
