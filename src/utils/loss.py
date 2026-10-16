import torch


__all__ = [
    'loss_with_sample_weights', 'loss_with_target_histogram',
    'dominant_label_histogram']


def loss_with_sample_weights(criterion, pred, y, weights):
    assert weights.dim() == 1
    assert pred.shape[0] == y.shape[0] == weights.shape[0]

    reduction_backup = criterion.reduction
    criterion.reduction = 'none'

    weights = weights.float() / weights.sum()

    loss = criterion(pred, y)
    loss = loss.sum(dim=1) if loss.dim() > 1 else loss
    loss = (loss * weights).sum()

    criterion.reduction = reduction_backup

    return loss


def loss_with_target_histogram(criterion, pred, y_hist):
    assert pred.dim() == 2
    assert y_hist.dim() == 2
    assert pred.shape[0] == y_hist.shape[0]

    y_mask = y_hist != 0
    logits_flat = pred.repeat_interleave(y_mask.sum(dim=1), dim=0)
    y_flat = torch.where(y_mask)[1]
    weights = y_hist[y_mask]

    loss = loss_with_sample_weights(
        criterion, logits_flat, y_flat, weights)

    return loss


def dominant_label_histogram(y_hist):
    """Histogram holding, for each row, all the counts of `y_hist` in
    its dominant label: `[[5, 1, 0], [0, 2, 7]]` gives
    `[[6, 0, 0], [0, 0, 9]]`. Used by the 'wce' losses.
    """
    y_dominant = y_hist.argmax(dim=1)
    out = torch.zeros_like(y_hist)
    out[torch.arange(y_hist.shape[0], device=y_hist.device), y_dominant] = \
        y_hist.sum(dim=1)
    return out
