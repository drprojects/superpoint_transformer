__all__ = ['WeightedLossMixIn']


class WeightedLossMixIn:
    """A mix-in for converting a torch loss into an item-weighted loss.
    """
    def forward(self, input, target, weight):
        if weight is not None:
            assert weight.ge(0).all(), "Weights must be positive."
            assert weight.gt(0).any(), "At least one weight must be non-zero."

        # Compute the loss, without reduction
        loss = super().forward(input, target)
        if loss.dim() == 1:
            loss = loss.view(-1, 1)

        # Sum the loss terms across the spatial dimension, so the
        # downstream averaging does not normalize by the number of
        # dimensions
        loss = loss.sum(dim=1).view(-1, 1)

        # If weights are None, fallback to normal unweighted L2 loss
        if weight is None:
            return loss.mean()

        # Compute the weighted mean
        return (loss * (weight / weight.sum()).view(-1, 1)).sum()
