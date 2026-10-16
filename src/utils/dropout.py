import torch


__all__ = ['dropout']


def dropout(a, p=0.5, dim=1, inplace=False, to_mean=False):
    # The dropped rows/columns are handled with a broadcast mask rather
    # than with their indices, whose computation would cause a CPU-GPU
    # sync
    n = a.shape[dim]
    shape = [1] * a.dim()
    shape[dim] = n
    mask = (torch.rand(n, device=a.device) < p).view(shape)
    out = a if inplace else a.clone()

    if not to_mean:
        out.masked_fill_(mask, 0)
        return out

    # Dropped values are replaced by the mean of their column
    out.copy_(torch.where(mask, a.mean(dim=0), a))
    return out
