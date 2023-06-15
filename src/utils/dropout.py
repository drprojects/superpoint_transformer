import torch


__all__ = ['dropout']


def dropout(a, p=0.5, dim=1, inplace=False, to_mean=False):
    n = a.shape[dim]
    to_drop = torch.where(torch.rand(n, device=a.device).detach() < p)[0]
    out = a if inplace else a.clone()


    if not to_mean:
        out.index_fill_(dim, to_drop, 0)
        return out

    if dim == 1:
        out[:, to_drop] = a.mean(dim=0)[to_drop]
        return out

    out[to_drop] = a.mean(dim=0)
    return out
