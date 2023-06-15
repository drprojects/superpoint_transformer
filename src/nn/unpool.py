from torch import nn


__all__ = ['IndexUnpool']


class IndexUnpool(nn.Module):
    """Simple unpooling operation that redistributes i+1-level features
    to i-level nodes based on their indexing.
    """

    def forward(self, x, idx):
        return x.index_select(0, idx)
