from torch import nn


__all__ = ['MultiLoss']


class MultiLoss(nn.Module):
    """Wrapper to compute the weighted sum of multiple criteria

    :param criteria: List(callable)
        List of criteria
    :param lambdas: List(str)

    """

    def __init__(self, criteria, lambdas):
        super().__init__()
        assert len(criteria) == len(lambdas)
        self.criteria = nn.ModuleList(criteria)
        self.lambdas = lambdas

    def __len__(self):
        return len(self.criteria)

    def to(self, *args, **kwargs):
        for i in range(len(self)):
            self.criteria[i] = self.criteria[i].to(*args, **kwargs)
            self.lambdas[i] = self.lambdas[i].to(*args, **kwargs)

    def extra_repr(self) -> str:
        return f'lambdas={self.lambdas}'

    def forward(self, a, b, **kwargs):
        loss = 0
        for lamb, criterion, a_, b_ in zip(self.lambdas, self.criteria, a, b):
            loss = loss + lamb * criterion(a_, b_, **kwargs)
        return loss

    @property
    def weight(self):
        """MultiLoss supports `weight` if all its criteria support it.
        """
        return self.criteria[0].weight

    @weight.setter
    def weight(self, weight):
        """MultiLoss supports `weight` if all its criteria support it.
        """
        for i in range(len(self)):
            self.criteria[i].weight = weight
