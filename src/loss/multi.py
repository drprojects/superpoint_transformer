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

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """Normal `state_dict` behavior, except for the shared criterion
        weights, which are not saved under `prefix.criteria.i.weight`
        but under `prefix.weight`.
        """
        destination = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars)

        # Remove the 'weight' from the criteria
        for i in range(len(self)):
            destination.pop(f"{prefix}criteria.{i}.weight")

        # Only save the global shared weight
        destination[f"{prefix}weight"] = self.weight

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Normal `load_state_dict` behavior, except for the shared
        criterion weights, which are not saved under `criteria.i.weight`
        but under `prefix.weight`.
        """
        # Get the weight from the state_dict
        old_format = state_dict.get('criteria.0.weight')
        new_format = state_dict.get('weight')
        weight = new_format if new_format is not None else old_format
        for k in [f"criteria.{i}.weight" for i in range(len(self))]:
            if k in state_dict.keys():
                state_dict.pop(k)

        # Normal load_state_dict, ignoring self.criteria.0.weight and
        # self.weight
        out = super().load_state_dict(state_dict, strict=strict)

        # Set the weight
        self.weight = weight

        return out
