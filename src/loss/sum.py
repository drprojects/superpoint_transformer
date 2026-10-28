from torch import nn


__all__ = ['SumLoss']


class SumLoss(nn.Module):
    """Weighted sum of criteria computed on the same input and target,
    eg the cross-entropy plus the Lovász-softmax loss.

    Unlike `MultiLoss`, which computes one criterion per output stage,
    all the criteria here see the same logits and labels. To use it on
    a single stage of a multi-stage loss, pass an explicit `MultiLoss`
    as the model criterion, eg for the first of 3 EZ-SP stages:

    ```yaml
    criterion:
      _target_: src.loss.MultiLoss
      criteria:
        - _target_: src.loss.SumLoss
          criteria:
            - _target_: src.loss.CrossEntropyLoss
            - _target_: src.loss.LovaszLoss
          lambdas: [1, 1]
        - _target_: src.loss.CrossEntropyLoss
        - _target_: src.loss.CrossEntropyLoss
      lambdas: ${model.multi_stage_loss_lambdas}
    ```

    :param criteria: List(callable)
        List of criteria. The class weights (`weight`) only apply to
        the first one: the others, such as `LovaszLoss`, are not
        designed for class weights
    :param lambdas: List(float)
        Weight of each criterion in the sum
    :param ignore_index: int
        If not None, set as the `ignore_index` of all the criteria
    """

    def __init__(self, criteria, lambdas, ignore_index=None):
        super().__init__()
        assert len(criteria) == len(lambdas)
        self.criteria = nn.ModuleList(criteria)
        self.lambdas = lambdas
        self.reduction = 'mean'
        if ignore_index is not None:
            self.ignore_index = ignore_index

    def __len__(self):
        return len(self.criteria)

    def extra_repr(self) -> str:
        return f'lambdas={self.lambdas}'

    def forward(self, input, target):
        # The criteria have different reductions (eg 'mean' for the
        # cross-entropy, 'sum' for the Lovász-softmax loss), so there is
        # no per-item loss to return, as `loss_with_target_histogram`
        # would expect
        if self.reduction == 'none':
            raise NotImplementedError(
                f"{type(self).__name__} does not support reduction='none'. "
                f"It cannot be used with the histogram-based losses "
                f"('wce', 'kl', and the levels 1+ of 'ce_kl', 'wce_kl')")
        loss = 0
        for lamb, criterion in zip(self.lambdas, self.criteria):
            loss = loss + lamb * criterion(input, target)
        return loss

    @property
    def ignore_index(self):
        return self.criteria[0].ignore_index

    @ignore_index.setter
    def ignore_index(self, ignore_index):
        for criterion in self.criteria:
            criterion.ignore_index = ignore_index

    @property
    def weight(self):
        """Class weights, applied to the first criterion only.
        """
        return self.criteria[0].weight

    @weight.setter
    def weight(self, weight):
        """Class weights, applied to the first criterion only.
        """
        self.criteria[0].weight = weight

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        """Normal `state_dict` behavior, except for the class weights,
        saved under `prefix.weight` rather than `prefix.criteria.0.weight`,
        as for `MultiLoss`.
        """
        destination = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        destination.pop(f"{prefix}criteria.0.weight", None)
        if self.weight is not None:
            destination[f"{prefix}weight"] = self.weight
        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Normal `load_state_dict` behavior, except for the class
        weights, read from `weight`. If the `state_dict` holds no
        weight, the current one is kept.
        """
        weight = state_dict.pop('weight', None)
        state_dict.pop('criteria.0.weight', None)
        weight_bckp = self.weight
        self.weight = None
        out = super().load_state_dict(state_dict, strict=strict)
        self.weight = weight if weight is not None else weight_bckp
        return out
