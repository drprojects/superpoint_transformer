import torch
from torch import nn
from torch_scatter import scatter
from src.utils import scatter_mean_weighted
from torch_geometric.nn.norm import LayerNorm, InstanceNorm, GraphNorm
from torch_geometric.nn.inits import ones, zeros
from torch_geometric.utils import degree


__all__ = [
    'BatchNorm',
    'UnitSphereNorm',
    'LayerNorm',
    'InstanceNorm',
    'GraphNorm',
    'GroupNorm',
    'INDEX_BASED_NORMS']


class BatchNorm(nn.Module):
    """Credits: https://github.com/torch-points3d/torch-points3d
    """

    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, **kwargs)

    def _forward_dense(self, x):
        return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def _forward_sparse(self, x):
        """Batch norm 1D is not optimised for 2D tensors. The first
        dimension is supposed to be the batch and therefore not very
        large. So we introduce a custom version that leverages
        BatchNorm1D in a more optimised way.
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze(dim=2)

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError(
                "Non supported number of dimensions {}".format(x.dim()))


class UnitSphereNorm(nn.Module):
    """Normalize positions of same-segment nodes in a unit sphere of
    diameter 1 (i.e. radius 1/2).

    :param log_diameter: bool
        Whether the returned diameter should be log-normalized. This may
        be useful if using the diameter as a feature in downstream
        learning tasks
    """

    def __init__(self, log_diameter=False):
        super().__init__()
        self.log_diameter = log_diameter

    def forward(self, pos, idx, w=None, num_super=None):
        # if w is not None:
        #     assert w.ge(0).all() and w.sum() > 0, \
        #         "At least one node must had a strictly positive weights"

        # Normalization
        if idx is None:
            pos, diameter = self._forward(pos, w=w)
        else:
            pos, diameter = self._forward_scatter(
                pos, idx, w=w, num_super=num_super)

        # Log-normalize the diameter if required. This facilitates using
        # the diameter as a feature in downstream learning tasks
        if self.log_diameter:
            diameter = torch.log(diameter + 1)

        return pos, diameter

    def _forward(self, pos, w=None):
        """Forward without scatter operations, in case `idx` is not
        provided. Applies the sphere normalization on all pos
        coordinates together.
        """
        # Compute the diameter (i.e. the maximum span along the main axes
        # here)
        min_ = pos.min(dim=0).values
        max_ = pos.max(dim=0).values
        diameter = (max_ - min_).max()

        # Compute the center of the nodes. If node weights are provided,
        # the weighted mean is computed
        if w is None:
            center = pos.mean(dim=0)
        else:
            w_sum = w.float().sum()
            w_sum = 1 if w_sum == 0 else w_sum
            center = (pos * w.view(-1, 1).float()).sum(dim=0) / w_sum
        center = center.view(1, -1)

        # Unit-sphere normalization
        pos = (pos - center) / (diameter + 1e-2)

        return pos, diameter.view(1, 1)

    def _forward_scatter(self, pos, idx, w=None, num_super=None):
        """Forward with scatter operations, in case `idx` is provided.
        Applies the sphere normalization for each segment separately.
        """
        # Compute the diameter (i.e. the maximum span along the main axes
        # here)
        min_segment = scatter(pos, idx, dim=0, dim_size=num_super, reduce='min')
        max_segment = scatter(pos, idx, dim=0, dim_size=num_super, reduce='max')
        diameter_segment = (max_segment - min_segment).max(dim=1).values

        # Compute the center of the nodes. If node weights are provided,
        # the weighted mean is computed
        if w is None:
            center_segment = scatter(
                pos, idx, dim=0, dim_size=num_super, reduce='mean')
        else:
            center_segment = scatter_mean_weighted(
                pos, idx, w, dim_size=num_super)

        # Compute per-node center and diameter
        center = center_segment[idx]
        diameter = diameter_segment[idx]

        # Unit-sphere normalization
        pos = (pos - center) / (diameter.view(-1, 1) + 1e-2)

        return pos, diameter_segment.view(-1, 1)


class GroupNorm(torch.nn.Module):
    """Group normalization on graphs.

    :param in_channels: int
        Number of input channels
    :param num_groups: int
        Number of groups. Must be a divider of `in_channels`
    :param eps: float
    :param affine: bool
    :param mode: str
        Normalization mode. `mode='graph'` will normalize input nodes
        based on the input `batch` they belong to. `mode='node'` will
        apply BatchNorm on each node separately.
    """
    def __init__(
            self, in_channels, num_groups=4, eps=1e-5, affine=True,
            mode='graph'):
        super().__init__()

        assert in_channels % num_groups == 0, \
            f"`in_channels` must be a multiple of `num_groups`"
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_channels = in_channels // num_groups
        self.eps = eps
        self.mode = mode

        if affine:
            self.weight = nn.Parameter(torch.Tensor(in_channels))
            self.bias = nn.Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        ones(self.weight)
        zeros(self.bias)

    def forward(self, x, batch=None):
        """"""
        if self.mode == 'graph':

            # If graph-wise normalization mode and 'batch' is not
            # provided, we consider all input nodes to belong to the
            # same graph
            if batch is None:
                batch = torch.zeros(
                    x.shape[0], dtype=torch.long, device=x.device)

            # Separate group features using a new dimension
            x = x.view(-1, self.num_groups, self.group_channels)

            # Compute the number of items in each group normalization
            batch_size = int(batch.max()) + 1
            norm = degree(batch, batch_size, dtype=x.dtype).clamp_(min=1)
            norm = norm.mul_(self.group_channels).view(-1, 1, 1)

            # Compute the groupwise mean
            mean = scatter(
                x, batch, dim=0, dim_size=batch_size, reduce='add').sum(
                dim=-1, keepdim=True) / norm

            # Groupwise mean-centering
            x = x - mean.index_select(0, batch)

            # Compute the groupwise variance
            var = scatter(
                x * x, batch, dim=0, dim_size=batch_size, reduce='add').sum(
                dim=-1, keepdim=True) / norm

            # Groupwise std scaling
            out = x / (var + self.eps).sqrt().index_select(0, batch)

            # Restore input shape
            out = out.view(-1, self.in_channels)

            # Apply learnable mean and variance to each channel
            if self.weight is not None and self.bias is not None:
                out = out * self.weight + self.bias

            return out

        # GroupNorm in a node wise fashion
        if self.mode == 'node':
            if batch is None:
                out = nn.functional.group_norm(
                    x, self.num_groups, weight=self.weight, bias=self.bias,
                    eps=self.eps)
                return out

        raise ValueError(f"Unknown normalization mode: {self.mode}")

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'num_groups={self.num_groups}, mode={self.mode})')


INDEX_BASED_NORMS = (LayerNorm, InstanceNorm, GraphNorm, GroupNorm)
