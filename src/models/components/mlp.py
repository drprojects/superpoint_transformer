from torch_scatter import scatter
from torch import nn
from src.data import NAG
from src.nn import MLP, BatchNorm


__all__ = ['NodeMLP']


class NodeMLP(nn.Module):
    """Simple MLP on the handcrafted features of the level-i in a NAG.
    This is used as a baseline to test how expressive handcrafted
    features are.
    """

    def __init__(
            self, dims, level=0, activation=nn.LeakyReLU(), norm=BatchNorm,
            drop=None, norm_mode='graph'):

        super().__init__()

        self.level = level
        self.mlp = MLP(dims, activation=activation, norm=norm, drop=drop)
        self.norm_mode = norm_mode

    @property
    def out_dim(self):
        return self.mlp.out_dim

    def forward(self, nag):
        assert isinstance(nag, NAG)
        assert nag.num_levels > self.level

        # Compute node features from the handcrafted features
        norm_index = nag[self.i_level].norm_index(mode=self.norm_mode)
        x = self.mlp(nag[self.level].x, batch=norm_index)

        # If node level is 1, output level-1 features
        if self.level == 1:
            return x

        # If node level is 0, max-pool to produce level-1 features
        if self.level == 0:
            return scatter(
                x, nag[0].super_index, dim=0, dim_size=nag[1].num_nodes,
                reduce='max')

        # If node level is larger than 1, distribute parent features to
        # level-1 nodes
        super_index = nag.get_super_index(self.level, low=1)
        return x[super_index]
