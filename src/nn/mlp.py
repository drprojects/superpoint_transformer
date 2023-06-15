from torch import nn
from src.nn.norm import BatchNorm, INDEX_BASED_NORMS


__all__ = ['MLP', 'FFN', 'Classifier']


def mlp(
        dims, activation=nn.LeakyReLU(), last_activation=True,
        norm=BatchNorm, drop=None):
    """Helper to build MLP-like structures.

    :param dims: List[int]
        List of channel sizes. Expects `len(dims) >= 2`
    :param activation: nn.Module instance
        Non-linearity
    :param last_activation: bool
        Whether the last layer should have an activation
    :param norm: nn.Module
        Normalization. Can be None, for FFN for instance. Must be
        instantiable using norm(in_channels). If more parameters need to
        be passed to the norm, consider using a partially instantiated
        class
    :param drop: float in [0, 1]
        Dropout on the output features. No dropout layer will be
        created if `drop=None` or `drop < 0`
    :return:
    """
    assert len(dims) >= 2

    # Only use bias if no normalization is applied
    bias = norm is None

    # Iteratively build the layers based on dims
    modules = []
    for i in range(1, len(dims)):
        modules.append(nn.Linear(dims[i - 1], dims[i], bias=bias))
        if norm is not None:
            modules.append(norm(dims[i]))
        if activation is not None and (last_activation or i < len(dims) - 1):
            modules.append(activation)

    # Add final dropout if required
    if drop is not None and drop > 0:
        modules.append(nn.Dropout(drop, inplace=True))

    return nn.ModuleList(modules)


class MLP(nn.Module):
    """MLP operating on features [N, D] tensors. You can think of
    it as a series of 1x1 conv -> 1D batch norm -> activation.
    """

    def __init__(
            self, dims, activation=nn.LeakyReLU(), last_activation=True,
            norm=BatchNorm, drop=None):
        super().__init__()
        self.mlp = mlp(
            dims, activation=activation, last_activation=last_activation,
            norm=norm, drop=drop)
        self.out_dim = dims[-1]

    def forward(self, x, batch=None):
        # We need to manually iterate over the ModuleList to be able to
        # pass the batch index when need be, for some specific
        # normalization layers
        for module in self.mlp:
            if isinstance(module, INDEX_BASED_NORMS):
                x = module(x, batch=batch)
            else:
                x = module(x)
        return x


class FFN(MLP):
    """Feed-Forward Network as used in Transformers. By convention,
    these MLPs have 2 Linear layers and no normalization, the last layer
    has no activation and an optional dropout may be applied on the
    output features.
    """

    def __init__(
            self, dim, hidden_dim=None, out_dim=None, activation=nn.LeakyReLU(),
            drop=None):

        # Build the channel sizes for the 2 linear layers
        hidden_dim = hidden_dim or dim
        out_dim = out_dim or dim
        channels = [dim, hidden_dim, out_dim]

        super().__init__(
            channels, activation=activation, last_activation=False, norm=None,
            drop=drop)


class Classifier(nn.Module):
    """A simple fully-connected head with no activation and no
    normalization.
    """

    def __init__(self, in_dim, num_classes, bias=True):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes, bias=bias)

    def forward(self, x):
        return self.classifier(x)
