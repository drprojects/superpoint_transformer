import torch.nn as nn


__all__ = ['LearnableParameter']


class LearnableParameter(nn.Parameter):
    """A simple class to be used for learnable parameters (e.g. learnable
    position encodings, queries, keys, ...). Using this is useful to use
    custom weight initialization.
    """

