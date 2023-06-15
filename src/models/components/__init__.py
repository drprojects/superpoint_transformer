"""Model components groups architectures ready to be used a `net` in a
LightningModule. These are complex architectures, on which a
LightningModule can add heads and train for different types of tasks.
"""
from .spt import *
from .mlp import *
