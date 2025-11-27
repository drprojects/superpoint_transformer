from typing import List, Tuple, Union
from omegaconf import ListConfig

from torch import nn

from torchsparse import SparseTensor
from torchsparse import nn as spnn

from src.nn.norm import GraphNorm, INDEX_BASED_NORMS, GroupNorm

import logging
log = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """
    A block that combines a sparse 3D convolution, normalization and activation.
    
    :param in_channels: int
        Number of input channels
    :param out_channels: int
        Number of output channels
    :param kernel_size: int
        Size of the convolution kernel
    :param stride: int
        Stride of the convolution
    :param dilation: int
        Dilation of the convolution
    :param bias: bool
        Whether to use bias in the convolution
    :param norm: nn.Module class
        Normalization layer to use (if None, no normalization is applied)
    :param activation: nn.Module
        Activation function to use (if None, no activation is applied)
    :param residual: bool
        Whether to use a pre-activation residual connection in the block
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        norm = None,
        activation = None,
        residual: bool = False,
    ):
        super().__init__()
        
        self.conv = spnn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias
        )
        
        self.norm = norm(out_channels) if norm is not None else None
        self.activation = activation
        self.residual = residual
        
        assert (not self.residual) or in_channels == out_channels, \
            "Residual connection in a block is only supported if the input and output channels are the same"
        
    def forward(self, x: SparseTensor, batch=None) -> SparseTensor:
        x_in = x
        x = self.conv(x)
        
        if self.norm is not None:
            if isinstance(self.norm, INDEX_BASED_NORMS):
                x.F = self.norm(x.F, batch=batch)
            else:
                x = self.norm(x)
                
        if self.residual:
            x = x + x_in
            
        if self.activation is not None:
            x = self.activation(x)
            
        return x

class SparseCNN(nn.ModuleList):
    """
    SparseCNN is a class that implements a sparse convolutional neural network.
    
    - Sparse Convolutions : https://arxiv.org/abs/1711.10275
    - TorchSparse : https://arxiv.org/abs/2311.12862v1
    
    :param cnn: List[int]
        List of channel sizes. Expects `len(cnn) >= 2`
        The first channal is the input channel size.
    :param kernel_size: int or List[int]
        The size of the kernel of the convolution.
    :param dilation: int or List[int]
        The dilation of the convolution.
    :param stride: int or List[int]
        The stride of the convolution.
        
    :param norm: nn.Module class (not instance)
        Normalization function in each block
    :param last_norm: bool
        Whether to apply the normalization function in the last block
    
    :param activation: nn.Module
        Activation function in each block
    :param last_activation: bool
        Whether to apply the activation function in the last block
        
    :param residual: bool or List[bool]
        Whether to use a pre-activation residual connection in each block of the CNN.
    :param global_residual: bool
        Whether to use a residual connection between the input and the output of the CNN.
    
    :param frozen: bool
        Whether to freeze the CNN
                
    """
    
    
    def __init__(
        self, 
        cnn: List[int], 
        kernel_size: int, 
        dilation: int,
        stride: int = 1,

        norm =  GraphNorm,
        last_norm: bool = True,
        
        activation: nn.Module = spnn.ReLU(),
        last_activation: bool = True,
        
        residual: bool = False,
        global_residual: bool = False,
        
        frozen: bool = False,
    ):
        super(SparseCNN, self).__init__()
        
        assert len(cnn) >= 2
        
        # Only use bias if no normalization is applied
        bias = norm is None
    
        self.cnn = cnn
        self.n_blocks = len(cnn) - 1
        self.global_residual = global_residual
        self.frozen = frozen
        
        assert (not self.global_residual) or cnn[0] == cnn[-1], \
            "Residual connection over the entire CNN is only supported if the input and output channels are the same"
        
        kernel_size = self.check_type(kernel_size)
        stride = self.check_type(stride)
        dilation = self.check_type(dilation)  
        residual = self.check_type(residual)
        activation = self._convert_activation(activation)
        
        if activation is None:
            log.warning("CNN activation function is `None`, no activation will be applied")
        
        if norm is None:
            log.warning("CNN normalization function is `None`, no normalization will be applied")
        
        for i,r in enumerate(residual):
            if r :
                assert cnn[i] == cnn[i+1], \
                    f"Block {i} has residual connection, but the input and output channels are not the same.\n" \
                    f"  Input channels: {cnn[i]}\n" \
                    f"  Output channels: {cnn[i+1]}"
        
        for i in range(1, len(cnn)):
            block = ConvBlock(
                in_channels=cnn[i-1],
                out_channels=cnn[i],
                kernel_size=kernel_size[i-1],
                stride=stride[i-1],
                dilation=dilation[i-1],
                bias=bias,
                norm=norm if (last_norm or i < len(cnn) - 1) else None,
                activation=activation if (last_activation or i < len(cnn) - 1) else None,
                residual=residual[i-1],
            )
            self.append(block)

    @property
    def out_dim(self):
        return self.cnn[-1] 

    def forward(self, x: SparseTensor, batch = None) -> SparseTensor:
        x_in = x
        
        
        for block in self:
            x = block(x, batch=batch)
            
        if self.global_residual:
            x = x + x_in
            
        return x

    def check_type(self, arg,):
        if isinstance(arg, int):
            return [arg] * self.n_blocks
        elif isinstance(arg, (list, ListConfig)):
            assert len(arg) == self.n_blocks, \
                f"The length of the list must be {self.n_blocks}"
            return arg
        else:
            raise ValueError(f"`kernel_size`, `stride` and `dilation` must be an int or a list of ints")


    def _convert_activation(self, activation):
        """
        The package `torchsparse` relies on `torchsparse.SparseTensor` objects, 
        which are not directly compatible with nn.Module objects.
        
        This function converts an nn.Module activation function to a `torchsparse.nn` 
        activation function.
        """
        
        if isinstance(activation, spnn.ReLU):
            return activation
        elif isinstance(activation, spnn.LeakyReLU):
            return activation
        elif isinstance(activation, spnn.SiLU):
            return activation
        
        elif isinstance(activation, nn.Module):
            if isinstance(activation, nn.ReLU):
                return spnn.ReLU()
            elif isinstance(activation, nn.LeakyReLU):
                return spnn.LeakyReLU(negative_slope=activation.negative_slope,
                                      inplace=activation.inplace)
            elif isinstance(activation, nn.SiLU):
                return spnn.SiLU(inplace=activation.inplace)
            else:
                raise ValueError(f"Invalid activation function: {activation}")
        
        elif isinstance(activation, str):
            if activation.lower() == 'relu':
                return spnn.ReLU()
            elif activation.lower() == 'leakyrelu':
                return spnn.LeakyReLU()
            elif activation.lower() == 'silu':
                return spnn.SiLU()
            else:
                raise ValueError(f"Invalid activation function: {activation}")
        elif activation is None:
            log.info("Activation function is `None`, no activation will be applied")
            return None
        else:
            raise ValueError(f"Invalid activation function: {activation}")
    
    def freeze(self):
        if not self.frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.frozen = True
            
    def unfreeze(self):
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = True
            self.frozen = False