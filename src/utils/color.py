import torch
import numpy as np
from colorhash import ColorHash


__all__ = [
    'to_float_rgb', 'to_byte_rgb', 'rgb_to_plotly_rgb', 'int_to_plotly_rgb',
    'hex_to_tensor', 'feats_to_rgb', 'identity_PCA']


def to_float_rgb(rgb):
    rgb = rgb.float()
    if rgb.max() > 1:
        rgb = rgb / 255
    rgb = rgb.clamp(min=0, max=1)
    return rgb


def to_byte_rgb(rgb):
    if rgb.is_floating_point() and rgb.max() <= 1:
        rgb = rgb * 255
    rgb = rgb.clamp(min=0, max=255).byte()
    return rgb


def rgb_to_plotly_rgb(rgb, alpha=None):
    """Convert torch.Tensor of float RGB values in [0, 1] to
    plotly-friendly RGB format. If alpha is provided, the output will be
    expressed in RGBA format.
    """
    assert isinstance(rgb, torch.Tensor)
    assert rgb.dim() <= 2
    if rgb.dim() == 1:
        rgb = rgb.unsqueeze(0)
    if rgb.dtype in [torch.uint8, torch.int, torch.long]:
        rgb = rgb.long().numpy()
    elif rgb.is_floating_point() and rgb.max() <= 1:
        rgb = (rgb * 255).long().numpy()
    else:
        raise ValueError(
            f'Not sure how to deal with RGB of dtype={rgb.dtype} and '
            f'max={rgb.max()}')

    if alpha is None:
        return np.array([f"rgb{tuple(x)}" for x in rgb])

    if isinstance(alpha, (int, float)):
        alpha = np.array([alpha] * rgb.shape[0])
    elif isinstance(alpha, torch.Tensor):
        alpha = alpha.numpy()
    assert isinstance(alpha, np.ndarray)
    assert alpha.ndim == 1
    assert alpha.shape[0] == rgb.shape[0]

    return np.array([
        f"rgba({x[0]}, {x[1]}, {x[1]}, {a})" for x, a in zip(rgb, alpha)])


def int_to_plotly_rgb(x):
    """Convert 1D torch.Tensor of int into plotly-friendly RGB format.
    This operation is deterministic on the int values.
    """
    assert isinstance(x, torch.Tensor)
    assert x.dim() == 1
    assert not x.is_floating_point()
    x = x.cpu().long().numpy()
    palette = np.array([ColorHash(i).rgb for i in range(x.max() + 1)])
    return palette[x]


def hex_to_tensor(h):
    h = h.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return to_float_rgb(torch.tensor(rgb))


def feats_to_rgb(feats, normalize=False):
    """Convert features of the format M x N with N>=1 to an M x 3
    tensor with values in [0, 1 for RGB visualization].
    """
    is_normalized = False

    if feats.dim() == 1:
        feats = feats.unsqueeze(1)
    elif feats.dim() > 2:
        raise NotImplementedError

    if feats.shape[1] == 3:
        color = feats

    elif feats.shape[1] == 1:
        # If only 1 feature is found convert to a 3-channel
        # repetition for grayscale visualization.
        color = feats.repeat_interleave(3, 1)

    elif feats.shape[1] == 2:
        # If 2 features are found, add an extra channel.
        color = torch.cat([feats, torch.ones(feats.shape[0], 1)], 1)

    elif feats.shape[1] > 3:
        # If more than 3 features or more are found, project features to
        # a 3-dimensional space using N-simplex PCA. Heuristics for
        # clamping:
        #   - most features live in [0, 1]
        #   - most n-simplex PCA features live in [-0.5, 0.6]
        color = identity_PCA(feats, dim=3, normalize=normalize)
        color = (torch.clamp(color, -0.5, 0.6) + 0.5) / 1.1
        is_normalized = True

    if normalize and not is_normalized:
        # Unit-normalize the features in a hypercube of shared scale
        # for nicer visualizations
        high = color.max(dim=0).values
        low = color.min(dim=0).values
        color = (color - low) / (high - low)
        color[color.isnan() | color.isinf()] = 0

    return color


def identity_PCA(x, dim=3, normalize=False):
    """Reduce dimension of x based on PCA on the union of the n-simplex.
    This is a way of reducing the dimension of x while treating all
    input dimensions with the same importance, independently of the
    input distribution in x.
    """
    assert x.dim() == 2, f"Expected x.dim()=2, got x.dim()={x.dim()} instead"

    # Create z the union of the N-simplex
    input_dim = x.shape[1]
    z = torch.eye(input_dim)

    # PCA on z
    z_offset = z.mean(axis=0)
    z_centered = z - z_offset
    cov_matrix = z_centered.T.mm(z_centered) / len(z_centered)
    _, eigenvectors = torch.linalg.eigh(cov_matrix)

    # Normalize x if need be
    if normalize:
        high = x.max(dim=0).values
        low = x.min(dim=0).values
        x = (x - low) / (high - low)
        x[x.isnan() | x.isinf()] = 0

    # Apply the PCA on x
    x_reduced = (x - z_offset).mm(eigenvectors[:, -dim:])

    return x_reduced
