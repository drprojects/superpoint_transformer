import torch
from src.utils.color import to_float_rgb


__all__ = ['rgb2hsv', 'rgb2lab']


def rgb2hsv(rgb, epsilon=1e-10):
    """Convert a 2D tensor of RGB colors int [0, 255] or float [0, 1] to
    HSV format.

    Credit: https://www.linuxtut.com/en/20819a90872275811439
    """
    assert rgb.ndim == 2
    assert rgb.shape[1] == 3

    rgb = rgb.clone()

    # Convert colors to float in [0, 1]
    rgb = to_float_rgb(rgb)

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    max_rgb, argmax_rgb = rgb.max(1)
    min_rgb, argmin_rgb = rgb.min(1)

    max_min = max_rgb - min_rgb + epsilon

    h1 = 60.0 * (g - r) / max_min + 60.0
    h2 = 60.0 * (b - g) / max_min + 180.0
    h3 = 60.0 * (r - b) / max_min + 300.0

    h = torch.stack((h2, h3, h1), dim=0).gather(
        dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)
    s = max_min / (max_rgb + epsilon)
    v = max_rgb

    return torch.stack((h, s, v), dim=1)


def rgb2lab(rgb):
    """Convert a tensor of RGB colors int[0, 255] or float [0, 1] to LAB
    colors.

    Reimplemented from:
    https://gist.github.com/manojpandey/f5ece715132c572c80421febebaf66ae
    """
    rgb = rgb.clone()
    device = rgb.device

    # Convert colors to float in [0, 1]
    rgb = to_float_rgb(rgb)

    # Prepare RGB to XYZ
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92
    rgb *= 100

    # RGB to XYZ conversion
    m = torch.tensor([
        [0.4124, 0.2126, 0.0193],
        [0.3576, 0.7152, 0.1192],
        [0.1805, 0.0722, 0.9505]], device=device)
    xyz = (rgb @ m).round(decimals=4)

    # Observer=2°, Illuminant=D6
    # ref_X=95.047, ref_Y=100.000, ref_Z=108.883
    scale = torch.tensor([[95.047, 100.0, 108.883]], device=device)
    xyz /= scale

    # Prepare XYZ for LAB
    mask = xyz > 0.008856
    xyz[mask] = xyz[mask] ** (1 / 3.)
    xyz[~mask] = 7.787 * xyz[~mask] + 1 / 7.25

    # XYZ to LAB conversion
    lab = torch.zeros_like(xyz)
    m = torch.tensor([
        [0, 500, 0],
        [116, -500, 200],
        [0, 0, -200]], device=device, dtype=torch.float)
    lab = xyz @ m
    lab[:, 0] -= 16
    lab = lab.round(decimals=4)

    return lab
