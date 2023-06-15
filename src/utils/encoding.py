import torch


__all__ = ['fourier_position_encoder']


def fourier_position_encoder(pos, dim, f_min=1e-1, f_max=1e1):
    """
    Heuristic: keeping ```f_min = 1 / f_max``` ensures that roughly 50%
    of the encoding dimensions are untouched and free to use. This is
    important when the positional encoding is added to learned feature
    embeddings. If the positional encoding uses too much of the encoding
    dimensions, it may be detrimental for the embeddings.

    The default `f_min` and `f_max` values are set so as to ensure
    a '~50% use of the encoding dimensions' and a '~1e-3 precision in
    the position encoding if pos is 1D'.

    :param pos: [M, M] Tensor
        Positions are expected to be in [-1, 1]
    :param dim: int
        Number of encoding dimensions, size of the encoding space. Note
        that increasing this is NOT the most direct way of improving
        spatial encoding precision or compactness. See `f_min` and
        `f_max` instead
    :param f_min: float
        Lower bound for the frequency range. Rules how much 'room' the
        positional encodings leave in the encoding space for additive
        embeddings
    :param f_max: float
        Upper bound for the frequency range. Rules how precise the
        encoding can be. Increase this if you need to capture finer
        spatial details
    :return:
    """
    assert pos.abs().max() <= 1, "Positions must be in [-1, 1]"
    assert 1 <= pos.dim() <= 2, "Positions must be a 1D or 2D tensor"

    # We preferably operate 2D tensors
    if pos.dim() == 1:
        pos = pos.view(-1, 1)

    # Make sure M divides dim
    N, M = pos.shape
    D = dim // M
    # assert dim % M == 0, "`dim` must be a multiple of the number of input spatial dimensions"
    # assert D % 2 == 0, "`dim / M` must be a even number"

    # To avoid uncomfortable border effects with -1 and +1 coordinates
    # having the same (or very close) encodings, we convert [-1, 1]
    # coordinates to [-π/2, π/2] for safety
    pos = pos * torch.pi / 2

    # Compute frequencies on a logarithmic range from f_min to f_max
    device = pos.device
    f_min = torch.tensor([f_min], device=device)
    f_max = torch.tensor([f_max], device=device)
    w = torch.logspace(f_max.log(), f_min.log(), D, device=device)

    # Compute sine and cosine encodings
    pos_enc = pos.view(N, M, 1) * w.view(1, -1)
    pos_enc[:, :, ::2] = pos_enc[:, :, ::2].cos()
    pos_enc[:, :, 1::2] = pos_enc[:, :, 1::2].sin()
    pos_enc = pos_enc.view(N, -1)

    # In case dim is not a multiple of 2 * M, we pad missing dimensions
    # with zeros
    if pos_enc.shape[1] < dim:
        zeros = torch.zeros(N, dim - pos_enc.shape[1], device=device)
        pos_enc = torch.hstack((pos_enc, zeros))

    return pos_enc
