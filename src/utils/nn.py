import torch
from torch import nn
from src.utils.parameter import LearnableParameter


__all__ = ['init_weights']


def init_weights(m, linear=None, rpe=None, activation='leaky_relu'):
    """Manual weight initialization. Allows setting specific init modes
    for certain modules. In particular, the linear and RPE layers are
    initialized with Xavier uniform initialization by default:
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    Supported initializations are:
      - 'xavier_uniform'
      - 'xavier_normal'
      - 'kaiming_uniform'
      - 'kaiming_normal'
      - 'trunc_normal'
    """
    from src.nn import SelfAttentionBlock

    linear = 'xavier_uniform' if linear is None else linear
    rpe = linear if rpe is None else rpe

    if isinstance(m, LearnableParameter):
        nn.init.trunc_normal_(m, std=0.02)
        return

    if isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
        return

    if isinstance(m, nn.Linear):
        _linear_init(m, method=linear, activation=activation)
        return

    if isinstance(m, SelfAttentionBlock):
        if m.k_rpe is not None:
            _linear_init(m.k_rpe, method=rpe, activation=activation)
        if m.q_rpe is not None:
            _linear_init(m.q_rpe, method=rpe, activation=activation)
        return


def _linear_init(m, method='xavier_uniform', activation='leaky_relu'):
    gain = torch.nn.init.calculate_gain(activation)

    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

    if method == 'xavier_uniform':
        nn.init.xavier_uniform_(m.weight, gain=gain)
    elif method == 'xavier_normal':
        nn.init.xavier_normal_(m.weight, gain=gain)
    elif method == 'kaiming_uniform':
        nn.init.kaiming_uniform_(m.weight, nonlinearity=activation)
    elif method == 'kaiming_normal':
        nn.init.kaiming_normal_(m.weight, nonlinearity=activation)
    elif method == "trunc_normal":
        nn.init.trunc_normal_(m.weight, std=0.02)
    else:
        raise NotImplementedError(f"Unknown initialization method: {method}")


def build_qk_scale_func(dim, num_heads, qk_scale):
    """Builds the QK-scale function that will be used to produce
    the qk-scale. This function follows the template:
        f(s), where `s` is the `edge_index[0]`
    even if it does not use it.
    """
    # If qk_scale provided, the default behavior will be
    # 1/(sqrt(dim)*sqrt(num))
    if qk_scale is None:
        def f(s):
            D = (dim // num_heads) ** -0.5
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return D * G
        return f

    # If qk_scale is provided as a scalar, it will be used as is
    if not isinstance(qk_scale, str):
        def f(s):
            return qk_scale
        return f

    # Convert input str to lowercase and remove spaces before
    # parsing
    qk_scale = qk_scale.lower().replace(' ', '')

    if qk_scale in ['d+g', 'g+d']:
        def f(s):
            D = (dim // num_heads) ** -0.5
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return D + G
        return f

    if qk_scale in ['dg', 'gd', 'd*g', 'g*d', 'd.g', 'g.d']:
        def f(s):
            D = (dim // num_heads) ** -0.5
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return D * G
        return f

    if qk_scale == 'd':
        def f(s):
            D = (dim // num_heads) ** -0.5
            return D
        return f

    if qk_scale == 'g':
        def f(s):
            G = (s.bincount() ** -0.5)[s].view(-1, 1, 1)
            return G
        return f

    raise ValueError(
        f"Unable to build QK scaling scheme for qk_scale='{qk_scale}'")
