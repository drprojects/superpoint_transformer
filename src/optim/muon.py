"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz).

References:
  - Keller Jordan et al., "Muon: An optimizer for hidden layers in
    neural networks", 2024. https://kellerjordan.github.io/posts/muon/
  - Liu et al., "Muon is Scalable for LLM Training", 2025.
    https://arxiv.org/abs/2502.16982 (decoupled weight decay and update
    RMS matching, so that AdamW's lr/weight decay can be reused as is)

For a 2D weight, Muon replaces the SGD-momentum update by its nearest
semi-orthogonal matrix (all singular values pushed towards 1), computed
with a few Newton-Schulz iterations. Muon only makes sense for 2D hidden
weights. Everything else (biases, norms, embeddings, classifier heads,
conv kernels...) must be optimized with AdamW, as recommended since
the original Muon. Our `Muon` optimizer is thus hybrid: it holds both
kinds of param groups, so that LR schedulers, warmup and LR monitoring
keep working on a single optimizer. NB: unlike `torch.optim.Muon`, which
only accepts 2D params.

Group flags:
  - `use_muon=True`: Muon update, the param must be at least 2D
  - `use_muon=False`: AdamW update
  - `row_splits` (Muon groups only, optional): list aligned with
    `params`, holding for each param either `None` or a sequence of row
    counts. A param with `row_splits` is cut along dim 0 into blocks of
    these sizes, each block being orthogonalized (and scaled) on its
    own. This is meant for fused projections such as `qkv`, whose
    q/k/v blocks should not be coupled by a joint orthogonalization.

Use `split_muon_param_groups` to build such groups from a model.
"""
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer


__all__ = [
    'zeropower_via_newtonschulz5',
    'Muon',
    'is_muon_optimizer',
    'split_muon_param_groups',
    'qkv_row_splits',
    'DEFAULT_MUON_EXCLUDE']


# Quintic Newton-Schulz coefficients of Keller Jordan. They are tuned
# to maximize the slope at zero rather than to converge exactly: the
# singular values end up in ~[0.7, 1.2] instead of exactly 1, which is
# good enough (and the paper's results rely on it)
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)

# Substrings of parameter names that must NOT be optimized by Muon,
# even if 2D: classifier/prediction heads
DEFAULT_MUON_EXCLUDE = ('head', 'classifier')

ADJUST_LR_FNS = ('match_rms_adamw', 'original')


def zeropower_via_newtonschulz5(
        G: Tensor,
        steps: int = 5,
        eps: float = 1e-7,
        coefficients: Tuple[float, float, float] = NS_COEFFICIENTS
) -> Tensor:
    """Approximate the orthogonalization `U V^T` of `G = U S V^T`
    with `steps` quintic Newton-Schulz iterations, run in bfloat16.

    :param G: Tensor
        2D matrix
    :param steps: int
        Number of iterations
    :param eps: float
        Numerical stability for the initial normalization
    :param coefficients: tuple
        (a, b, c) of the iteration `X <- aX + b(XX^T)X + c(XX^T)^2 X`
    """
    assert G.ndim == 2, f"Expected a 2D matrix, got shape {tuple(G.shape)}"
    a, b, c = coefficients
    X = G.to(torch.bfloat16)

    # Work on the smallest Gram matrix
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T

    # Ensure the spectral norm is <= 1, required for convergence
    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X


def _adjusted_lr(lr: float, shape: Sequence[int], adjust_lr_fn: str) -> float:
    """Per-matrix LR scaling, `A x B` being the (2D-flattened) shape.

    - 'match_rms_adamw' (Liu et al.): 0.2 * sqrt(max(A, B)). Makes the
      update RMS match AdamW's (~0.2), so the AdamW `lr` and
      `weight_decay` can be reused directly for Muon.
    - 'original' (Jordan): sqrt(max(1, A / B)), the update has unit
      spectral-ish scale, a much larger `lr` (~0.02) is expected.
    """
    A, B = shape[0], int(math.prod(shape[1:]))
    if adjust_lr_fn == 'match_rms_adamw':
        return lr * 0.2 * math.sqrt(max(A, B))
    if adjust_lr_fn == 'original':
        return lr * math.sqrt(max(1, A / B))
    raise ValueError(
        f"Unknown adjust_lr_fn '{adjust_lr_fn}', expected one of "
        f"{ADJUST_LR_FNS}")


class Muon(Optimizer):
    """Hybrid Muon/AdamW optimizer, single-device: Muon on the param
    groups flagged `use_muon=True`, AdamW on the others. See the module
    docstring and `configs/optimizer/muon.yaml`.

    Param groups without `use_muon` default to AdamW. Group-level `lr`
    and `weight_decay` can be overridden as usual with any optimizer.
    Weight decay is decoupled (AdamW-style) for both kinds of groups.

    :param params: iterable of tensors or param group dicts
    :param lr: float
        Learning rate, shared by Muon and AdamW groups by default. With
        `adjust_lr_fn='match_rms_adamw'`, the AdamW LR can be reused for
        Muon. With 'original', Muon expects a much larger LR (~0.02)
        than AdamW: scale AdamW groups with `split_muon_param_groups`
    :param weight_decay: float
    :param momentum: float
        Muon momentum
    :param nesterov: bool
        Muon Nesterov momentum
    :param ns_steps: int
        Muon Newton-Schulz iterations
    :param adjust_lr_fn: str
        Muon per-matrix LR scaling, among ['match_rms_adamw', 'original']
    :param betas: tuple
        AdamW betas
    :param eps: float
        AdamW epsilon
    """

    def __init__(
            self,
            params,
            lr: float = 1e-3,
            weight_decay: float = 0.1,
            momentum: float = 0.95,
            nesterov: bool = True,
            ns_steps: int = 5,
            adjust_lr_fn: str = 'match_rms_adamw',
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0 <= momentum < 1:
            raise ValueError(f"Invalid momentum: {momentum}")
        if adjust_lr_fn not in ADJUST_LR_FNS:
            raise ValueError(
                f"Unknown adjust_lr_fn '{adjust_lr_fn}', expected one of "
                f"{ADJUST_LR_FNS}")
        defaults = dict(
            lr=lr, weight_decay=weight_decay, momentum=momentum,
            nesterov=nesterov, ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn, betas=betas, eps=eps,
            use_muon=False)
        super().__init__(params, defaults)

        for group in self.param_groups:
            if not group['use_muon']:
                continue
            for p in group['params']:
                if p.ndim < 2:
                    raise ValueError(
                        f"Muon only applies to >=2D params but got a "
                        f"param of shape {tuple(p.shape)}. Put it in an "
                        f"AdamW group (use_muon=False).")
            splits = group.get('row_splits')
            if splits is None:
                continue
            if len(splits) != len(group['params']):
                raise ValueError(
                    "`row_splits` must have one item per param")
            for p, sizes in zip(group['params'], splits):
                if sizes is not None and sum(sizes) != p.shape[0]:
                    raise ValueError(
                        f"`row_splits` {tuple(sizes)} does not sum to the "
                        f"{p.shape[0]} rows of the param")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            splits = group.get('row_splits') or [None] * len(group['params'])
            for p, row_splits in zip(group['params'], splits):
                if p.grad is None:
                    continue
                if group['use_muon']:
                    self._muon_update(p, group, row_splits)
                else:
                    self._adamw_update(p, group)
        return loss

    def _muon_update(
            self,
            p: Tensor,
            group: Dict,
            row_splits: Optional[Sequence[int]] = None
    ) -> None:
        if p.grad.is_sparse:
            raise RuntimeError("Muon does not support sparse gradients")

        # Momentum, in the param's dtype (fp32 for fp32 params)
        state = self.state[p]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p.grad)
        buf = state['momentum_buffer']
        buf.lerp_(p.grad, 1 - group['momentum'])
        update = p.grad.lerp(buf, group['momentum']) \
            if group['nesterov'] else buf

        # Orthogonalize, row blocks (e.g. q/k/v) being independent and
        # conv-like params being seen as (out, -1)
        lr = group['lr']
        p.mul_(1 - lr * group['weight_decay'])
        blocks = update.split(row_splits, dim=0) if row_splits \
            else (update,)
        start = 0
        for block in blocks:
            block_update = zeropower_via_newtonschulz5(
                block.reshape(block.shape[0], -1), steps=group['ns_steps'])
            scale = _adjusted_lr(lr, block.shape, group['adjust_lr_fn'])
            p[start:start + block.shape[0]].add_(
                block_update.reshape(block.shape).to(p.dtype), alpha=-scale)
            start += block.shape[0]

    def _adamw_update(self, p: Tensor, group: Dict) -> None:
        if p.grad.is_sparse:
            raise RuntimeError("AdamW does not support sparse gradients")

        state = self.state[p]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
        state['step'] += 1
        step = state['step']
        beta1, beta2 = group['betas']
        lr = group['lr']

        state['exp_avg'].lerp_(p.grad, 1 - beta1)
        state['exp_avg_sq'].mul_(beta2).addcmul_(
            p.grad, p.grad, value=1 - beta2)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)
                 ).add_(group['eps'])

        p.mul_(1 - lr * group['weight_decay'])
        p.addcdiv_(state['exp_avg'], denom, value=-lr / bias_correction1)


def is_muon_optimizer(optimizer) -> bool:
    """Whether a (possibly `functools.partial`) optimizer class builds a
    `Muon` optimizer."""
    cls = getattr(optimizer, 'func', optimizer)
    return isinstance(cls, type) and issubclass(cls, Muon)


def _use_muon(
        name: str,
        p: Tensor,
        exclude: Sequence[str]
) -> bool:
    """Muon is for 2D weight matrices only, not for heads/embeddings.
    Conv-like kernels (e.g. the 3D TorchSparse ones) are left to AdamW,
    as are all 1D params (biases, norms)."""
    return p.ndim == 2 and not any(e in name for e in exclude)


def qkv_row_splits(model: torch.nn.Module) -> Dict[str, Tuple[int, ...]]:
    """Row blocks of the fused `qkv` weights of a model, keyed by
    parameter name. Looks for submodules with a `qkv` linear and the
    `qk_dim`, `num_heads` and `dim` attributes of
    `src.nn.attention.SelfAttentionBlock`, whose `qkv` rows are
    `[queries | keys | values]` of sizes
    `(qk_dim * num_heads, qk_dim * num_heads, dim)`. Note that the
    blocks are only equal when `qk_dim * num_heads == dim`.
    """
    splits = {}
    for name, module in model.named_modules():
        if not all(hasattr(module, k)
                   for k in ('qkv', 'qk_dim', 'num_heads', 'dim')):
            continue
        dh = module.qk_dim * module.num_heads
        sizes = (dh, dh, module.dim)
        assert sum(sizes) == module.qkv.weight.shape[0], \
            f"Unexpected qkv layout in '{name}'"
        splits[f"{name}.qkv.weight" if name else "qkv.weight"] = sizes
    return splits


def split_muon_param_groups(
        named_groups: Iterable[Dict],
        exclude: Sequence[str] = DEFAULT_MUON_EXCLUDE,
        adamw_lr_scale: float = 1.0,
        row_splits: Optional[Dict[str, Sequence[int]]] = None
) -> List[Dict]:
    """Split param groups into Muon and AdamW groups for the hybrid
    `Muon` optimizer.

    :param named_groups: iterable of dict
        Param groups whose `params` is a list of `(name, param)` pairs.
        Other keys (e.g. `lr`) are copied to the resulting groups. A
        group missing `lr` will use the optimizer's default one
    :param exclude: sequence of str
        Params whose name contains one of these substrings go to AdamW
    :param adamw_lr_scale: float
        Factor applied to the `lr` of the AdamW groups, relative to the
        Muon ones. Only useful with `adjust_lr_fn='original'`, where
        Muon needs a much larger LR than AdamW. Requires groups to
        explicitly carry their `lr`
    :param row_splits: dict, optional
        Maps param names to row block sizes, for the Muon params to
        orthogonalize block-wise (see `qkv_row_splits`)
    :return: list of dict, with empty groups removed
    """
    row_splits = row_splits or {}
    out = []
    for group in named_groups:
        base = {k: v for k, v in group.items() if k != 'params'}
        muon = [
            (n, p) for n, p in group['params'] if _use_muon(n, p, exclude)]
        adamw = [
            p for n, p in group['params'] if not _use_muon(n, p, exclude)]
        if muon:
            muon_group = {
                **base, 'params': [p for _, p in muon], 'use_muon': True}
            if any(n in row_splits for n, _ in muon):
                muon_group['row_splits'] = [
                    tuple(row_splits[n]) if n in row_splits else None
                    for n, _ in muon]
            out.append(muon_group)
        if adamw:
            adamw_group = {**base, 'params': adamw, 'use_muon': False}
            if adamw_lr_scale != 1:
                adamw_group['lr'] = base['lr'] * adamw_lr_scale
            out.append(adamw_group)
    return out
