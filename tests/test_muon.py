import copy
import functools
import math

import pytest
import torch

from src.optim.muon import (
    Muon,
    _adjusted_lr,
    is_muon_optimizer,
    qkv_row_splits,
    split_muon_param_groups,
    zeropower_via_newtonschulz5)


@pytest.mark.parametrize('shape', [(64, 64), (192, 64), (32, 128)])
def test_newton_schulz_orthogonalizes(shape):
    torch.manual_seed(0)
    # Well-conditioned input (singular values in [0.2, 2]). A random
    # square Gaussian matrix has singular values arbitrarily close to
    # 0, which 5 NS iterations cannot lift (expected Muon behavior).
    m, n = shape
    k = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, k))
    V, _ = torch.linalg.qr(torch.randn(n, k))
    G = U @ torch.diag(torch.linspace(0.2, 2, k)) @ V.T

    X = zeropower_via_newtonschulz5(G, steps=5).float()
    assert X.shape == G.shape
    # Singular values pushed to ~1 (the quintic coefficients only
    # target roughly [0.7, 1.2])
    s = torch.linalg.svdvals(X)
    assert s.min() > 0.5 and s.max() < 1.3

    # Same directions as the exact polar factor U V^T
    P = U @ V.T
    cos = (X * P).sum() / (X.norm() * P.norm())
    assert cos > 0.97


def test_adjust_lr_fn():
    assert _adjusted_lr(1., (64, 192), 'match_rms_adamw') \
        == pytest.approx(0.2 * math.sqrt(192))
    assert _adjusted_lr(1., (192, 64), 'match_rms_adamw') \
        == pytest.approx(0.2 * math.sqrt(192))
    assert _adjusted_lr(1., (192, 64), 'original') \
        == pytest.approx(math.sqrt(3))
    assert _adjusted_lr(1., (64, 192), 'original') == pytest.approx(1.)
    with pytest.raises(ValueError):
        _adjusted_lr(1., (2, 2), 'nope')


def test_muon_rejects_1d_params():
    p = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError):
        Muon([{'params': [p], 'use_muon': True}])


def test_muon_update_matches_reference():
    """One Muon step, vs. a hand-written Nesterov+NS+decay update."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(16, 32))
    w0 = w.detach().clone()
    g = torch.randn(16, 32)
    w.grad = g.clone()
    lr, wd, mom = 0.01, 0.1, 0.95
    opt = Muon(
        [{'params': [w], 'use_muon': True}], lr=lr, weight_decay=wd,
        momentum=mom)
    opt.step()

    buf = (1 - mom) * g
    upd = zeropower_via_newtonschulz5(g.lerp(buf, mom)).float()
    expected = w0 * (1 - lr * wd) - lr * 0.2 * math.sqrt(32) * upd
    assert torch.allclose(w.detach(), expected, atol=1e-5)


def test_adamw_group_matches_torch_adamw():
    torch.manual_seed(0)
    w_a = torch.nn.Parameter(torch.randn(8, 4))
    w_b = torch.nn.Parameter(w_a.detach().clone())
    opt_a = Muon([w_a], lr=1e-2, weight_decay=1e-2)
    opt_b = torch.optim.AdamW([w_b], lr=1e-2, weight_decay=1e-2)
    for _ in range(5):
        g = torch.randn(8, 4)
        w_a.grad, w_b.grad = g.clone(), g.clone()
        opt_a.step()
        opt_b.step()
    assert torch.allclose(w_a, w_b, atol=1e-6)


@pytest.mark.parametrize('adjust_lr_fn', ['match_rms_adamw', 'original'])
def test_muon_adamw_converges_on_regression(adjust_lr_fn):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    x = torch.randn(256, 16)
    y = x[:, :1] * 2 - x[:, 1:2]
    named = [{'params': list(model.named_parameters()),
              'lr': 5e-3 if adjust_lr_fn == 'match_rms_adamw' else 2e-2}]
    groups = split_muon_param_groups(
        named, adamw_lr_scale=1 if adjust_lr_fn == 'match_rms_adamw'
        else 0.05)
    opt = Muon(groups, adjust_lr_fn=adjust_lr_fn)
    loss0 = torch.nn.functional.mse_loss(model(x), y).item()
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.2 * loss0


def test_state_dict_roundtrip_resumes_identically():
    torch.manual_seed(0)

    def make():
        torch.manual_seed(1)
        m = torch.nn.Linear(8, 8)
        groups = split_muon_param_groups(
            [{'params': list(m.named_parameters()), 'lr': 1e-2}])
        return m, Muon(groups)

    def run(m, opt, n, seed):
        gen = torch.Generator().manual_seed(seed)
        for _ in range(n):
            for p in m.parameters():
                p.grad = torch.randn(p.shape, generator=gen)
            opt.step()

    m1, o1 = make()
    run(m1, o1, 3, seed=0)
    m2, o2 = make()
    m2.load_state_dict(m1.state_dict())
    # A checkpoint is a copy: `state_dict()` returns references to the
    # in-place updated state tensors
    o2.load_state_dict(copy.deepcopy(o1.state_dict()))
    run(m1, o1, 3, seed=1)
    run(m2, o2, 3, seed=1)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_split_muon_param_groups_routing():
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.norm = torch.nn.LayerNorm(4)
            self.head = torch.nn.Linear(4, 2)
            self.conv_kernel = torch.nn.Parameter(torch.zeros(27, 4, 4))

    net = Net()
    groups = split_muon_param_groups(
        [{'params': list(net.named_parameters()), 'lr': 1e-2}],
        adamw_lr_scale=0.1)
    muon = [g for g in groups if g['use_muon']]
    adamw = [g for g in groups if not g['use_muon']]
    assert len(muon) == 1 and len(adamw) == 1
    assert [id(p) for p in muon[0]['params']] == [id(net.fc.weight)]
    assert muon[0]['lr'] == 1e-2 and adamw[0]['lr'] == pytest.approx(1e-3)
    # Every param is in exactly one group
    ids = [id(p) for g in groups for p in g['params']]
    assert sorted(ids) == sorted(id(p) for p in net.parameters())


def test_is_muon_optimizer():
    assert is_muon_optimizer(Muon)
    assert is_muon_optimizer(functools.partial(Muon, lr=1e-3))
    assert not is_muon_optimizer(functools.partial(torch.optim.AdamW, lr=1))


def _muon_step(w, g, row_splits=None, **kwargs):
    group = {'params': [w], 'use_muon': True}
    if row_splits is not None:
        group['row_splits'] = [row_splits]
    w.grad = g.clone()
    opt = Muon([group], **kwargs)
    opt.step()
    return w.detach().clone()


@pytest.mark.parametrize('sizes', [(16, 16, 16), (8, 8, 32)])
def test_row_splits_orthogonalizes_blocks_independently(sizes):
    """Each block of a split param is updated exactly as if it were a
    standalone param, which a joint orthogonalization does not do."""
    torch.manual_seed(0)
    rows, cols, lr = sum(sizes), 32, 0.01
    w0 = torch.randn(rows, cols)
    g = torch.randn(rows, cols)

    kwargs = dict(lr=lr, weight_decay=0.)
    fused = _muon_step(torch.nn.Parameter(w0.clone()), g, **kwargs)
    split = _muon_step(
        torch.nn.Parameter(w0.clone()), g, row_splits=sizes, **kwargs)

    start = 0
    for n in sizes:
        alone = _muon_step(
            torch.nn.Parameter(w0[start:start + n].clone()),
            g[start:start + n], **kwargs)
        assert torch.allclose(split[start:start + n], alone, atol=1e-6)
        start += n
    assert not torch.allclose(split, fused, atol=1e-4)


def test_row_splits_validation():
    w = torch.nn.Parameter(torch.zeros(10, 4))
    with pytest.raises(ValueError):
        Muon([{'params': [w], 'use_muon': True,
                    'row_splits': [(4, 4)]}])
    with pytest.raises(ValueError):
        Muon([{'params': [w], 'use_muon': True,
                    'row_splits': []}])


def test_qkv_row_splits_on_attention_block():
    from src.nn.attention import SelfAttentionBlock
    # Unequal blocks: qk_dim * num_heads = 8 != dim = 32
    block = torch.nn.Sequential(
        SelfAttentionBlock(32, num_heads=4, qk_dim=2))
    splits = qkv_row_splits(block)
    assert splits == {'0.qkv.weight': (8, 8, 32)}
    assert sum(splits['0.qkv.weight']) == block[0].qkv.weight.shape[0]


def test_split_muon_param_groups_row_splits():
    net = torch.nn.ModuleDict({
        'qkv': torch.nn.Linear(4, 12), 'fc': torch.nn.Linear(4, 4)})
    groups = split_muon_param_groups(
        [{'params': list(net.named_parameters()), 'lr': 1e-2}],
        row_splits={'qkv.weight': (4, 4, 4)})
    muon = [g for g in groups if g['use_muon']][0]
    names = {id(p): n for n, p in net.named_parameters()}
    assert [(names[id(p)], s) for p, s in
            zip(muon['params'], muon['row_splits'])] \
        == [('qkv.weight', (4, 4, 4)), ('fc.weight', None)]
    # The optimizer accepts the groups
    Muon(groups)
    # No `row_splits` key at all when there is nothing to split
    assert 'row_splits' not in split_muon_param_groups(
        [{'params': list(net.named_parameters()), 'lr': 1e-2}])[0]
