import torch

from src.nn.stage import DownNFuseStage, UpNFuseStage


def test_down_stage_injects_the_diameter_it_receives():
    """`DownNFuseStage.forward` used to accept `diameter` without
    forwarding it to `Stage.forward`, so `use_diameter=True` only ever
    injected a column of zeros.
    """
    stage = DownNFuseStage(
        4, num_blocks=0, use_pos=False, use_diameter=True, fusion='cat')
    x_parent = torch.rand(2, 2)
    x_child = torch.rand(5, 2)
    pool_index = torch.tensor([0, 0, 1, 1, 1])
    diameter = torch.tensor([[3.], [7.]])

    x, _ = stage(
        x_parent, x_child, torch.zeros(2, dtype=torch.long), pool_index,
        diameter=diameter, num_super=2)

    torch.testing.assert_close(x[:, :1], diameter)


def test_up_stage_injects_the_diameter_it_receives():
    stage = UpNFuseStage(
        4, num_blocks=0, use_pos=False, use_diameter=True, fusion='cat')
    x_child = torch.rand(5, 2)
    x_parent = torch.rand(2, 2)
    unpool_index = torch.tensor([0, 0, 1, 1, 1])
    diameter = torch.rand(5, 1)

    x, _ = stage(
        x_child, x_parent, torch.zeros(5, dtype=torch.long), unpool_index,
        diameter=diameter)

    torch.testing.assert_close(x[:, :1], diameter)
