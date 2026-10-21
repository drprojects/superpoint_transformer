import torch
from torch import nn
from torch.nn import functional as F

from src.data import NAG, Data
from src.models.semantic import SemanticSegmentationModule


class _TwoStageNet(nn.Module):
    """Minimal stand-in for `SPT` with a stage-wise output."""
    out_dim = [4, 4]
    output_stage_wise = True
    store_features = False

    def __init__(self):
        super().__init__()
        self.first_stage = nn.Identity()
        self.x = nn.ParameterList([
            nn.Parameter(torch.rand(2, 4)), nn.Parameter(torch.rand(2, 4))])

    def forward(self, nag):
        return list(self.x)


def test_multi_stage_wce_uses_the_dominant_label_of_each_node():
    """With a multi-stage loss, `loss_type='wce'` wrote the per-node
    totals into every row of the dominant-label histogram, as fixed for
    the single-stage 'wce' and for 'wce_kl'. Each node must be
    supervised by its own dominant label only, weighted by its size.
    """
    torch.manual_seed(0)
    model = SemanticSegmentationModule(
        net=_TwoStageNet(), criterion=nn.CrossEntropyLoss(), optimizer=None,
        scheduler=None, num_classes=3, loss_type='wce',
        multi_stage_loss_lambdas=[1, 1])
    y = torch.tensor([[5, 1, 0, 0], [0, 2, 7, 0]])
    nag = NAG([
        Data(pos=torch.rand(4, 3), super_index=torch.tensor([0, 0, 1, 1])),
        Data(pos=torch.rand(2, 3), y=y, super_index=torch.tensor([0, 1])),
        Data(pos=torch.rand(2, 3), y=y)])

    loss, output = model.model_step(nag)

    expected = 0
    for logits in output.logits:
        ce = F.cross_entropy(logits, torch.tensor([0, 2]), reduction='none')
        expected = expected + (6 * ce[0] + 9 * ce[1]) / 15
    torch.testing.assert_close(loss, expected)
