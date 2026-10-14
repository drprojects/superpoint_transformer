import itertools

import pytest
import torch

from src.transforms.graph import RadiusHorizontalGraph
from src.utils.neighbors import cluster_radius_nn_graph

# `knn_1` (FRNN) only runs on GPU
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU")

GAP = 30.
SPACING = 5.


def _toy_clusters(num_clusters=6, num_points=40, size=2.):
    """Small blobs of points (`size` m wide) whose centers are spaced by
    `SPACING` m along x, and the brute-force set of pairs of clusters
    having 2 points closer than `GAP`.
    """
    g = torch.Generator().manual_seed(0)
    blobs = [
        torch.rand(num_points, 3, generator=g) * size
        + torch.tensor([i * SPACING, 0., 0.])
        for i in range(num_clusters)]
    x = torch.cat(blobs).cuda()
    idx = torch.arange(num_clusters).repeat_interleave(num_points).cuda()
    expected = {
        (i, j) for i, j in itertools.combinations(range(num_clusters), 2)
        if torch.cdist(blobs[i], blobs[j]).min() < GAP}
    return x, idx, expected


def _edges(edge_index):
    return {tuple(sorted(e)) for e in edge_index.t().tolist()}


@requires_gpu
def test_cluster_radius_nn_graph_gap_is_a_distance():
    """`knn_1` returns squared distances. They used to be compared to a
    distance in the coarse edge trimming, which discarded every pair of
    clusters with centers further apart than ~sqrt(r_max_edge) m,
    whatever `gap`. The clusters here are 5 m apart, with `gap=30` m: all
    the pairs with 2 points closer than 30 m must be neighbors, not
    only the direct ones.
    """
    x, idx, expected = _toy_clusters()
    assert len(expected) > 5  # more than the 5 direct neighbors

    edge_index, _ = cluster_radius_nn_graph(
        x, idx, k_max=5, gap=GAP, trim=True)

    assert _edges(edge_index) == expected
