import time
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from torch_scatter import scatter_min
from src.utils.partition import xy_partition
from src.utils.point import is_xyz_tensor
from src.utils.neighbors import knn_2



__all__ = [
    'filter_by_z_distance_of_global_min', 'filter_by_local_z_min',
    'filter_by_verticality', 'single_plane_model',
    'neighbor_interpolation_model', 'mlp_model']


def filter_by_z_distance_of_global_min(pos, threshold):
    """Search for points within `threshold` Z-distance of the lowest
    point in the input cloud `xyz`.

    This can be used to filter out points far from the ground, with some
    limitations:
    - if the point cloud contains below-ground points
    - if the ground is not even and involves stairs, slopes, ...

    :param pos: Tensor
        Input 3D point cloud
    :param threshold: float
        Z-distance threshold. Points within a Z-offset of `threshold`
        or lower of the lowest point (i.e. smallest Z) will be selected
    :return:
    """
    assert is_xyz_tensor(pos)
    return pos[:, 2] - pos[:, 2].min() < threshold


def filter_by_local_z_min(pos, grid):
    """Search for the lowest point in each cell of a horizontal XY grid.

    This can be used to filter out points far from the ground, with some
    limitations:
    - if the point cloud contains below-ground points
    - if the ground has slopes, the size of the `grid` may produce
    downstream staircasing effects if the local Z-min points are used as
    Z reference for local ground altitude

    :param pos: Tensor
        Input 3D point cloud
    :param grid: float
        Size of the grid "XY-voxel"
    :return:
    """
    assert is_xyz_tensor(pos)

    # Bin points into an XY grid
    super_index = xy_partition(pos, grid, consecutive=True)

    # Search for the lowest point in each grid cell
    z_min, z_argmin = scatter_min(pos[:, 2], super_index, dim=0)
    is_local_z_min = torch.full((pos.shape[0],), False, device=pos.device)
    is_local_z_min[z_argmin] = True

    return is_local_z_min


def filter_by_verticality(verticality, threshold):
    """Search for the points with low verticality.

    For verticality computation, see the `PointFeatures`.

    This can be used to filter out non-ground points, with some
    limitations:
    - if the point cloud is very noisy, or if the verticality
      was computed on too-small, or too-large neighborhoods, the
      verticality may not be sufficiently discriminative
    - if the ground has slopes, the steepest areas may be filtered out
    - if other non-ground horizontal surfaces are present in the point
      cloud, these will also be preserved (e.g. table, ceiling,
      horizontal building roof, ...)

    :param verticality: Tensor
        1D tensor holding verticality values as computed by
        `PointFeatures`
    :param threshold: float
        Verticality threshod below which points are considered
        "horizontal" enough
    :return:
    """
    return verticality.squeeze() < threshold


def single_plane_model(pos, random_state=0, residual_threshold=1e-3):
    """Model the ground as a single plane using RANSAC.

    Returns a callable taking an XYZ tensor as input and returning the
    pointwise elevation.

    :param pos: Tensor
        Input 3D point cloud
    :param random_state: int
        Seed for RANSAC
    :param residual_threshold: float
        Residual threshold for RANSAC
    :return:
    """
    assert is_xyz_tensor(pos)

    xy = pos[:, :2].cpu().numpy()
    z = pos[:, 2].cpu().numpy()

    # Search the ground plane using RANSAC
    ransac = RANSACRegressor(
        random_state=random_state, residual_threshold=residual_threshold).fit(
        xy, z)

    def predict_elevation(pos_query):
        assert is_xyz_tensor(pos_query)
        device = pos_query.device
        xy = pos_query[:, :2]
        z = pos_query[:, 2]
        return z - torch.from_numpy(ransac.predict(xy.cpu().numpy())).to(device)

    return predict_elevation


def neighbor_interpolation_model(pos, k=3, r_max=1):
    """Model the ground based on a trimmed point cloud carrying ground
    points only. At inference, a point is associated with its nearest
    neighbors in L2 XY distance in the reference ground cloud. The
    ground surface is estimated as a linear interpolation of the
    neighboring reference points. The elevation is then computed as the
    corresponding gap in Z-coordinates.

    Returns a callable taking an XYZ tensor as input and returning the
    pointwise elevation.

    :param pos: Tensor
        Input 3D point cloud
    :param k: int
        Number of neighbors to consider for interpolation
    :param r_max: float
        Maximum radius for the neighbor search
    :return:
    """
    assert is_xyz_tensor(pos)

    def predict_elevation(pos_query):
        # Neighbor search in XY space
        xy0 = F.pad(input=pos[:, :2], pad=(0, 1), mode='constant', value=0)
        xy0_query = F.pad(
            input=pos_query[:, :2], pad=(0, 1), mode='constant', value=0)
        neighbors, distances = knn_2(xy0, xy0_query, k, r_max=r_max)

        # In case some points received 0 neighbors, we search again for
        # those, with a radius so large that no point should be left
        # without a neighbor
        has_no_neighbor = (neighbors == -1).all(dim=1)
        if has_no_neighbor.any():
            high = xy0.max(dim=0).values
            low = xy0.min(dim=0).values
            high_query = xy0_query.max(dim=0).values
            low_query = xy0_query.min(dim=0).values
            r_max_ = max((high_query - low).norm(), (high - low_query).norm())

            neighbors_, distances_ = knn_2(
                xy0, xy0_query[has_no_neighbor], k, r_max=r_max_)

            neighbors[has_no_neighbor] = neighbors_
            distances[has_no_neighbor] = distances_

        # If only 1 neighbor is needed, no need for interpolation
        if k == 1:
            return pos_query[:, 2] - pos[neighbors][:, 2]

        # Note there might still be some missing neighbors here and
        # there, but no completely empty neighborhood. We treat these
        # missing neighbors by attributing a 0-weight
        weights = 1 / (distances + 1e-3)
        weights[distances == -1] = 0
        weights = weights / weights.sum(dim=1).view(-1, 1)

        # Estimate the ground height as the weighted combination of the
        # neighbors' height
        z_query = (pos[:, 2][neighbors] * weights).sum(dim=1)

        return pos_query[:, 2] - z_query

    return predict_elevation


def mlp_model(
        pos,
        layers=[32, 16, 8],
        batch_ratio=1,
        lr=0.01,
        lr_decay=1,
        weight_decay=0.01,
        criterion='l2',
        steps=1000,
        check_every_n_steps=50,
        device='cuda',
        verbose=False):
    """Fit an MLP to a point cloud. Assuming the point cloud mostly
    contains ground points, this function will train an MLP to model the
    ground surface as a piecewise-planar function.

    :param pos: Tensor
        Input 3D point cloud
    :param layers: int or List[int]
        Hidden layers for the MLP. Too many weights may let the model
        overfit to non-ground patterns. Not enough weights will underfit
        the ground and miss some patterns. Having more neurons in the
        first layer allows faster convergence
    :param batch_ratio: float
        Ratio of points to sample from the cloud at each training
        iteration. Allows adding some stochasticity to the training.
        In practice, `batch_ratio=1` gives better results if the entire
        cloud fits in memory
    :param lr: float
        Initial learning rate
    :param lr_decay: float
        Multiplicative factor applied to the learning rate after each
        iteration
    :param weight_decay: float
        Weight decay for regularization
    :param criterion: str
        Loss, either 'l1' or 'l2'
    :param steps: int
        Number of training steps. This largely affects overall compute
        time
    :param check_every_n_steps: int
        If `verbose=True` the loss will be logged every n iteration for
        final visualization
    :param device: str or torch.device
        Device on which to do the training and inference
    :param verbose: bool
        If True, a plot of the training loss and some stats will be
        printed at the end of the training
    :return:
    """
    # Local imports to avoid import loop errors
    from src.nn import MLP
    from src.nn.norm import BatchNorm

    assert is_xyz_tensor(pos)

    torch.cuda.synchronize()
    start = time.time()

    # Normalize the XYZ coordinates to live in a manageable range
    pos = pos.to(device)
    num_points = pos.shape[0]
    means = pos.mean(dim=0)
    stds = pos.std(dim=0)
    pos = (pos - means) / stds

    # Prepare the training
    batch_size = min(num_points, round(num_points * batch_ratio))
    layers = [layers] if isinstance(layers, int) else layers
    model = MLP(
        [2] + layers + [1],
        activation=nn.ReLU(),
        last_activation=False,
        norm=BatchNorm,
        last_norm=False,
        drop=None).to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay)
    weights = torch.ones(num_points, device=device)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

    # For drawing the loss plot for debugging purposes
    if verbose:
        l = []
        t = []

    # Training loop
    for step in range(steps):
        # Optionally, randomly drop some data points for augmentation
        if 0 < batch_ratio < 1:
            idx = torch.multinomial(weights, batch_size, replacement=False)
            pos_ = pos[idx]
        else:
            pos_ = pos

        # Forward pass
        xy = pos_[:, :2]
        z = pos_[:, 2]
        z_hat = model(xy)

        # Loss computation
        if criterion == 'l2':
            loss = ((z - z_hat.squeeze()) ** 2).mean()
        elif criterion == 'l1':
            loss = (z - z_hat.squeeze()).abs().mean()
        else:
            raise NotImplementedError("")

        # Gradient computation and backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if not verbose or step % check_every_n_steps != check_every_n_steps - 1:
            continue

        # Keep track of the loss
        # print(f"Step {step + 1} loss: {loss:0.3f}")
        t.append(step)
        l.append(loss.detach().cpu().item())

    if verbose:
        torch.cuda.synchronize()
        print(f"Training time: {time.time() - start:0.1f} sec")
        print(f"Loss: {l[-1]:0.3f}")
        plt.plot(t, l)
        plt.show()

    # Training is finished, set the model to inference mode
    model = model.eval()

    def predict_elevation(pos):
        input_device = pos.device
        pos = pos.to(device)

        xy = (pos[:, :2] - means[:2]) / stds[:2]
        z = model(xy).squeeze().detach()
        z = z * stds[2] + means[2]

        elevation = pos[:, 2] - z

        return elevation.to(input_device)

    return predict_elevation
