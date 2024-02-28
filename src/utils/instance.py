import sys
import hydra
import torch
import numpy as np
import pandas as pd
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from torch.nn.functional import one_hot
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from src.utils.hydra import init_config
from src.utils.neighbors import knn_2
from src.utils.graph import to_trimmed
from src.utils.cpu import available_cpu_count
from src.utils.scatter import scatter_mean_weighted


src_folder = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(src_folder)
sys.path.append(osp.join(src_folder, "dependencies/grid_graph/python/bin"))
sys.path.append(osp.join(src_folder, "dependencies/parallel_cut_pursuit/python/wrappers"))


from grid_graph import edge_list_to_forward_star
from cp_d0_dist import cp_d0_dist


__all__ = [
    'generate_random_bbox_data', 'generate_random_segment_data',
    'instance_cut_pursuit', 'oracle_superpoint_clustering', 'get_stuff_mask',
    'compute_panoptic_metrics', 'compute_metrics_s3dis_6fold',
    'grid_search_panoptic_partition']


_MAX_NUM_EDGES = 4294967295


def generate_random_bbox_data(
        num_img=1,
        num_classes=1,
        height=128,
        width=128,
        h_split=1,
        w_split=2,
        det_gt_ratio=1):
    # Create some images with a ground truth partition
    instance_images = -torch.ones(num_img, height, width, dtype=torch.long)
    label_images = -torch.ones(num_img, height, width, dtype=torch.long)

    h_gt = height // h_split
    w_gt = width // w_split

    gt_boxes = torch.zeros(num_img * h_split * w_split, 4)
    gt_labels = torch.randint(0, num_classes, (num_img * h_split * w_split,))
    iterator = product(range(num_img), range(h_split), range(w_split))
    for idx, (i_img, i, j) in enumerate(iterator):
        h1 = i * h_gt
        h2 = (i + 1) * h_gt
        w1 = j * w_gt
        w2 = (j + 1) * w_gt
        instance_images[i_img, h1:h2, w1:w2] = idx
        label_images[i_img, h1:h2, w1:w2] = gt_labels[idx]
        gt_boxes[idx] = torch.tensor([h1, w1, h2, w2])

    # Create some random detection boxes
    num_gt = (instance_images.max() + 1).item()
    num_det = int(num_gt * det_gt_ratio)

    i_center_det = torch.randint(0, height, (num_det,))
    j_center_det = torch.randint(0, width, (num_det,))
    h_det = torch.randint(int(h_gt * 0.7), int(h_gt * 1.3), (num_det,))
    w_det = torch.randint(int(w_gt * 0.7), int(w_gt * 1.3), (num_det,))

    det_boxes = torch.vstack([
        (i_center_det - h_det / 2).clamp(min=0),
        (j_center_det - w_det / 2).clamp(min=0),
        (i_center_det + h_det / 2).clamp(max=height),
        (j_center_det + w_det / 2).clamp(max=width)]).T.round()
    det_img_idx = torch.randint(0, num_img, (num_det,))
    det_labels = torch.randint(0, num_classes, (num_det,))
    det_scores = torch.rand(num_det)

    # Display the images stacked along their height (first dim) and draw
    # the box for each detection
    fig, ax = plt.subplots()
    ax.imshow(instance_images.view(-1, width), cmap='jet')
    for idx_det in range(num_det):
        i = det_boxes[idx_det, 0] + det_img_idx[idx_det] * height
        j = det_boxes[idx_det, 1]
        h = det_boxes[idx_det, 2] - det_boxes[idx_det, 0]
        w = det_boxes[idx_det, 3] - det_boxes[idx_det, 1]
        rect = patches.Rectangle(
            (j, i),
            w,
            h,
            linewidth=3,
            edgecolor=cm.nipy_spectral(idx_det / num_det),
            facecolor='none')
        ax.add_patch(rect)
    plt.show()

    # Display the images stacked along their height (first dim) and draw the
    # box for each detection
    fig, ax = plt.subplots()
    ax.imshow(label_images.view(-1, width).float() / num_classes, cmap='jet')
    for idx_det in range(num_det):
        i = det_boxes[idx_det, 0] + det_img_idx[idx_det] * height
        j = det_boxes[idx_det, 1]
        h = det_boxes[idx_det, 2] - det_boxes[idx_det, 0]
        w = det_boxes[idx_det, 3] - det_boxes[idx_det, 1]
        c = cm.nipy_spectral(det_labels[idx_det].float().item() / num_classes)
        rect = patches.Rectangle(
            (j, i),
            w,
            h,
            linewidth=3,
            edgecolor=c,
            facecolor='none')
        ax.add_patch(rect)
    plt.show()

    # Compute the metrics using torchmetrics
    iterator = zip(gt_boxes.view(num_img, -1, 4), gt_labels.view(num_img, -1))
    targets = [
        dict(boxes=boxes, labels=labels)
        for boxes, labels in iterator]

    preds = [
        dict(
            boxes=det_boxes[det_img_idx == i_img],
            labels=det_labels[det_img_idx == i_img],
            scores=det_scores[det_img_idx == i_img])
        for i_img in range(num_img)]

    # For each predicted pixel, we compute the gt object idx, and the gt
    # label, to build an InstanceData.
    # NB: we cannot build this by creating a single pred_idx image,
    # because predictions may overlap in this toy setup, unlike our 3D
    # superpoint partition paradigm...
    pred_idx = []
    gt_idx = []
    gt_y = []
    for idx_det in range(num_det):
        i_img = det_img_idx[idx_det]
        x1, y1, x2, y2 = det_boxes[idx_det].long()
        num_points = (x2 - x1) * (y2 - y1)
        pred_idx.append(torch.full((num_points,), idx_det))
        gt_idx.append(instance_images[i_img, x1:x2, y1:y2].flatten())
        gt_y.append(label_images[i_img, x1:x2, y1:y2].flatten())
    pred_idx = torch.cat(pred_idx)
    gt_idx = torch.cat(gt_idx)
    gt_y = torch.cat(gt_y)
    count = torch.ones_like(pred_idx)

    from src.data.instance import InstanceData
    instance_data = InstanceData(pred_idx, gt_idx, count, gt_y, dense=True)

    return targets, preds, gt_idx, gt_y, count, instance_data


def generate_single_random_segment_image(
        num_gt=10,
        num_pred=12,
        num_classes=3,
        height=32,
        width=64,
        shift=5,
        random_pred_label=False,
        show=True,
        iterations=20):
    """Generate an image with random ground truth and predicted instance
    and semantic segmentation data. To make the images realisitc, and to
    ensure that the instances form a PARTITION of the image, we rely on
    voronoi cells. Besides, to encourage a realistic overalp between the
    predicted and target instances, the predcition cell centers are
    sampled near the target samples.
    """
    # Generate random pixel positions for the ground truth and the
    # prediction centers. To produce predictions with "controllable"
    # overlap with the targets, we use the gt's centers as seeds for the
    # prediction centers and randomly sample shift them
    x = torch.randint(0, height, (num_gt,))
    y = torch.randint(0, width, (num_gt,))
    gt_xy = torch.vstack((x, y)).T
    if num_pred <= num_gt:
        idx_ref_gt = torch.from_numpy(
            np.random.choice(num_gt, num_pred, replace=False))
    else:
        idx_ref_gt = torch.from_numpy(
            np.random.choice(num_gt, num_pred % num_gt, replace=False))
        idx_ref_gt = torch.cat((
            torch.arange(num_gt).repeat(num_pred // num_gt), idx_ref_gt))
    xy_shift = torch.randint(0, 2 * shift, (num_pred, 2)) - shift
    pred_xy = gt_xy[idx_ref_gt] + xy_shift
    clamp_min = torch.tensor([0, 0])
    clamp_max = torch.tensor([height, width])
    pred_xy = pred_xy.clamp(min=clamp_min, max=clamp_max)

    # The above prediction center generation process may produce
    # duplicates, which can in turn generate downstream errors. To avoid
    # this, we greedily search for duplicates and shift them
    already_used_xy_ids = []
    for i_pred, xy in enumerate(pred_xy):
        xy_id = xy[0] * width + xy[1]
        count = 0

        while xy_id in already_used_xy_ids and count < iterations:
            xy_shift = torch.randint(0, 2 * shift, (2,)) - shift
            xy = gt_xy[idx_ref_gt[i_pred]] + xy_shift
            xy = xy.clamp(min=clamp_min, max=clamp_max)
            xy_id = xy[0] * width + xy[1]
            count += 1

        if count == iterations:
            raise ValueError(
                f"Reached max iterations={iterations} while resampling "
                "duplicate prediction centers")

        already_used_xy_ids.append(xy_id)
        pred_xy[i_pred] = xy

    # Generate labels and scores
    gt_labels = torch.randint(0, num_classes, (num_gt,))
    if random_pred_label:
        pred_labels = torch.randint(0, num_classes, (num_pred,))
    else:
        pred_labels = gt_labels[idx_ref_gt]
    pred_scores = torch.rand(num_pred)

    # Generate a 3D point cloud representing the pixel coordinates of the
    # image. This will be used to compute the 1-NNs and, from there, a
    # partition into voronoi cells
    x, y = torch.meshgrid(
        torch.arange(height), torch.arange(width), indexing='ij')
    x = x.flatten()
    y = y.flatten()
    z = torch.zeros_like(x)
    xyz = torch.vstack((x, y, z)).T

    # Compute a gt segmentation image from the 1-NN of each pixel, wrt the
    # gt segment centers
    gt_xyz = torch.cat((gt_xy, torch.zeros_like(gt_xy[:, [0]])), dim=1).float()
    gt_nn = knn_2(gt_xyz, xyz.float(), 1, r_max=max(width, height))[0]
    gt_seg_image = gt_nn.view(height, width)
    gt_label_image = gt_labels[gt_seg_image]

    # Compute a pred segmentation image from the 1-NN of each pixel, wrt the
    # pred segment centers
    pred_xyz = torch.cat((pred_xy, torch.zeros_like(pred_xy[:, [0]])), dim=1).float()
    pred_nn = knn_2(pred_xyz, xyz.float(), 1, r_max=max(width, height))[0]
    pred_seg_image = pred_nn.view(height, width)
    pred_label_image = pred_labels[pred_seg_image]

    # Display the segment images
    if show:
        plt.subplot(2, 2, 1)
        plt.title('Ground truth instances')
        plt.imshow(gt_seg_image)
        plt.subplot(2, 2, 2)
        plt.title('Predicted instances')
        plt.imshow(pred_seg_image)
        plt.subplot(2, 2, 3)
        plt.title('Ground truth labels')
        plt.imshow(gt_label_image)
        plt.subplot(2, 2, 4)
        plt.title('Predicted labels')
        plt.imshow(pred_label_image)
        plt.show()

    # Organize the data into torchmetric-friendly format
    tm_targets = dict(
        masks=torch.stack([gt_seg_image == i_gt for i_gt in range(num_gt)]),
        labels=gt_labels)

    tm_preds = dict(
        masks=torch.stack([pred_seg_image == i_pred for i_pred in range(num_pred)]),
        labels=pred_labels,
        scores=pred_scores)

    tm_data = (tm_preds, tm_targets)

    # Organize the data into our custom format
    pred_idx = pred_seg_image.flatten()
    gt_idx = gt_seg_image.flatten()
    gt_y = gt_label_image.flatten()
    count = torch.ones_like(pred_idx)

    from src.data.instance import InstanceData
    instance_data = InstanceData(pred_idx, gt_idx, count, gt_y, dense=True)
    spt_data = (pred_scores, pred_labels, instance_data)

    return tm_data, spt_data


def generate_random_segment_data(
        num_img=2,
        num_gt_per_img=10,
        num_pred_per_img=14,
        num_classes=2,
        height=32,
        width=64,
        shift=5,
        random_pred_label=False,
        verbose=True):
    """Generate multiple images with random ground truth and predicted
    instance and semantic segmentation data. To make the images
    realistic, and to ensure that the instances form a PARTITION of the
    image, we rely on voronoi cells. Besides, to encourage a realistic
    overlap between the predicted and target instances, the prediction
    cell centers are sampled near the target samples.
    """
    tm_data = []
    spt_data = []

    for i_img in range(num_img):
        if verbose:
            print(f"\nImage {i_img + 1}/{num_img}")
        tm_data_, spt_data_ = generate_single_random_segment_image(
            num_gt=num_gt_per_img,
            num_pred=num_pred_per_img,
            num_classes=num_classes,
            height=height,
            width=width,
            shift=shift,
            random_pred_label=random_pred_label,
            show=verbose)
        tm_data.append(tm_data_)
        spt_data.append(spt_data_)

    return tm_data, spt_data


def _instance_cut_pursuit(
        node_x,
        node_logits,
        node_size,
        edge_index,
        edge_affinity_logits,
        loss_type='l2_kl',
        regularization=1e-2,
        x_weight=1,
        p_weight=1,
        cutoff=1,
        parallel=True,
        iterations=10,
        trim=False,
        discrepancy_epsilon=1e-4,
        temperature=1,
        dampening=0,
        verbose=False):
    """Partition an instance graph using cut-pursuit.

    :param node_x: Tensor of shape [num_nodes, num_dim]
        Node features
    :param node_logits: Tensor of shape [num_nodes, num_classes]
        Predicted classification logits for each node
    :param node_size: Tensor of shape [num_nodes]
        Size of each node
    :param edge_index: Tensor of shape [2, num_edges]
        Edges of the graph, in torch-geometric's format
    :param edge_affinity_logits: Tensor of shape [num_edges]
        Predicted affinity logits (ie in R+, before sigmoid) of each
        edge
    :param loss_type: str
        Rules the loss applied on the node features. Accepts one of
        'l2' (L2 loss on node features and probabilities),
        'l2_kl' (L2 loss on node features and Kullback-Leibler
        divergence on node probabilities)
    :param regularization: float
        Regularization parameter for the partition
    :param x_weight: float
        Weight used to mitigate the impact of the node position in the
        partition. The larger, the lesser features importance before
        the probabilities
    :param p_weight: float
        Weight used to mitigate the impact of the node probabilities in
        the partition. The larger, the lesser features importance before
        the features
    :param cutoff: float
        Minimum number of points in each cluster
    :param parallel: bool
        Whether cut-pursuit should run in parallel
    :param iterations: int
        Maximum number of iterations for each partition
    :param trim: bool
        Whether the input graph should be trimmed. See `to_trimmed()`
        documentation for more details on this operation
    :param discrepancy_epsilon: float
        Mitigates the maximum discrepancy. More precisely:
        `affinity=1 ⇒ discrepancy=1/discrepancy_epsilon`
    :param temperature: float
        Temperature used in the softmax when converting node logits to
        probabilities
    :param dampening: float
        Dampening applied to the node probabilities to mitigate the
        impact of near-zero probabilities in the Kullback-Leibler
        divergence
    :param verbose: bool
    :return:
    """

    # Sanity checks
    assert node_x.dim() == 2, \
        "`node_x` must have shape `[num_nodes, num_dim]`"
    assert node_logits.dim() == 2, \
        "`node_logits` must have shape `[num_nodes, num_classes]`"
    assert node_logits.shape[0] == node_x.shape[0], \
        "`node_logits` and `node_x` must have the same number of points"
    assert node_size.dim() == 1, \
        "`node_size` must have shape `[num_nodes]`"
    assert node_size.shape[0] == node_x.shape[0], \
        "`node_size` and `node_x` must have the same number of points"
    assert edge_index.dim() == 2 and edge_index.shape[0] == 2, \
        "`edge_index` must be of shape `[2, num_edges]`"
    edge_affinity_logits = edge_affinity_logits.squeeze()
    assert edge_affinity_logits.dim() == 1, \
        "`edge_affinity_logits` must be of shape `[num_edges]`"
    assert edge_affinity_logits.shape[0] == edge_index.shape[1], \
        "`edge_affinity_logits` and `edge_index` must have the same number " \
        "of edges"
    loss_type = loss_type.lower()
    assert loss_type in ['l2', 'l2_kl'], \
        "`loss_type` must be one of ['l2', 'l2_kl']"
    assert 0 < discrepancy_epsilon, \
        "`discrepancy_epsilon` must be strictly positive"
    assert 0 < temperature, "`temperature` must be strictly positive"
    assert 0 <= dampening <= 1, "`dampening` must be in [0, 1]"

    device = node_x.device
    num_nodes = node_x.shape[0]
    x_dim = node_x.shape[1]
    p_dim = node_logits.shape[1]
    dim = x_dim + p_dim
    num_edges = edge_affinity_logits.numel()

    assert num_nodes < np.iinfo(np.uint32).max, \
        "Too many nodes for `uint32` indices"
    assert num_edges < np.iinfo(np.uint32).max, \
        "Too many edges for `uint32` indices"

    # Initialize the number of threads used for parallel cut-pursuit
    num_threads = available_cpu_count() if parallel else 1

    # Exit if the graph contains only one node
    if num_nodes < 2:
        return torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Trim the graph, if need be
    if trim:
        edge_index, edge_affinity_logits = to_trimmed(
            edge_index, edge_attr=edge_affinity_logits, reduce='mean')

    if verbose:
        print(
            f'Launching instance partition reg={regularization}, '
            f'cutoff={cutoff}')

    # User warning if the number of edges exceeds uint32 limits
    if num_edges > _MAX_NUM_EDGES and verbose:
        print(
            f"WARNING: number of edges {num_edges} exceeds the uint32 limit "
            f"{_MAX_NUM_EDGES}. Please update the cut-pursuit source code to "
            f"accept a larger data type for `index_t`.")

    # Convert affinity logits to discrepancies
    edge_affinity = edge_affinity_logits.sigmoid()
    edge_discrepancy = edge_affinity / (1 - edge_affinity + discrepancy_epsilon)

    # Convert edges to forward-star (or CSR) representation
    source_csr, target, reindex = edge_list_to_forward_star(
        num_nodes, edge_index.T.contiguous().cpu().numpy())
    source_csr = source_csr.astype('uint32')
    target = target.astype('uint32')
    edge_weights = edge_discrepancy.cpu().numpy()[reindex] * regularization \
        if edge_discrepancy is not None else regularization

    # Convert logits to class probabilities
    node_probas = torch.nn.functional.softmax(node_logits / temperature, dim=1)

    # Apply some dampening to the probability distributions. This brings
    # the distributions closer to a uniform distribution, limiting the
    # impact of near-zero probabilities in the Kullback-Leibler
    # divergence in the partition
    num_classes = node_probas.shape[1]
    node_probas = (1 - dampening) * node_probas + dampening / num_classes

    # Mean-center the node features, in case values have a very large
    # mean. This is optional, but favors maintaining values in a
    # reasonable float32 range
    node_x = node_x - node_x.mean(dim=0).view(1, -1)

    # Build the node features as the concatenation of positions and
    # class probabilities
    x = torch.cat((node_x, node_probas), dim=1)
    x = np.asfortranarray(x.cpu().numpy().T)
    node_size = node_size.float().cpu().numpy()

    # The `loss` term will decide which portion of `x` should be treated
    # with L2 loss and which should be treated with Kullback-Leibler
    # divergence
    l2_dim = dim if loss_type == 'l2' else x_dim

    # Weighting to apply on the features and probabilities
    coor_weights_dim = dim if loss_type == 'l2' else x_dim + 1
    coor_weights = np.ones(coor_weights_dim, dtype=np.float32)
    coor_weights[:x_dim] *= x_weight
    coor_weights[x_dim:] *= p_weight

    # Partition computation
    obj_index, x_c, cluster, edges, times = cp_d0_dist(
        l2_dim,
        x,
        source_csr,
        target,
        edge_weights=edge_weights,
        vert_weights=node_size,
        coor_weights=coor_weights,
        min_comp_weight=cutoff,
        cp_dif_tol=1e-2,
        K=4,
        cp_it_max=iterations,
        split_damp_ratio=0.7,
        verbose=verbose,
        max_num_threads=num_threads,
        balance_parallel_split=True,
        compute_Time=True,
        compute_List=True,
        compute_Graph=True)

    if verbose:
        delta_t = (times[1:] - times[:-1]).round(2)
        print(f'Instance partition times: {delta_t}')

    # Convert the obj_index to the input format
    obj_index = torch.from_numpy(obj_index.astype('int64')).to(device)

    return obj_index


def instance_cut_pursuit(
        batch,
        node_x,
        node_logits,
        stuff_classes,
        node_size,
        edge_index,
        edge_affinity_logits,
        loss_type='l2_kl',
        regularization=1e-2,
        x_weight=1,
        p_weight=1,
        cutoff=1,
        parallel=True,
        iterations=10,
        trim=False,
        discrepancy_epsilon=1e-4,
        temperature=1,
        dampening=0,
        verbose=False):
    """The forward step will compute the partition on the instance
    graph, based on the node features, node logits, and edge
    affinities. The partition segments will then be further merged
    so that there is at most one instance of each stuff class per
    batch item (ie per scene).

    :param batch: Tensor of shape [num_nodes]
        Batch index of each node
    :param node_x: Tensor of shape [num_nodes, num_dim]
        Predicted node embeddings
    :param node_logits: Tensor of shape [num_nodes, num_classes]
        Predicted classification logits for each node
    :param stuff_classes: List or Tensor
        List of 'stuff' class labels. These are used for merging
        stuff segments together to ensure there is at most one
        predicted instance of each 'stuff' class per batch item
    :param node_size: Tensor of shape [num_nodes]
        Size of each node
    :param edge_index: Tensor of shape [2, num_edges]
        Edges of the graph, in torch-geometric's format
    :param edge_affinity_logits: Tensor of shape [num_edges]
        Predicted affinity logits (ie in R+, before sigmoid) of each
        edge
    :param loss_type: str
        Rules the loss applied on the node features. Accepts one of
        'l2' (L2 loss on node features and probabilities),
        'l2_kl' (L2 loss on node features and Kullback-Leibler
        divergence on node probabilities)
    :param regularization: float
        Regularization parameter for the partition
    :param x_weight: float
        Weight used to mitigate the impact of the node position in the
        partition. The larger, the lesser features importance before
        the probabilities
    :param p_weight: float
        Weight used to mitigate the impact of the node probabilities in
        the partition. The larger, the lesser features importance before
        the features
    :param cutoff: float
        Minimum number of points in each cluster
    :param parallel: bool
        Whether cut-pursuit should run in parallel
    :param iterations: int
        Maximum number of iterations for each partition
    :param trim: bool
        Whether the input graph should be trimmed. See `to_trimmed()`
        documentation for more details on this operation
    :param discrepancy_epsilon: float
        Mitigates the maximum discrepancy. More precisely:
        `affinity=1 ⇒ discrepancy=1/discrepancy_epsilon`
    :param temperature: float
        Temperature used in the softmax when converting node logits to
        probabilities
    :param dampening: float
        Dampening applied to the node probabilities to mitigate the
        impact of near-zero probabilities in the Kullback-Leibler
        divergence
    :param verbose: bool

    :return: obj_index: Tensor of shape [num_nodes]
        Indicates which predicted instance each node belongs to
    """

    # Actual partition, returns a tensor indicating which predicted
    # object each node belongs to
    obj_index = _instance_cut_pursuit(
        node_x,
        node_logits,
        node_size,
        edge_index,
        edge_affinity_logits,
        loss_type=loss_type,
        regularization=regularization,
        x_weight=x_weight,
        p_weight=p_weight,
        cutoff=cutoff,
        parallel=parallel,
        iterations=iterations,
        trim=trim,
        discrepancy_epsilon=discrepancy_epsilon,
        temperature=temperature,
        dampening=dampening,
        verbose=verbose)

    # Compute the mean logits for each predicted object, weighted by
    # the node sizes
    obj_logits = scatter_mean_weighted(node_logits, obj_index, node_size)
    obj_y = obj_logits.argmax(dim=1)

    # Identify, out of the predicted objects, which are of type stuff.
    # These will need to be merged to ensure there as most one instance
    # of each stuff class in each scene
    obj_is_stuff = get_stuff_mask(obj_y, stuff_classes)

    # Distribute the object-wise labels to the nodes
    node_obj_y = obj_y[obj_index]
    node_is_stuff = obj_is_stuff[obj_index]

    # Since we only want at most one prediction of each stuff class
    # per batch item (ie per scene), we assign nodes predicted as a
    # stuff class to new indices. These new indices are built in
    # such a way that there can be only one instance of each stuff
    # class per batch item
    batch = batch if batch is not None else torch.zeros_like(obj_index)
    num_batch_items = batch.max() + 1
    final_obj_index = obj_index.clone()
    final_obj_index[node_is_stuff] = \
        obj_index.max() + 1 \
        + node_obj_y[node_is_stuff] * num_batch_items \
        + batch[node_is_stuff]
    final_obj_index, perm = consecutive_cluster(final_obj_index)

    return final_obj_index


def oracle_superpoint_clustering(
        nag,
        num_classes,
        stuff_classes,
        mode='pas',
        graph_kwargs=None,
        partition_kwargs=None):
    """Compute an oracle for superpoint clustering for instance and
    panoptic segmentation. This is a proxy for the highest achievable
    graph clustering performance with the superpoint partition at hand
    and the input clustering parameters.

    The output `InstanceData` can then be used to compute final
    segmentation metrics using:
      - `InstanceData.semantic_segmentation_oracle()`
      - `InstanceData.instance_segmentation_oracle()`
      - `InstanceData.panoptic_segmentation_oracle()`

    More precisely, for the optimal superpoint clustering:
      - build the instance graph on the input `NAG` `level`-partition
      - for each edge, the oracle perfectly predicts the affinity
      - for each node, the oracle perfectly predicts the offset
      - for each node, the oracle predicts the dominant label from its
        label histogram (excluding the 'void' label)
      - partition the instance graph using the oracle edge affinities,
        node offsets and node classes
      - merge superpoints if they are assigned to the same object
      - merge 'stuff' predictions together, so that there is at most 1
        prediction of each 'stuff' class per batch item

    :param nag: NAG object
    :param num_classes: int
        Number of classes in the dataset, allows differentiating between
        valid and void classes
    :param stuff_classes: List[int]
        List of labels for 'stuff' classes
    :param mode: str
        String characterizing whether edge affinities, node semantics,
        positions and offsets should be used in the graph clustering.
        'p': use node position.
        'o': use oracle offset.
        'a': use oracle edge affinities.
        's': use oracle node semantics.
        In contrast, not setting 'p', nor 'o' is equivalent to setting
        all nodes positions and offsets to 0.
        Similarly, not setting 'a' will set the same weight to all the
        edges.
        Finally, not setting 's' will set the same class to all the
        nodes.
    :param graph_kwargs: dict
        Dictionary of kwargs to be passed to the graph constructor
        `OnTheFlyInstanceGraph()`
    :param partition_kwargs: dict
        Dictionary of kwargs to be passed to the partition function
        `instance_cut_pursuit()`
    :return:
    """

    # TODO: maybe remove this function, redundant with grid_search_panoptic_partition

    # Local import to avoid import loop errors
    from src.transforms import OnTheFlyInstanceGraph
    from src.models.panoptic import PanopticSegmentationOutput
    from src.metrics import PanopticQuality3D

    # Instance graph computation
    graph_kwargs = {} if graph_kwargs is None else graph_kwargs
    graph_kwargs = dict(graph_kwargs, **dict(level=1, num_classes=num_classes))
    nag = OnTheFlyInstanceGraph(**graph_kwargs)(nag)

    # Get node target semantics, size and instance graph
    node_y = nag[1].y[:, :num_classes].argmax(dim=1)
    node_size = nag.get_sub_size(1)
    edge_index = nag[1].obj_edge_index

    # Prepare input for instance graph partition. If 's' is used, the
    # oracle will assign the target semantic label to each node
    # NB: we assign only to valid classes and ignore void
    # NB2: `instance_cut_pursuit()` expects logits, which it converts to
    # probabilities using a softmax, hence the `one_hot * 100`
    node_logits = one_hot(node_y, num_classes=num_classes).float() * 100

    # Otherwise, the nodes will all have the same logits and the
    # semantics will not influence the partition
    if 's' not in mode.lower():
        partition_kwargs['p_weight'] = 0

    # Prepare edge affinity logits. If affinities are not used, we set
    # all edge affinity logits to 0 (ie 0.5 sigmoid-ed weights)
    edge_affinity_logits = torch.special.logit(nag[1].obj_edge_affinity) \
        if 'a' in mode.lower() \
        else torch.zeros(edge_index.shape[1], device=nag.device)

    # Prepare node position features. If 'o' is used, the oracle
    # perfectly predicts the offset to the object center for each node,
    # except for stuff and void classes, whose offset is set to 0
    if 'o' in mode.lower():
        node_x = nag[1].obj_pos
        is_stuff = get_stuff_mask(node_y, stuff_classes)
        node_x[is_stuff] = nag[1].pos[is_stuff]

    # If 'p' only node positions are used
    elif 'p' in mode.lower():
        node_x = nag[1].pos

    # Otherwise, positions and offsets are not used in the partition
    else:
        partition_kwargs['x_weight'] = 0
        node_x = nag[1].pos * 0

    # For each node, recover the index of the batch item it belongs to
    batch = nag[1].batch if nag[1].batch is not None \
        else torch.zeros(nag[1].num_nodes, dtype=torch.long, device=nag.device)

    # Instance graph partition
    partition_kwargs = {} if partition_kwargs is None else partition_kwargs
    obj_index = instance_cut_pursuit(
        batch,
        node_x,
        node_logits,
        stuff_classes,
        node_size,
        edge_index,
        edge_affinity_logits,
        **partition_kwargs)

    # Gather results in an output object
    output = PanopticSegmentationOutput(
        node_logits,
        stuff_classes,
        edge_affinity_logits,
        # node_offset_pred,
        node_size)

    # Store the panoptic segmentation results in the output object
    output.obj_edge_index = getattr(nag[1], 'obj_edge_index', None)
    output.obj_edge_affinity = getattr(nag[1], 'obj_edge_affinity', None)
    output.pos = nag[1].pos
    output.obj_pos = getattr(nag[1], 'obj_pos', None)
    output.obj = nag[1].obj
    output.y_hist = nag[1].y
    output.obj_index_pred = obj_index

    # Create the metrics tracking objects
    panoptic_metrics = PanopticQuality3D(
        num_classes,
        ignore_unseen_classes=True,
        stuff_classes=stuff_classes,
        compute_on_cpu=True)

    # Recover the predicted instance score, semantic label and instance
    # partition
    obj_score, obj_y, instance_data = output.panoptic_pred

    # Compute the metrics on the oracle partition
    panoptic_metrics.update(obj_y, instance_data.cpu())
    results = panoptic_metrics.compute()

    return results


def get_stuff_mask(y, stuff_classes):
    """Helper function producing a boolean mask of size `y.shape[0]`
    indicating which of the `y` (labels if 1D or logits/probabilities if
    2D) are among the `stuff_classes`.
    """
    # Get labels from y, in case y are logits
    labels = y.long() if y.dim() == 1 else y.argmax(dim=1)

    # Search the labels belonging to the set of stuff classes
    stuff_classes = torch.as_tensor(
        stuff_classes, dtype=labels.dtype, device=labels.device)
    return torch.isin(labels, stuff_classes)


def compute_panoptic_metrics(
        model,
        datamodule,
        stage='val',
        graph_kwargs=None,
        partition_kwargs=None,
        verbose=True):
    """Helper function to compute the semantic, instance, panoptic
    segmentation metrics of a model on a given dataset, for given
    instance graph and partition parameters.
    """
    # Local imports to avoid import loop errors
    from src.data import NAGBatch

    # Pick among train, val, and test datasets. It is important to note that
    # the train dataset produces augmented spherical samples of large
    # scenes, while the val and test dataset
    if stage == 'train':
        dataset = datamodule.train_dataset
        dataloader = datamodule.train_dataloader()
    elif stage == 'val':
        dataset = datamodule.val_dataset
        dataloader = datamodule.val_dataloader()
    elif stage == 'test':
        dataset = datamodule.test_dataset
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unknown stage : {stage}")

    # Prevent `NAGAddKeysTo` from removing attributes to allow
    # visualizing them after model inference
    dataset = _set_attribute_preserving_transforms(dataset)

    # Set the instance graph construction parameters
    dataset = _set_graph_construction_parameters(dataset, graph_kwargs)

    # Set the partitioner parameters
    model, backup_kwargs = _set_partitioner_parameters(model, partition_kwargs)

    # Load a dataset item. This will return the hierarchical partition
    # of an entire tile, within a NAG object
    with torch.no_grad():
        enum = tqdm(dataloader) if verbose else dataloader
        for nag_list in enum:
            nag = NAGBatch.from_nag_list([nag.cuda() for nag in nag_list])

            # Apply on-device transforms on the NAG object. For the
            # train dataset, this will select a spherical sample of the
            # larger tile and apply some data augmentations. For the
            # validation and test datasets, this will prepare an entire
            # tile for inference
            nag = dataset.on_device_transform(nag)

            model.validation_step(nag, None)

        # Actions taken from on_validation_epoch_end()
        # panoptic_results = model.val_panoptic.compute()
        # instance_miou = model.val_semantic.miou()
        # instance_oa = model.val_semantic.oa()
        # instance_macc = model.val_semantic.macc()
        panoptic = deepcopy(model.val_panoptic)
        instance = deepcopy(model.val_instance)
        semantic = deepcopy(model.val_semantic)
        model.val_affinity_oa.reset()
        model.val_affinity_f1.reset()
        model.val_panoptic.reset()
        model.val_semantic.reset()
        model.val_instance.reset()

    # Restore the partitioner initial kwargs
    model, _ = _set_partitioner_parameters(model, backup_kwargs)

    if not verbose:
        return panoptic, instance, semantic

    for k, v in panoptic.compute().items():
        print(f"{k:<22}: {v}")

    if not model.no_instance_metrics:
        for k, v in instance.compute().items():
            print(f"{k:<22}: {v}")

    print(f"mIoU                  : {semantic.miou().cpu().item()}")

    return panoptic, instance, semantic


def compute_metrics_s3dis_6fold(
        fold_ckpt,
        experiment_config,
        stage='val',
        graph_kwargs=None,
        partition_kwargs=None,
        verbose=False):
    """Helper function to compute the semantic, instance, panoptic
    segmentation metrics of a model on a S3DIS 6-fold, for given
    instance graph and partition parameters.

    :param fold_ckpt: dict
        Dictionary with S3DIS folds as keys and checkpoint paths as
        values
    :param experiment_config: str
        Experiment config to use for inference. For instance for S3DIS
        with stuff panoptic segmentation: 'panoptic/s3dis_with_stuff'
    :param stage: str
    :param graph_kwargs: dict
    :param partition_kwargs: dict
    :param verbose: bool
    :return:
    """
    # Local import to avoid import loop errors
    from src.metrics import PanopticQuality3D, MeanAveragePrecision3D, \
        ConfusionMatrix

    # Very ugly fix to ignore lightning's warning messages about the
    # trainer and modules not being connected
    import warnings
    warnings.filterwarnings("ignore")

    panoptic_list = []
    instance_list = []
    semantic_list = []
    no_instance_metrics = None
    min_instance_size = None
    num_classes = None
    stuff_classes = None

    for fold, ckpt_path in fold_ckpt.items():

        if verbose:
            print(f"\nFold {fold}")

        # Parse the configs using hydra
        cfg = init_config(overrides=[
            f"experiment={experiment_config}",
            f"datamodule.fold={fold}",
            f"ckpt_path={ckpt_path}"])

        # Instantiate the datamodule
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.prepare_data()
        datamodule.setup()

        # Instantiate the model
        model = hydra.utils.instantiate(cfg.model)

        # Load pretrained weights from a checkpoint file
        model = model.__class__.load_from_checkpoint(
            cfg.ckpt_path,
            net=model.net,
            edge_affinity_head=model.edge_affinity_head,
            partitioner=model.partitioner,
            criterion=model.criterion)
        # model.criterion = hydra.utils.instantiate(cfg.model).criterion
        model = model.eval().cuda()

        # Compute metrics on the fold
        panoptic, instance, semantic = compute_panoptic_metrics(
            model,
            datamodule,
            stage=stage,
            graph_kwargs=graph_kwargs,
            partition_kwargs=partition_kwargs,
            verbose=verbose)

        # Gather some details from the model and datamodule before
        # deleting them
        no_instance_metrics = model.no_instance_metrics
        min_instance_size = model.hparams.min_instance_size
        num_classes = datamodule.train_dataset.num_classes
        stuff_classes = datamodule.train_dataset.stuff_classes

        del model, datamodule

        # Store the metrics for each fold
        panoptic_list.append(panoptic)
        instance_list.append(instance)
        semantic_list.append(semantic)

    # Initialize the 6-fold metrics
    panoptic_6fold = PanopticQuality3D(
        num_classes,
        ignore_unseen_classes=True,
        stuff_classes=stuff_classes,
        compute_on_cpu=True)

    instance_6fold = MeanAveragePrecision3D(
        num_classes,
        stuff_classes=stuff_classes,
        min_size=min_instance_size,
        compute_on_cpu=True,
        remove_void=True)

    semantic_6fold = ConfusionMatrix(num_classes)

    # Group together per-fold panoptic and semantic results
    for i in range(len(panoptic_list)):

        panoptic_6fold.instance_data += panoptic_list[i].instance_data
        panoptic_6fold.prediction_semantic += panoptic_list[i].prediction_semantic

        if not no_instance_metrics:
            instance_6fold.prediction_score += instance_list[i].prediction_score
            instance_6fold.prediction_semantic += instance_list[i].prediction_semantic
            instance_6fold.instance_data += instance_list[i].instance_data

        semantic_6fold.confmat += semantic_list[i].confmat.cpu()

    # Print computed the metrics
    print(f"\n6-fold")
    for k, v in panoptic_6fold.compute().items():
        print(f"{k:<22}: {v}")

    if not no_instance_metrics:
        for k, v in instance_6fold.compute().items():
            print(f"{k:<22}: {v}")

    print(f"mIoU                  : {semantic_6fold.miou().cpu().item()}")

    return (panoptic_6fold, panoptic_list), (instance_6fold, instance_list), (semantic_6fold, semantic_list)


def _set_attribute_preserving_transforms(dataset):
    """For the sake of visualization, we require that `NAGAddKeysTo`
    does not remove input `Data` attributes after moving them to `Data.x`,
    so we may visualize them.
    """
    # Local imports to avoid import loop errors
    from src.transforms import NAGAddKeysTo

    for t in dataset.on_device_transform.transforms:
        if isinstance(t, NAGAddKeysTo):
            t.delete_after = False

    return dataset


def _set_graph_construction_parameters(dataset, graph_kwargs):
    """Searches for the last occurrence of `OnTheFlyInstanceGraph` among
    the `on_device_transform` of the dataset and modifies the graph
    construction parameters passed in the `graph_kwargs` dictionary.
    """
    if graph_kwargs is None:
        return dataset

    # Local imports to avoid import loop errors
    from src.transforms import OnTheFlyInstanceGraph

    # Search for the `OnTheFlyInstanceGraph` instance graph construction
    # transform among the on-device transforms
    i_transform = None
    for i, transform in enumerate(dataset.on_device_transform.transforms):
        if isinstance(transform, OnTheFlyInstanceGraph):
            i_transform = i

    # Set OnTheFlyInstanceGraph parameters if need be
    if i_transform is not None and graph_kwargs is not None:
        for k, v in graph_kwargs.items():
            setattr(dataset.on_device_transform.transforms[i_transform], k, v)

    return dataset


def _set_partitioner_parameters(model, partition_kwargs):
    """Modifies the `model.partitioner` parameters with parameters
    passed in the `partition_kwargs` dictionary.
    """
    backup_kwargs = {}

    if partition_kwargs is None:
        return model, backup_kwargs

    # Set partitioner parameters if need be
    if partition_kwargs is not None:
        for k, v in partition_kwargs.items():
            backup_kwargs[k] = getattr(model.partitioner, k, None)
            setattr(model.partitioner, k, v)

    return model, backup_kwargs


def _forward_multi_partition(
        model,
        nag,
        partition_kwargs,
        mode='pas'):
    """Local helper to compute multiple instance partitions from the
    same input data, based on diverse partition parameter settings.
    """
    # Local import to avoid import loop errors
    from src.models.panoptic import PanopticSegmentationOutput

    # Make sure each element of `partition_kwargs` is a list,
    # to facilitate computing Cartesian product of the lists for
    # grid search
    partition_kwargs = {
        k: v if isinstance(v, list) else [v]
        for k, v in partition_kwargs.items()}

    with torch.no_grad():

        # Extract features
        x = model.net(nag)

        # Compute level-1 or multi-level semantic predictions
        semantic_pred = [head(x_) for head, x_ in zip(model.head, x)] \
            if model.multi_stage_loss else model.head(x)

        # Recover level-1 features only
        x = x[0] if model.multi_stage_loss else x

        # Compute edge affinity predictions
        # NB: we make edge features symmetric, since we want to compute
        # edge affinity, which is not directed
        x_edge = x[nag[1].obj_edge_index]
        x_edge = torch.cat(
            ((x_edge[0] - x_edge[1]).abs(), (x_edge[0] + x_edge[1]) / 2), dim=1)
        edge_affinity_logits = model.edge_affinity_head(x_edge).squeeze()

        # Ignore predicted affinities (sets all edge affinity logits to
        # 0, which will set edge weights to 0.5 for the partition)
        if 'a' not in mode.lower():
            edge_affinity_logits = edge_affinity_logits * 0

        # Oracle edge affinities
        elif 'A' in mode:
            edge_affinity_logits = torch.special.logit(nag[1].obj_edge_affinity)

        # Ignore predicted semantic labels
        if 's' not in mode.lower():
            partition_kwargs['p_weight'] = [0]

        # Oracle node semantics predicts perfect semantic logits for
        # each node
        # NB: we assign only to valid classes and ignore void
        # NB2: `instance_cut_pursuit()` expects logits, which it
        # converts to probabilities using a softmax, hence the
        # `one_hot * 10`
        elif 'S' in mode:
            node_y = nag[1].y[:, :model.num_classes].argmax(dim=1)
            node_logits = one_hot(
                node_y, num_classes=model.num_classes).float() * 10
            if model.multi_stage_loss:
                semantic_pred[0] = node_logits
            else:
                semantic_pred = node_logits

        # Ignore positions and predicted offsets
        if 'p' not in mode.lower() and 'o' not in mode.lower():
            partition_kwargs['x_weight'] = [0]

        # Compute node offset predictions
        elif 'o' in mode:
            node_offset_pred = model.node_offset_head(x)

            # Forcefully set 0-offset for nodes with stuff predictions
            node_logits = semantic_pred[0] if model.multi_stage_loss \
                else semantic_pred
            is_stuff = get_stuff_mask(node_logits, model.stuff_classes)
            node_offset_pred[is_stuff] = 0

        # Oracle node offsets sets perfect offsets for all nodes and
        # keeps node centroid for nodes with target stuff label
        # (ie 0-offset)
        elif 'O' in mode:
            is_stuff = get_stuff_mask(nag[1].y, model.stuff_classes)
            nag[1].pos[~is_stuff] = nag[1].obj_pos[~is_stuff]

        # Compute the partition on the Cartesian product of parameters
        partition_keys = list(partition_kwargs.keys())
        enum = [
            {k: v for k, v in zip(partition_keys, values)}
            for values in product(*partition_kwargs.values())]
        partitions = {}
        for kwargs in tqdm(enum):

            # Apply the kwargs to the partitioner
            model, backup_kwargs = _set_partitioner_parameters(model, kwargs)

            # Gather results in an output object
            output = PanopticSegmentationOutput(
                semantic_pred,
                model.stuff_classes,
                edge_affinity_logits,
                # node_offset_pred,
                nag.get_sub_size(1))

            # Compute the panoptic partition
            output = model._forward_partition(nag, output)

            # Store the predicted partition wrt the parameter values
            # (can't directly store kwargs dict because unhashable)
            partitions[tuple(kwargs.values())] = output.obj_index_pred

            # Restore the initial partitioner kwargs
            model, _ = _set_partitioner_parameters(model, backup_kwargs)

        output = model.get_target(nag, output)

    return output, partitions, partition_keys


def grid_search_panoptic_partition(
        model,
        dataset,
        i_cloud=0,
        graph_kwargs=None,
        partition_kwargs=None,
        mode='pas',
        panoptic=True,
        instance=False):
    """Runs a grid search on the partition parameters to find the best
    setup on a given sample `dataset[i_cloud]`.

    :param model: PanopticSegmentationModule
    :param dataset: BaseDataset
    :param i_cloud: int
        The grid search will be computed on `dataset[i_cloud]`
    :param graph_kwargs: dict
        Dictionary of parameters to be passed to the instance graph
        constructor `OnTheFlyInstanceGraph`. NB: the grid search does
        not cover these parameters---only a single value can be passed
        for each of these parameters
    :param partition_kwargs: dict
        Dictionary of parameters to be passed to `model.partitioner`.
        Passing a list of values for a given parameter will trigger the
        grid search across these values. Beware of the combinatorial
        explosion !
    :param mode: str
        String characterizing whether edge affinities, node semantics,
        positions and offsets should be used in the graph clustering.
        'p': use node position.
        'o': use predicted node offset.
        'O': use oracle offset.
        'a': use predicted edge affinity.
        'A': use oracle edge affinities.
        's': use predicted node semantics.
        'S': use oracle node semantics.
        In contrast, not setting 'p', 'o', nor 'O' is equivalent to
        setting all node positions and offsets to 0.
        Similarly, not setting 'a' nor 'A' will set the same weight to
        all the edges.
        Finally, not setting 's', nor 'S' will set the same class to all
        the nodes.
    :param panoptic: bool
        Whether panoptic segmentation metrics should be computed
    :param instance: bool
        Whether instance segmentation metrics should be computed
    :return:
    """
    # TODO: grid search on the whole dataset rather than a single cloud

    # Local import to avoid import loop errors
    from src.metrics import PanopticQuality3D, MeanAveragePrecision3D

    assert panoptic or instance, \
        "At least 'panoptic' or 'instance' must be True"

    # Limit the column header size for printed tables
    max_len = 6

    # Prevent `NAGAddKeysTo` from removing attributes to allow
    # visualizing them after model inference
    dataset = _set_attribute_preserving_transforms(dataset)

    # Set the instance graph construction parameters
    dataset = _set_graph_construction_parameters(dataset, graph_kwargs)

    # Load a dataset item. This will return the hierarchical partition
    # of an entire tile, within a NAG object
    nag = dataset[i_cloud]

    # Apply on-device transforms on the NAG object. For the train
    # dataset, this will select a spherical sample of the larger tile
    # and apply some data augmentations. For the validation and test
    # datasets, this will prepare an entire tile for inference
    nag = dataset.on_device_transform(nag.cuda())

    # Compute the partition for each parameterization
    output, partitions, partition_keys = _forward_multi_partition(
        model,
        nag,
        partition_kwargs,
        mode=mode)

    # Get the target labels
    output = model.get_target(nag, output)

    # Create the metrics tracking objects
    instance_metrics = MeanAveragePrecision3D(
        model.num_classes,
        stuff_classes=model.stuff_classes,
        min_size=model.hparams.min_instance_size,
        compute_on_cpu=True,
        remove_void=True)

    panoptic_metrics = PanopticQuality3D(
        model.num_classes,
        ignore_unseen_classes=True,
        stuff_classes=model.stuff_classes,
        compute_on_cpu=True)

    # Compute and print metric results for each partition setup
    results = {}
    results_data = []
    best_pq = -1
    best_map = -1
    best_pq_params = None
    best_map_params = None
    for (kwargs_values), obj_index_pred in partitions.items():

        # Reconstruct the kwargs dict from the kwargs values
        kwargs = {k: v for k, v in zip(partition_keys, kwargs_values)}

        output.obj_index_pred = obj_index_pred

        obj_score, obj_y, instance_data = output.panoptic_pred
        obj_score = obj_score.detach().cpu()
        obj_y = obj_y.detach().cpu()

        if panoptic:
            panoptic_metrics.update(obj_y, instance_data.cpu())
            panoptic_results = panoptic_metrics.compute()
            panoptic_metrics.reset()
            if panoptic_results.pq > best_pq:
                best_pq_params = tuple(kwargs.values())
                best_pq = panoptic_results.pq
        else:
            panoptic_results = None

        if instance:
            instance_metrics.update(obj_score, obj_y, instance_data.cpu())
            instance_results = instance_metrics.compute()
            instance_metrics.reset()
            if instance_results.map > best_map:
                best_map_params = tuple(kwargs.values())
                best_map = instance_results.map
        else:
            instance_results = None

        # Store the panoptic and instance metric results for the
        # parameters at hand
        results[tuple(kwargs.values())] = (panoptic_results, instance_results)

        # Track the results to build a global summary DataFrame
        current_results = [*kwargs.values()]
        if panoptic:
            current_results += [
                round(panoptic_results.pq.item() * 100, 2),
                round(panoptic_results.sq.item() * 100, 2),
                round(panoptic_results.rq.item() * 100, 2)]
        if instance:
            current_results += [
                round(instance_results.map.item() * 100, 2),
                round(instance_results.map_50.item() * 100, 2)]
        results_data.append(current_results)

    # Print a DataFrame summarizing the results
    metric_columns = []
    if panoptic:
        metric_columns += ['PQ', 'SQ', 'RQ']
    if instance:
        metric_columns += ['mAP', 'mAP 50']
    with pd.option_context('display.precision', 2):
        print(pd.DataFrame(
            data=results_data,
            columns=[
                *[
                    x[:max_len - 1] + '.' if len(x) > max_len else x
                    for x in partition_keys
                ],
                *metric_columns]))
    print()

    # Print more details about the best panoptic setup
    if panoptic and best_pq_params is not None:

        # Print global results
        print(f"\nBest panoptic setup: PQ={100 * best_pq:0.2f}")
        with pd.option_context('display.precision', 2):
            print(pd.DataFrame(
                data=[best_pq_params],
                columns=[
                    x[:max_len - 1] + '.' if len(x) > max_len else x
                    for x in partition_keys]))

        print()

        # Print per-class results
        res = results[best_pq_params][0]
        with pd.option_context('display.precision', 2):
            print(pd.DataFrame(
                data=torch.column_stack([
                    res.pq_per_class.mul(100),
                    res.sq_per_class.mul(100),
                    res.rq_per_class.mul(100),
                    res.precision_per_class.mul(100),
                    res.recall_per_class.mul(100),
                    res.tp_per_class,
                    res.fp_per_class,
                    res.fn_per_class]),
                index=dataset.class_names[:-1],
                columns=['PQ', 'SQ', 'RQ', 'PREC.', 'REC.', 'TP', 'FP', 'FN']))
        print()

        # Store the best panoptic partition indexing in the output
        output.obj_index_pred = partitions[best_pq_params]

    # Print more details about the best instance setup
    if instance and best_map_params is not None:

        # Print global results
        print(f"\nBest instance setup: mAP={100 * best_map:0.2f}")
        with pd.option_context('display.precision', 2):
            print(pd.DataFrame(
                data=[best_map_params],
                columns=[
                    x[:max_len - 1] + '.' if len(x) > max_len else x
                    for x in partition_keys]))
        print()

        # Print per-class results
        res = results[best_map_params][1]
        thing_class_names = [
            c for i, c in enumerate(dataset.class_names) if i in dataset.thing_classes]
        with pd.option_context('display.precision', 2):
            print(pd.DataFrame(
                data=torch.column_stack([res.map_per_class.mul(100)]),
                index=thing_class_names,
                columns=['mAP']))
        print()

    return output, partitions, results
