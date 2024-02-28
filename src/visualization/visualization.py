import torch
import numpy as np
import os.path as osp
import plotly.graph_objects as go
from src.data import Data, NAG, Cluster
from src.transforms import GridSampling3D, SaveNodeIndex
from src.utils import fast_randperm, to_trimmed
from torch_scatter import scatter_mean
from src.utils.color import *


# TODO: To go further with ipwidgets :
#  - https://plotly.com/python/figurewidget-app/
#  - https://ipywidgets.readthedocs.io/en/stable/


def visualize_3d(
        input,
        figsize=1000,
        width=None,
        height=None,
        class_names=None,
        class_colors=None,
        stuff_classes=None,
        num_classes=None,
        hide_void_pred=False,
        voxel=-1,
        max_points=50000,
        point_size=3,
        centroid_size=None,
        error_color=None,
        centroids=False,
        v_edge=False,
        h_edge=False,
        h_edge_attr=False,
        gap=None,
        radius=None,
        center=None,
        select=None,
        alpha=0.1,
        alpha_super=None,
        alpha_stuff=0.2,
        h_edge_width=None,
        v_edge_width=None,
        point_symbol='circle',
        centroid_symbol='circle',
        colorscale='Agsunset',
        keys=None,
        **kwargs):
    """3D data interactive visualization.

    :param input: Data or NAG object
    :param figsize: int
        Figure dimensions will be (figsize, figsize/2) if `width` and
        `height` are not specified
    :param width: int
        Figure width
    :param height: int
        Figure height
    :param class_names: List(str)
        Names for point labels found in attributes 'y' and
        'semantic_pred'
    :param class_colors: List(List(int, int, int))
        Colors palette for point labels found in attributes 'y' and
        'semantic_pred'
    :param stuff_classes: List(int)
        Semantic labels of the classes considered as 'stuff' for
        instance and panoptic segmentation. If 'y' and 'obj' are found
        in the point attributes, the stuff annotations will appear
        accordingly. Otherwise, stuff instance labeling will appear as
        any other object
    :param num_classes: int
        Number of valid classes. By convention, we assume
        `y ∈ [0, num_classes-1]` are VALID LABELS, while
        `y < 0` AND `y >= num_classes` ARE VOID LABELS
    :param voxel: float
        Voxel size to subsample the point cloud to facilitate
        visualization
    :param max_points: int
        Maximum number of points displayed to facilitate visualization
    :param point_size: int or float
        Size of point markers
    :param centroid_size: int or float
        Size of superpoint markers
    :param error_color: List(int, int, int)
        Color used to identify mis-predicted points
    :param centroids: bool
        Whether superpoint centroids should be displayed
    :param v_edge: bool
        Whether vertical edges should be displayed (only if
        centroids=True and gap is not None)
    :param h_edge: bool
        Whether horizontal edges should be displayed (only if
        centroids=True)
    :param h_edge_attr: bool
        Whether the edges should be colored by their features found in
        'edge_attr' (only if h_edge=True)
    :param gap: List(float, float, float)
        If None, the hierarchical graphs will be overlaid on the points.
        If not None, a 3D tensor indicating the offset by which the
        hierarchical graphs should be plotted
    :param radius: float
        If not None, only visualize a spherical sampling of the input
        data, centered on `center` and with size `radius`. This option
        is not compatible with `select`
    :param center: List(float, float, float)
        If `radius` is provided, only visualize a spherical sampling of
        the input data, centered on `center` and with size `radius`. If
        None, the center of the scene will be used
    :param select:
        If not None, will call Data.select(select) or
        NAG.select(*select) on the input data (depending on its nature)
        and the coloring schemes will illustrate it. This option is not
        compatible with `radius`
    :param alpha: float
        Rules the whitening of selected points, nodes and edges (only if
         select is not None)
    :param alpha_super:
        Float ruling the whitening of superpoints (only if select is not
        None). If None, alpha will be used as fallback
    :param alpha_stuff:
        Float ruling the whitening of stuff points (only if the input
        points have 'obj' and 'semantic_pred' attributes, and
        'stuff_classes' or 'num_classes' is specified). If None,
        `alpha` will be used as fallback
    :param point_symbol: str
        Marker symbol used for points. Must be one of
        ['circle', 'circle-open', 'square', 'square-open', 'diamond',
        'diamond-open', 'cross', 'x']. Defaults to 'circle'
    :param centroid_symbol: str
        Marker symbol used for centroids. Must be one of
        ['circle', 'circle-open', 'square', 'square-open', 'diamond',
        'diamond-open', 'cross', 'x']. Defaults to 'circle'
    :param colorscale: str
        Plotly colorscale used for coloring 1D features. See
        https://plotly.com/python/builtin-colorscales for options
    :param kwargs

    :return:
    """
    # assert isinstance(input, (Data, NAG))
    gap = torch.tensor(gap) if gap is not None else gap
    assert gap is None or gap.shape == torch.Size([3])
    assert not (radius and (select is not None)), \
        "Cannot use both a `radius` and `select` at once"

    # We work on copies of the input data, to allow modified in this
    # scope
    input = input.clone().cpu()

    # If the input is a simple Data object, we convert it to a NAG
    input = NAG([input]) if isinstance(input, Data) else input

    # If the last level of the NAG has super_index, we manually
    # construct an additional Data level and append it to the NAG
    if input[input.num_levels - 1].is_sub:
        data_last = input[input.num_levels - 1]
        sub = Cluster(
            data_last.super_index, torch.arange(data_last.num_nodes),
            dense=True)
        obj = data_last.obj.merge(data_last.super_index)
        pos = scatter_mean(data_last.pos, data_last.super_index, dim=0)
        input = NAG(input.to_list() + [Data(pos=pos, sub=sub, obj=obj)])
    is_nag = isinstance(input, NAG)
    num_levels = input.num_levels if is_nag else 1

    # Make sure alpha is in [0, 1]
    alpha = max(0, min(alpha, 1))
    alpha_super = max(0, min(alpha_super, 1)) if alpha_super else alpha
    alpha_stuff = max(0, min(alpha_stuff, 1)) if alpha_stuff else alpha

    # If `radius` is provided, we only visualize a spherical selection
    # of size `radius` around the `center`
    if radius is not None:
        # If no `center` provided, pick the middle of the scene
        if center is None:
            hi = input[0].pos.max(dim=0).values
            lo = input[0].pos.min(dim=0).values
            center = (hi + lo) / 2

            # For Z, we center on the average Z, because the middle
            # value may cause empty samplings for outdoor scenes with
            # some very high objects and most of the interesting stuff
            # happening near the ground 
            center[2] = input[0].pos[:, 2].mean()
        else:
            center = torch.as_tensor(center).cpu()
        center = center.view(1, -1)

        # Create a mask on level-0 (ie points) to be used for indexing
        # the NAG structure
        mask = torch.where(
            torch.linalg.norm(input[0].pos - center, dim=1) < radius)[0]

        # Subselect the hierarchical partition based on the level-0 mask
        input = input.select(0, mask)

    # If `select` is provided, we will call NAG.select on the input data
    # and illustrate the selected/discarded pattern in the figure
    if select is not None and is_nag:

        # Add an ID to the points before applying NAG.select
        nag_temp = input.clone()
        for i in range(nag_temp.num_levels):
            nag_temp._list[i] = SaveNodeIndex()(nag_temp[i])

        # Apply the selection
        nag_temp = nag_temp.select(*select)

        # Indicate, for each node of the hierarchical graph, whether it
        # has been selected
        for i in range(num_levels):
            selected = torch.zeros(input[i].num_nodes, dtype=torch.bool)
            selected[nag_temp[i][SaveNodeIndex.DEFAULT_KEY]] = True
            input[i].selected = selected

        del nag_temp, selected

    elif select is not None and not is_nag:

        # Add an ID to the points before applying NAG.select
        data_temp = SaveNodeIndex()(Data(pos=input.pos.clone()))

        # Apply the selection
        data_temp = data_temp.select(select)[0]

        # Indicate, for each node of the hierarchical graph, whether it
        # has been selected
        selected = torch.zeros(input.num_nodes, dtype=torch.bool)
        selected[data_temp[SaveNodeIndex.DEFAULT_KEY]] = True
        input.selected = selected

        del data_temp, selected

    elif is_nag:
        for i in range(num_levels):
            input[i].selected = torch.ones(
                input[i].num_nodes, dtype=torch.bool)

    else:
        input.selected = torch.ones(input.num_nodes, dtype=torch.bool)

    # Data_0 accounts for the lowest level of hierarchy, the points
    # themselves
    data_0 = input[0] if is_nag else input

    # Subsample to limit the drawing time
    # If the level-0 cloud needs to be voxelized or sampled, a NAG
    # structure will be affected too. To maintain NAG consistency, we
    # only support 'GridSampling3D' with mode='last' and random sampling
    # without replacement. To keep track of the sampled points and index
    # the NAG accordingly, we use 'SaveNodeIndex'
    idx = torch.arange(data_0.num_points)

    # If a voxel size is specified, voxelize the level-0. We first
    # isolate the 'pos' and the input indices of data_0 and apply
    # voxelization on this. We then recover the original grid-sampled
    # points indices to be used with Data.select or NAG.select
    if voxel > 0:
        data_temp = SaveNodeIndex()(Data(pos=data_0.pos.clone()))
        data_temp = GridSampling3D(voxel, mode='last')(data_temp)
        idx = data_temp[SaveNodeIndex.DEFAULT_KEY]
        del data_temp

    # If the cloud is too large with respect to required 'max_points',
    # sample without replacement
    if idx.shape[0] > max_points:
        idx = idx[fast_randperm(idx.shape[0])[:max_points]]

    # If a sampling is needed, apply it to the input Data or NAG,
    # depending on the structure
    if idx.shape[0] < data_0.num_points:
        input = input.select(0, idx) if is_nag else input.select(idx)[0]
        data_0 = input[0] if is_nag else input

    # Round to the cm for cleaner hover info
    data_0.pos = (data_0.pos * 100).round() / 100

    # Class colors initialization
    if class_colors is not None and not isinstance(class_colors[0], str):
        class_colors = np.asarray(class_colors)
    else:
        class_colors = None

    # Prepare figure
    width = width if width and height else figsize
    height = height if width and height else int(figsize / 2)
    margin = int(0.02 * min(width, height))
    layout = go.Layout(
        width=width,
        height=height,
        scene=dict(aspectmode='data', ),  # preserve aspect ratio
        margin=dict(l=margin, r=margin, b=margin, t=margin),
        uirevision=True)
    fig = go.Figure(layout=layout)

    # To keep track of which trace should be seen under which mode
    # (i.e. button), we build trace_modes. This is a list of dictionaries
    # indicating, for each trace (list element), which mode (dict key)
    # it should appear in and with which attributes (values are dict of
    # parameters for plotly figure updates)
    trace_modes = []
    i_point_trace = 0
    i_unselected_point_trace = 1

    # Draw a trace for position-colored 3D point cloud
    mini = data_0.pos.min(dim=0).values
    maxi = data_0.pos.max(dim=0).values
    colors = (data_0.pos - mini) / (maxi - mini + 1e-6)
    colors = rgb_to_plotly_rgb(colors)
    data_0.pos_colors = colors

    fig.add_trace(
        go.Scatter3d(
            x=data_0.pos[data_0.selected, 0],
            y=data_0.pos[data_0.selected, 1],
            z=data_0.pos[data_0.selected, 2],
            mode='markers',
            marker=dict(
                symbol=point_symbol,
                size=point_size,
                color=colors[data_0.selected]),
            hoverinfo='x+y+z+text',
            hovertext=None,
            showlegend=False,
            visible=True, ))
    trace_modes.append({
        'Position RGB': {
            'marker.color': colors[data_0.selected], 'hovertext': None}})

    fig.add_trace(
        go.Scatter3d(
            x=data_0.pos[~data_0.selected, 0],
            y=data_0.pos[~data_0.selected, 1],
            z=data_0.pos[~data_0.selected, 2],
            mode='markers',
            marker=dict(
                symbol=point_symbol,
                size=point_size,
                color=colors[~data_0.selected],
                opacity=alpha),
            hoverinfo='x+y+z+text',
            hovertext=None,
            showlegend=False,
            visible=True, ))
    trace_modes.append({
        'Position RGB': {
            'marker.color': colors[~data_0.selected], 'hovertext': None}})

    # Draw a trace for RGB 3D point cloud
    if data_0.rgb is not None:
        colors = data_0.rgb
        colors = rgb_to_plotly_rgb(colors)
        data_0.rgb_colors = colors
        trace_modes[i_point_trace]['RGB'] = {
            'marker.color': colors[data_0.selected], 'hovertext': None}
        trace_modes[i_unselected_point_trace]['RGB'] = {
            'marker.color': colors[~data_0.selected], 'hovertext': None}

    # Color the points with ground truth semantic labels. If labels are
    # expressed as histograms, keep the most frequent one
    if data_0.y is not None:
        y = data_0.y
        y = y.argmax(1).numpy() if y.dim() == 2 else y.numpy()
        colors = class_colors[y] if class_colors is not None \
            else int_to_plotly_rgb(torch.LongTensor(y))
        data_0.y_colors = colors
        if class_names is None:
            text = np.array([f"Class {i}" for i in range(y.max() + 1)])
        else:
            text = np.array([str.title(c) for c in class_names])
        text = text[y]
        trace_modes[i_point_trace]['Semantic'] = {
            'marker.color': colors[data_0.selected],
            'hovertext': text[data_0.selected]}
        trace_modes[i_unselected_point_trace]['Semantic'] = {
            'marker.color': colors[~data_0.selected],
            'hovertext': text[~data_0.selected]}

    # Color the points with predicted semantic labels. If labels are
    # expressed as histograms, keep the most frequent one
    if data_0.semantic_pred is not None:
        pred = data_0.semantic_pred
        pred = pred.argmax(1).numpy() if pred.dim() == 2 else pred.numpy()

        # If the ground truth labels are available, we use them to
        # identify void points in the predictions
        if data_0.y is not None and hide_void_pred:
            # Get the target label
            y_gt = data_0.y
            y_gt = y_gt.argmax(1) if y_gt.dim() == 2 else y_gt

            # Create a mask over the points identifying those whose
            # ground truth label is void
            is_void = np.zeros(y_gt.max() + 1, dtype='bool')
            void_classes = [num_classes] if num_classes else []
            for i in void_classes:
                if i < is_void.shape[0]:
                    is_void[i] = True
            is_void = is_void[y_gt]

            # Set the predicted label to void if the ground truth is
            # void, this avoids visualizing predictions on void
            # labels
            pred[is_void] = y_gt[is_void]

        colors = class_colors[pred] if class_colors is not None else None
        data_0.pred_colors = colors
        if class_names is None:
            text = np.array([f"Class {i}" for i in range(pred.max() + 1)])
        else:
            text = np.array([str.title(c) for c in class_names])
        text = text[pred]
        trace_modes[i_point_trace]['Semantic Pred.'] = {
            'marker.color': colors[data_0.selected],
            'hovertext': text[data_0.selected]}
        trace_modes[i_unselected_point_trace]['Semantic Pred.'] = {
            'marker.color': colors[~data_0.selected],
            'hovertext': text[~data_0.selected]}

    # Color the points with ground truth instance labels. If semantic
    # labels and stuff_classes/void_classes also passed, the stuff/void
    # annotations will be treated accordingly
    if data_0.obj is not None and (class_names is None or data_0.y is None):
        obj = data_0.obj if isinstance(data_0.obj, torch.Tensor) \
            else data_0.obj.major(num_classes=num_classes)[0]
        colors = int_to_plotly_rgb(obj)
        data_0.obj_colors = colors
        text = np.array([f"Object {o}" for o in obj])
        trace_modes[i_point_trace]['Panoptic'] = {
            'marker.color': colors[data_0.selected],
            'hovertext': text[data_0.selected]}
        trace_modes[i_unselected_point_trace]['Panoptic'] = {
            'marker.color': colors[~data_0.selected],
            'hovertext': text[~data_0.selected]}
    elif data_0.obj is not None:
        # Colors and text for thing points
        obj = data_0.obj.major(num_classes=num_classes)[0]
        colors_thing = int_to_plotly_rgb(obj)
        text_thing = np.array([f"Object {o}" for o in obj])

        # For simplicity, we merge void_classes into the stuff_classes,
        # the expected behavior is the same, except that we will ensure
        # that the hover text distinguishes between stuff and void
        stuff_classes = stuff_classes if stuff_classes is not None else []
        void_classes = [num_classes] if num_classes else []
        stuff_classes = list(set(stuff_classes).union(set(void_classes)))

        # Colors and text for stuff points
        y = data_0.y
        y = y.argmax(1).numpy() if y.dim() == 2 else y.numpy()
        colors_stuff = class_colors[y] if class_colors is not None \
            else int_to_plotly_rgb(torch.LongTensor(y))
        if class_names is None:
            text_stuff = np.array([
                f"{'Void' if i in void_classes else 'Stuff'} - Class {i}"
                for i in range(y.max() + 1)])
        else:
            text_stuff = np.array([
                f"{'Void' if i in void_classes else 'Stuff'} - {str.title(c)}"
                for i, c in enumerate(class_names)])
        text_stuff = text_stuff[y]

        # Apply alpha-whitening on stuff points
        colors_stuff = colors_stuff.astype('float')
        white = np.full((colors_stuff.shape[0], 3), 255, dtype='float')
        colors_stuff = colors_stuff * alpha_stuff + white * (1 - alpha_stuff)
        colors_stuff = colors_stuff.astype('int64')

        # Compute mask for stuff points
        stuff_classes = np.asarray([i for i in stuff_classes if i <= y.max()])
        is_stuff = np.zeros(y.max() + 1, dtype='bool')
        for i in stuff_classes:
            if i < is_stuff.shape[0]:
                is_stuff[i] = True
        is_stuff = is_stuff[y]

        # Merge thing and stuff colors and text
        colors = colors_thing
        text = text_thing
        colors[is_stuff] = colors_stuff[is_stuff]
        text[is_stuff] = text_stuff[is_stuff]
        data_0.obj_colors = colors

        # Create trace modes
        trace_modes[i_point_trace]['Panoptic'] = {
            'marker.color': colors[data_0.selected],
            'hovertext': text[data_0.selected]}
        trace_modes[i_unselected_point_trace]['Panoptic'] = {
            'marker.color': colors[~data_0.selected],
            'hovertext': text[~data_0.selected]}

    # Color the points with predicted instance labels. If semantic
    # labels and stuff_classes/void_classes also passed, the
    # stuff/void predictions will be treated accordingly. This
    # expects `data_0.obj_pred` to be an InstanceData object
    if getattr(data_0, 'obj_pred', None) is not None and class_names is None:
        obj, _, y = data_0.obj_pred.major(num_classes=num_classes)
        colors = int_to_plotly_rgb(obj)
        data_0.obj_pred_colors = colors
        text = np.array([f"Object {o}" for o in obj])
        trace_modes[i_point_trace]['Panoptic Pred.'] = {
            'marker.color': colors[data_0.selected],
            'hovertext': text[data_0.selected]}
        trace_modes[i_unselected_point_trace]['Panoptic Pred.'] = {
            'marker.color': colors[~data_0.selected],
            'hovertext': text[~data_0.selected]}
    elif getattr(data_0, 'obj_pred', None) is not None:
        # Colors and text for thing points
        obj, _, y = data_0.obj_pred.major(num_classes=num_classes)
        colors_thing = int_to_plotly_rgb(obj)
        text_thing = np.array([f"Object {o}" for o in obj])

        # For simplicity, we merge void_classes into the stuff_classes,
        # the expected behavior is the same, except that we will ensure
        # that the hover text distinguishes between stuff and void
        stuff_classes = stuff_classes if stuff_classes is not None else []
        void_classes = [num_classes] if num_classes else []
        stuff_classes = list(set(stuff_classes).union(set(void_classes)))

        # If the ground truth labels are available, we use them to
        # identify void points in the predictions
        if data_0.y is not None and hide_void_pred:
            # Get the target label
            y_gt = data_0.y
            y_gt = y_gt.argmax(1) if y_gt.dim() == 2 else y_gt

            # Create a mask over the points identifying those whose
            # ground truth label is void
            is_void = np.zeros(y_gt.max() + 1, dtype='bool')
            for i in void_classes:
                if i < is_void.shape[0]:
                    is_void[i] = True
            is_void = is_void[y_gt]

            # Set the predicted label to void if the ground truth is
            # void, this avoids visualizing predictions on void
            # labels
            y[is_void] = y_gt[is_void]

        # Colors and text for stuff points
        colors_stuff = class_colors[y] if class_colors is not None \
            else int_to_plotly_rgb(torch.LongTensor(y))
        if class_names is None:
            text_stuff = np.array([
                f"{'Void' if i in void_classes else 'Stuff'} - Class {i}"
                for i in range(y.max() + 1)])
        else:
            text_stuff = np.array([
                f"{'Void' if i in void_classes else 'Stuff'} - {str.title(c)}"
                for i, c in enumerate(class_names)])
        text_stuff = text_stuff[y]

        # Apply alpha-whitening on stuff points
        colors_stuff = colors_stuff.astype('float')
        white = np.full((colors_stuff.shape[0], 3), 255, dtype='float')
        colors_stuff = colors_stuff * alpha_stuff + white * (1 - alpha_stuff)
        colors_stuff = colors_stuff.astype('int64')

        # Compute mask for stuff points
        stuff_classes = np.asarray([i for i in stuff_classes if i <= y.max()])
        is_stuff = np.zeros(y.max() + 1, dtype='bool')
        for i in stuff_classes:
            if i < is_stuff.shape[0]:
                is_stuff[i] = True
        is_stuff = is_stuff[y]

        # Merge thing and stuff colors and text
        colors = colors_thing
        text = text_thing
        colors[is_stuff] = colors_stuff[is_stuff]
        text[is_stuff] = text_stuff[is_stuff]
        data_0.obj_pred_colors = colors

        # Create trace modes
        trace_modes[i_point_trace]['Panoptic Pred.'] = {
            'marker.color': colors[data_0.selected],
            'hovertext': text[data_0.selected]}
        trace_modes[i_unselected_point_trace]['Panoptic Pred.'] = {
            'marker.color': colors[~data_0.selected],
            'hovertext': text[~data_0.selected]}

    # Draw a trace for 3D point cloud features
    if data_0.x is not None:
        colors = feats_to_plotly_rgb(
            data_0.x, normalize=True, colorscale=colorscale)
        data_0.x_colors = colors
        trace_modes[i_point_trace]['Features 3D'] = {
            'marker.color': colors[data_0.selected], 'hovertext': None}
        trace_modes[i_unselected_point_trace]['Features 3D'] = {
            'marker.color': colors[~data_0.selected], 'hovertext': None}

    # Draw a trace for each key specified in keys
    if keys is None:
        keys = []
    elif isinstance(keys, str):
        keys = [keys]
    for key in keys:
        if getattr(data_0, key, None) is not None:
            colors = feats_to_plotly_rgb(
                data_0[key], normalize=True, colorscale=colorscale)
            data_0[f"{key}_colors"] = colors
            trace_modes[i_point_trace][str(key).title()] = {
                'marker.color': colors[data_0.selected], 'hovertext': None}
            trace_modes[i_unselected_point_trace][str(key).title()] = {
                'marker.color': colors[~data_0.selected], 'hovertext': None}

    # Draw a trace for 3D point cloud sampling (for sampling debugging)
    if 'super_sampling' in data_0.keys:
        colors = data_0.super_sampling
        colors = int_to_plotly_rgb(colors)
        colors[data_0.super_sampling == -1] = 230
        data_0.super_sampling_colors = colors
        trace_modes[i_point_trace]['Super sampling'] = {
            'marker.color': colors[data_0.selected], 'hovertext': None}
        trace_modes[i_unselected_point_trace]['Super sampling'] = {
            'marker.color': colors[~data_0.selected], 'hovertext': None}

    # Draw a trace for each cluster level
    for i_level, data_i in enumerate(input if is_nag else []):

        # Exit in case the Data has no 'super_index'
        if not data_i.is_sub:
            break

        # 'Data.super_index' are expressed between levels i and i+1, but
        # we need to recover the 'super_index' between level 0 and i+1,
        # to draw clusters on the level-0 points. To this end, we
        # compute the desired 'super_index' iteratively, with a
        # bottom-up approach
        if i_level == 0:
            super_index = data_i.super_index
        else:
            super_index = data_i.super_index[super_index]

        # Note that we update the 'trace_modes' 0th element here, this
        # assumes only it is the trace holding all level-0 points and on
        # which all other colors modes are defined
        colors = int_to_plotly_rgb(super_index)
        data_0[f"{i_level}_level_colors"] = colors
        text = np.array([f"↑: {i}" for i in super_index])
        trace_modes[i_point_trace][f"Level {i_level + 1}"] = {
            'marker.color': colors[data_0.selected],
            'hovertext': text[data_0.selected]}
        trace_modes[i_unselected_point_trace][f"Level {i_level + 1}"] = {
            'marker.color': colors[~data_0.selected],
            'hovertext': text[~data_0.selected]}

        # Skip to the next level if we do not need to draw the cluster
        # centroids
        if not centroids:
            continue

        # To recover centroids of the i+1 level superpoints, we either
        # read them from the next NAG level or compute them using the
        # level i 'super_index' indices
        num_levels = input.num_levels
        is_last_level = i_level == num_levels - 1
        if is_last_level or input[i_level + 1].pos is None:
            super_pos = scatter_mean(data_0.pos, super_index, dim=0)
        else:
            super_pos = input[i_level + 1].pos

        # Add the gap offset, if need be
        if gap is not None:

            super_pos += gap * (i_level + 1)

        # Round to the cm for cleaner hover info
        super_pos = (super_pos * 100).round() / 100

        # Save the drawing position of centroids to facilitate vertical
        # edges drawing later on
        input[i_level + 1].draw_pos = super_pos

        # Draw the level-i+1 cluster centroids
        idx_sp = torch.arange(data_i.super_index.max() + 1)
        colors = int_to_plotly_rgb(idx_sp)
        text = np.array([f"<b>#: {i}</b>" for i in idx_sp])
        ball_size = centroid_size if centroid_size else point_size * 3

        fig.add_trace(
            go.Scatter3d(
                x=super_pos[input[i_level + 1].selected, 0],
                y=super_pos[input[i_level + 1].selected, 1],
                z=super_pos[input[i_level + 1].selected, 2],
                mode='markers+text',
                marker=dict(
                    symbol=centroid_symbol,
                    size=ball_size,
                    color=colors[input[i_level + 1].selected.numpy()],
                    line_width=min(ball_size / 2, 2),
                    line_color='black'),
                textposition="bottom center",
                textfont=dict(size=16),
                hovertext=text,
                hoverinfo='x+y+z+text',
                showlegend=False,
                visible=gap is not None, ))

        fig.add_trace(
            go.Scatter3d(
                x=super_pos[~input[i_level + 1].selected, 0],
                y=super_pos[~input[i_level + 1].selected, 1],
                z=super_pos[~input[i_level + 1].selected, 2],
                mode='markers+text',
                marker=dict(
                    symbol=centroid_symbol,
                    size=ball_size,
                    color=colors[~input[i_level + 1].selected.numpy()],
                    line_width=min(ball_size / 2, 2),
                    line_color='black',
                    opacity=alpha_super),
                textposition="bottom center",
                textfont=dict(size=16),
                hovertext=text,
                hoverinfo='x+y+z+text',
                showlegend=False,
                visible=gap is not None, ))

        keys = [f"Level {i_level + 1}"] if gap is None \
            else trace_modes[i_point_trace].keys()
        trace_modes.append(
            {k: {
                'marker.color': colors[input[i_level + 1].selected.numpy()],
                'hovertext': text[input[i_level + 1].selected.numpy()]}
            for k in keys})
        trace_modes.append(
            {k: {
                'marker.color': colors[~input[i_level + 1].selected.numpy()],
                'hovertext': text[~input[i_level + 1].selected.numpy()]}
            for k in keys})

        if i_level > 0 and v_edge and gap is not None and is_nag:
            # Recover the source and target positions for vertical edges
            # between i_level -> i_level+1
            low_pos = data_i.draw_pos[data_i.selected]
            high_pos = super_pos[data_i.super_index[data_i.selected]]

            # Convert into a plotly-friendly format for 3D lines
            edges = np.full((low_pos.shape[0] * 3, 3), None)
            edges[::3] = low_pos
            edges[1::3] = high_pos

            # Color the vertical edges based on the parent cluster index
            # Plotly is a bit hacky with colors for 3D lines. We cannot
            # directly pass individual edge colors, we must instead give
            # edge color as an int corresponding to a colorscale list
            # holding plotly-friendly colors
            colors = data_i.super_index[data_i.selected]
            colors = np.repeat(colors, 3)
            n_colors = colors.max().item() + 1
            edge_colorscale = int_to_plotly_rgb(torch.arange(n_colors))
            edge_colorscale = [
                [i / (n_colors - 1), f"rgb({x[0]}, {x[1]}, {x[2]})"]
                for i, x in enumerate(edge_colorscale)]

            # Since plotly 3D lines do not support opacity, we draw
            # these edges as super thin to limit clutter
            edge_width = 0.5 if v_edge_width is None else v_edge_width

            # Draw the level i -> i+1 vertical edges. NB we only draw
            # edges that are selected and do not draw the unselected
            # edges. This is because plotly does not handle opacity
            # on lines (yet), which means the unselected edges will tend
            # to clutter the figure. For this reason we choose to simply
            # not show them
            fig.add_trace(
                go.Scatter3d(
                    x=edges[:, 0],
                    y=edges[:, 1],
                    z=edges[:, 2],
                    mode='lines',
                    line=dict(
                        width=edge_width,
                        color=colors,
                        colorscale=edge_colorscale),
                    hoverinfo='skip',
                    showlegend=False,
                    visible=gap is not None, ))

            # NB: at this point, trace_modes contains 'Level i+1' as its
            # last key, but we do not want vertical edges to be seen
            # when 'Level i+1' is selected, because it means 'Level i'
            # nodes are hidden
            keys = list(trace_modes[i_point_trace].keys())[:-1]
            trace_modes.append({k: {} for k in keys})

        # Do not draw superedges if not required or if the i+1 level
        # does not have any
        if not h_edge or is_last_level or not input[i_level + 1].has_edges:
            continue

        # Recover the superedge source and target positions
        se = input[i_level + 1].edge_index
        se_attr = input[i_level + 1].edge_attr

        # Since we can only draw one edge direction (they would overlap
        # otherwise), we can trim the graph to only keep one direction
        # for each undirected edge pair. However, this requires picking
        # one direction for the edge attributes to we ARBITRARILY TAKE
        # THE MAX EDGE FEATURE for each undirected edge
        input[i_level + 1].raise_if_edge_keys()
        if se_attr is not None:
            se, se_attr = to_trimmed(se, edge_attr=se_attr, reduce='max')
        else:
            se = to_trimmed(se)

        # Recover corresponding source and target coordinates using the
        # previously-computed 'super_pos' cluster centroid positions
        s_pos = super_pos[se[0]].numpy()
        t_pos = super_pos[se[1]].numpy()

        # Convert into a plotly-friendly format for 3D lines
        edges = np.full((se.shape[1] * 3, 3), None)
        edges[::3] = s_pos
        edges[1::3] = t_pos

        if h_edge_attr and se_attr is not None:

            # Recover edge features and convert them to RGB colors. NB:
            # edge features are assumed to be in [0, 1] or [-1, 1].
            # Since we only draw edges in one direction, we choose to
            # only represent the absolute value of the features. This
            # implies that features are either direction-independent or
            # that the edge direction only changes the sign of the
            # feature
            colors = feats_to_plotly_rgb(
                se_attr.abs(), normalize=True, colorscale=colorscale)
            colors = np.repeat(colors, 3, axis=0)
            edge_width = point_size if h_edge_width is None else h_edge_width

        else:
            colors = feats_to_plotly_rgb(
                torch.zeros(edges.shape[0]), normalize=True, colorscale=colorscale)
            edge_width = point_size if h_edge_width is None else h_edge_width

        selected_edge = input[i_level + 1].selected[se].all(axis=0)
        selected_edge = selected_edge.repeat_interleave(3).numpy()

        # Draw the level-i+1 superedges. NB we only draw edges that are
        # selected and do not draw the unselected edges. This is because
        # plotly does not handle opacity on lines (yet), which means the
        # unselected edges will tend to clutter the figure. For this
        # reason we choose to simply not show them
        fig.add_trace(
            go.Scatter3d(
                x=edges[selected_edge, 0],
                y=edges[selected_edge, 1],
                z=edges[selected_edge, 2],
                mode='lines',
                line=dict(
                    width=edge_width,
                    color=colors[selected_edge]),
                hoverinfo='skip',
                showlegend=False,
                visible=gap is not None, ))

        keys = [f"Level {i_level + 1}"] if gap is None \
            else trace_modes[i_point_trace].keys()
        trace_modes.append({k: {} for k in keys})

    # Add a trace for prediction errors. NB: it is important that this
    # trace is created last, as the button behavior for this one is
    # particular
    has_error = data_0.y is not None and data_0.semantic_pred is not None
    if has_error:

        # Recover prediction and ground truth and deal with potential
        # histograms
        y = data_0.y
        y = y.argmax(1).numpy() if y.dim() == 2 else y.numpy()
        pred = data_0.semantic_pred
        pred = pred.argmax(1).numpy() if pred.dim() == 2 else pred.numpy()

        # Identify erroneous point indices
        ignore = void_classes if void_classes else []
        ignore = ignore + [-1]
        indices = np.where((pred != y) & (~np.in1d(y, ignore)))[0]

        # Prepare the color for erroneous points
        error_color = 'red' if error_color is None \
            else np.asarray[error_color].squeeze()

        # Draw the erroneous points
        fig.add_trace(
            go.Scatter3d(
                x=data_0.pos[indices, 0],
                y=data_0.pos[indices, 1],
                z=data_0.pos[indices, 2],
                mode='markers',
                marker=dict(
                    symbol=point_symbol,
                    size=int(point_size * 1.5),
                    color=error_color, ),
                showlegend=False,
                visible=False, ))

    # Recover the keys for all visualization modes, as an ordered set,
    # with respect to their order of first appearance
    modes = list(dict.fromkeys([k for m in trace_modes for k in m.keys()]))

    # Traces color for interactive point cloud coloring
    def trace_update(mode):
        # Prepare the output args for the figure update attributes. By
        # default, all traces are non visible, with no color and no
        # hover text
        n_traces = len(trace_modes)
        out = {
            'visible': [False] * (n_traces + has_error),
            'marker.color': [None] * n_traces,
            'hovertext': [''] * n_traces}

        # For each trace in 'trace_modes' see if it contains 'mode' and
        # adapt out accordingly
        for i_trace, t_modes in enumerate(trace_modes):

            # The trace has no action for the mode, skip it and leave
            # the default args for the trace
            if mode not in t_modes:
                continue

            # Note that a trace will only be visible for its modes
            # declared in trace_modes
            out['visible'][i_trace] = True
            for key, val in t_modes[mode].items():
                out[key][i_trace] = val

        return [out, list(range(len(trace_modes)))]

    # Create the buttons that will serve for toggling trace visibility
    updatemenus = [
        dict(
            buttons=[dict(
                label=mode, method='update', args=trace_update(mode))
                for mode in modes if mode.lower() != 'errors'],
            pad={'r': 10, 't': 10},
            showactive=True,
            type='dropdown',
            direction='right',
            xanchor='left',
            x=0.02,
            yanchor='top',
            y=1.02, ),]

    if has_error:
        updatemenus.append(
            dict(
                buttons=[dict(
                    method='restyle',
                    label='Semantic Errors',
                    visible=True,
                    args=[
                        {'visible': True, 'marker.color': error_color},
                        [len(trace_modes)]],
                    args2=[
                        {'visible': False,},
                        [len(trace_modes)]],)],
                pad={'r': 10, 't': 10},
                showactive=False,
                type='buttons',
                xanchor='left',
                x=1.02,
                yanchor='top',
                y=1.02, ),)

    fig.update_layout(updatemenus=updatemenus)

    # Place the legend on the left
    fig.update_layout(
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=0.99))

    # Hide all axes and no background
    fig.update_layout(
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title='',
            xaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)"),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)"),
            zaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False,
                backgroundcolor="rgba(0, 0, 0, 0)")))

    output = {'figure': fig, 'data': data_0}

    return output


def figure_html(fig):
    # Save plotly figure to temp HTML
    fig.write_html(
        '/tmp/fig.html',
        config={'displayModeBar': False},
        include_plotlyjs='cdn',
        full_html=False)

    # Read the HTML
    with open("/tmp/fig.html", "r") as f:
        fig_html = f.read()

    # Center the figure div for cleaner display
    fig_html = fig_html.replace(
        'class="plotly-graph-div" style="',
        'class="plotly-graph-div" style="margin:0 auto;')

    return fig_html


def show(
        input, path=None, title=None, no_output=True, pt_path=None, **kwargs):
    """Interactive data visualization.

    :param input: Data or NAG object
    :param path: str
        Path to save the visualization into a sharable HTML
    :param title: str
        Figure title
    :param no_output: bool
        Set to True if you want to return the 3D Plotly figure objects
    :param kwargs:
    :return:
    """
    # Sanitize title and path
    if title is None:
        title = "Large-scale point cloud"
    if path is not None:
        if osp.isdir(path):
            path = osp.join(path, f"{title}.html")
        else:
            path = osp.splitext(path)[0] + '.html'
        fig_html = f'<h1 style="text-align: center;">{title}</h1>'

    # Draw a figure for 3D data visualization
    out_3d = visualize_3d(input, **kwargs)
    if no_output:
        if path is None:
            out_3d['figure'].show(config={'displayModeBar': False})
        else:
            fig_html += figure_html(out_3d['figure'])

    if path is not None:
        with open(path, "w") as f:
            f.write(fig_html)

    # Save to a .pt file for other downstream tasks.
    # NB: we only save the 'pos' and all data attributes containing
    # 'color', the rest is discarded
    # NB: we save a dictionary, to limit dependencies
    if pt_path is not None:
        if osp.isdir(pt_path):
            pt_path = osp.join(pt_path, f"viz_data.pt")
        else:
            pt_path = osp.splitext(pt_path)[0] + '.pt'

        data = {}
        for key in out_3d['data'].keys:
            if key == 'pos' or 'color' in key:
                data[key] = out_3d['data'][key]

        torch.save(data, pt_path)

    if not no_output:
        return out_3d

    return
