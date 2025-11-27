import torch
from torch import nn
from typing import Any, List, Tuple, Dict, Union
from src.data import Data, NAG
from src.utils import listify_with_reference, VersionHolder
from src.nn import Stage, PointStage, DownNFuseStage, UpNFuseStage, \
    BatchNorm, CatFusion, MLP, LayerNorm
from src.nn.pool import BaseAttentivePool
from src.nn.pool import pool_factory

__all__ = ['SPT']


class SPT(nn.Module):
    """Superpoint Transformer. A UNet-like architecture processing NAG.

    The architecture can be (roughly) summarized as:

    p_0, x_0 --------- PointStage
                           \
    p_1, x_1, e_1 -- DownNFuseStage_1 ------- UpNFuseStage_1 --> out_1
                            \                       |
    p_2, x_2, e_2 -- DownNFuseStage_2 ------- UpNFuseStage_2 --> out_2
                            \                       |
                           ...                     ...

    Where:
    - p_0: point positions
    - x_0: input point features (handcrafted)
    - p_i: node positions (i.e. superpoint centroid) at level i
    - x_i: input node features (handcrafted superpoint features) at
      level i
    - e_i: input edge features (handcrafted horizontal superpoint graph
      edge features) at level i
    - out_i: node-wise output features at level i


    :param point_mlp: List[int]
        Channels for the input MLP of the `PointStage`
    :param point_drop: float
        Dropout rate for the last layer of the input and output MLPs
        in `PointStage`
        
    :point_cnn_blocks: bool
        Whether to use a sparse CNN before the in_MLP in the `PointStage`
    :point_cnn List[int]
        List of channels for the sparse CNN in the `PointStage`
    :point_cnn_kernel_size: int
        Kernel size for the sparse CNN in the `PointStage`
    :point_cnn_dilation: int
        Dilation for the sparse CNN in the `PointStage`
    
    :point_cnn_norm: nn.Module
        Normalization function for the sparse CNN in the `PointStage`
    :point_cnn_activation: nn.Module
        Activation function for the sparse CNN in the `PointStage`
    :point_cnn_residual: bool
        Whether to use a pre-activation residual connection 
        in each block of the CNN in the `PointStage`
    :point_cnn_global_residual: bool
        Whether to use a global residual connection between the 
        input and the output of the CNN in the `PointStage`

    :param point_mlp_on_cnn_feats: bool
        Whether to use the CNN features as input to the 
        MLP (see scheme in the `PointStage` docstring)

    :param nano: bool
        If True, the `PointStage` will be removed and the model will
        only operate on superpoints, without extracting features
        from the points. This lightweight model saves compute and
        memory, at the potential expense of high-resolution
        reasoning

    :param down_dim: List[int], int
        Feature dimension for each `DownNFuseStage` (i.e. not
        including the `PointStage` when `nano=False`)
    :param down_pool_dim: List[str], str
        Dimension after pooling in each `DownNFuseStage`.
        See `pool_factory()` for more
    :param down_in_mlp: List[List[int]], List[int]
        Channels for the input MLP of each `DownNFuseStage`
    :param down_out_mlp: List[List[int]], List[int]
        Channels for the output MLP of each `DownNFuseStage`. The
        first channel for each stage must match with what is passed
        in `down_dim`
    :param down_mlp_drop: List[float], float
        Dropout rate for the last layer of the input and output MLPs
        of each `DownNFuseStage`
    :param down_num_heads: List[int], int
        Number of self-attention heads for each `DownNFuseStage
    :param down_num_blocks: List[int], int
        Number of self-attention blocks for each `DownNFuseStage
    :param down_ffn_ratio: List[float], float
        Multiplicative factor for computing the dimension of the
        `FFN` inverted bottleneck, for each `DownNFuseStage. See
        `TransformerBlock`
    :param down_residual_drop: List[float], float
        Dropout on the output self-attention features for each
        `DownNFuseStage`. See `TransformerBlock`
    :param down_attn_drop: List[float], float
        Dropout on the self-attention weights for each
        `DownNFuseStage`. See `TransformerBlock`
    :param down_drop_path: List[float], float
        Dropout on the residual paths for each `DownNFuseStage`. See
        `TransformerBlock`

    :param up_dim: List[int], int
        Feature dimension for each `UpNFuseStage`
    :param up_in_mlp: List[List[int]], List[int]
        Channels for the input MLP of each `UpNFuseStage`
    :param up_out_mlp: List[List[int]], List[int]
        Channels for the output MLP of each `UpNFuseStage`. The
        first channel for each stage must match with what is passed
        in `up_dim`
    :param up_mlp_drop: List[float], float
        Dropout rate for the last layer of the input and output MLPs
        of each `UpNFuseStage`
    :param up_num_heads: List[int], int
        Number of self-attention heads for each `UpNFuseStage
    :param up_num_blocks: List[int], int
        Number of self-attention blocks for each `UpNFuseStage
    :param up_ffn_ratio: List[float], float
        Multiplicative factor for computing the dimension of the
        `FFN` inverted bottleneck, for each `UpNFuseStage. See
        `TransformerBlock`
    :param up_residual_drop: List[float], float
        Dropout on the output self-attention features for each
        `UpNFuseStage`. See `TransformerBlock`
    :param up_attn_drop: List[float], float
        Dropout on the self-attention weights for each
        `UpNFuseStage`. See `TransformerBlock`
    :param up_drop_path: List[float], float
        Dropout on the residual paths for each `UpNFuseStage`. See
        `TransformerBlock`

    :param node_mlp: List[int]
        Channels for the MLPs that will encode handcrafted node
        (i.e. segment, superpoint) features. These will be called
        before each `DownNFuseStage` and their output will be
        concatenated to any already-existing features and passed
        to `DownNFuseStage` and `UpNFuseStage`. For the special case
        the `nano=True` model, the first MLP will be run before the
        first `Stage` too
    :param h_edge_mlp: List[int]
        Channels for the MLPs that will encode handcrafted
        horizontal edge (i.e. edges in the superpoint adjacency
        graph at each partition level) features. These will be
        called before each `DownNFuseStage` and their output will be
        passed as `edge_attr` to `DownNFuseStage` and `UpNFuseStage`
    :param v_edge_mlp: List[int]
        Channels for the MLPs that will encode handcrafted
        vertical edge (i.e. edges connecting nodes to their parent
        in the above partition level) features. These will be
        called before each `DownNFuseStage` and their output will be
        passed as `v_edge_attr` to `DownNFuseStage` and
        `UpNFuseStage`
    :param mlp_activation: nn.Module
        Activation function used for all MLPs throughout the
        architecture
    :param mlp_norm: n.Module
        Normalization function for all MLPs throughout the
        architecture
    :param qk_dim: int
        Dimension of the queries and keys. See `SelfAttentionBlock`
    :param qkv_bias: bool
        Whether the linear layers producing queries, keys, and
        values should have a bias. See `SelfAttentionBlock`
    :param qk_scale: str
        Scaling applied to the query*key product before the softmax.
        More specifically, one may want to normalize the query-key
        compatibilities based on the number of dimensions (referred
        to as 'd' here) as in a vanilla Transformer implementation,
        or based on the number of neighbors each node has in the
        attention graph (referred to as 'g' here). If nothing is
        specified the scaling will be `1 / (sqrt(d) * sqrt(g))`,
        which is equivalent to passing `'d.g'`. Passing `'d+g'` will
        yield `1 / (sqrt(d) + sqrt(g))`. Meanwhile, passing 'd' will
        yield `1 / sqrt(d)`, and passing `'g'` will yield
        `1 / sqrt(g)`. See `SelfAttentionBlock`
    :param in_rpe_dim:
    :param activation: nn.Module
        Activation function used in the `FFN` modules. See
        `TransformerBlock`
    :param norm: nn.Module
        Normalization function for the `FFN` module. See
        `TransformerBlock`
    :param pre_norm: bool
        Whether the normalization should be applied before or after
        the `SelfAttentionBlock` and `FFN` in the residual branches.
        See`TransformerBlock`
    :param no_sa: bool
        Whether a self-attention residual branch should be used at
        all. See`TransformerBlock`
    :param no_ffn: bool
        Whether a feed-forward residual branch should be used at
        all. See`TransformerBlock`
    :param k_rpe: bool
        Whether keys should receive relative positional encodings
        computed from edge features. See `SelfAttentionBlock`
    :param q_rpe: bool
        Whether queries should receive relative positional encodings
        computed from edge features. See `SelfAttentionBlock`
    :param v_rpe: bool
        Whether values should receive relative positional encodings
        computed from edge features. See `SelfAttentionBlock`
    :param k_delta_rpe: bool
        Whether keys should receive relative positional encodings
        computed from the difference between source and target node
        features. See `SelfAttentionBlock`
    :param q_delta_rpe: bool
        Whether queries should receive relative positional encodings
        computed from the difference between source and target node
        features. See `SelfAttentionBlock`
    :param qk_share_rpe: bool
        Whether queries and keys should use the same parameters for
        building relative positional encodings. See
        `SelfAttentionBlock`
    :param q_on_minus_rpe: bool
        Whether relative positional encodings for queries should be
        computed on the opposite of features used for keys. This allows,
        for instance, to break the symmetry when `qk_share_rpe` but we
        want relative positional encodings to capture different meanings
        for keys and queries. See `SelfAttentionBlock`
    :param share_hf_mlps: bool
        Whether stages should share the MLPs for encoding
        handcrafted node, horizontal edge, and vertical edge
        features
    :param stages_share_rpe: bool
        Whether all `Stage`s should share the same parameters for
        building relative positional encodings
    :param blocks_share_rpe: bool
        Whether all the `TransformerBlock` in the same `Stage`
        should share the same parameters for building relative
        positional encodings
    :param heads_share_rpe: bool
        Whether attention heads should share the same parameters for
        building relative positional encodings

    :param use_pos: bool
        Whether the node's position (normalized with `UnitSphereNorm`)
        should be concatenated to the features. See `Stage`
    :param use_node_hf: bool
        Whether handcrafted node (i.e. segment, superpoint) features
        should be used at all. If False, `node_mlp` will be ignored
    :param use_diameter: bool
        Whether the node's diameter (currently estimated with
        `UnitSphereNorm`) should be concatenated to the node features.
        See `Stage`
    :param use_diameter_parent: bool
        Whether the node's parent diameter (currently estimated with
        `UnitSphereNorm`) should be concatenated to the node features.
        See `Stage`
    :param pool: List[str], str, nn.Module
        Pooling mechanism for `DownNFuseStage`s. Supports 'max',
        'min', 'mean', 'sum', and 'std' for string arguments.
        See `pool_factory()` for more
    :param unpool: str
        Unpooling mechanism for `UpNFuseStage`s. Only supports
        'index' for now
    :param fusion: str
        Fusion mechanism used in `DownNFuseStage` and `UpNFuseStage`
        to merge node features from different branches. Supports
        'cat', 'residual', 'first', 'second'. See `fusion_factory()`
        for more
    :param norm_mode: str
        Indexing mode used for feature normalization. This will be
        passed to `Data.norm_index()`. 'graph' will normalize
        features per graph (i.e. per cloud, i.e. per batch item).
        'node' will normalize per node (i.e. per point). 'segment'
        will normalize per segment (i.e.  per cluster)
    :param output_stage_wise: bool
        If True, the output contain the features for each node of
        each partition 1+ level. IF False, only the features for the
        partition level 1 will be returned. Note we do not compute
        the features for level 0, since the entire goal of this
        superpoint-based reasoning is to mitigate compute and memory
        by circumventing the need to manipulate such full-resolution
        objects
    :param store_features: bool
        Visualization purposes: whether to store the computed features of
        the superpoints (level 1) in the output object. (These are the features 
        that the classifier head  takes as input.) 
        It also stores the handcrafted features of points and superpoints 
        in the NAG (otherwise they are deleted during forward).
    """

    def __init__(
            self,

            point_hf=[],
            post_cnn_point_hf=[],
            segment_hf=[],

            point_mlp=None,
            point_drop=None,

            point_cnn_blocks=False,
            point_cnn=None,
            point_cnn_kernel_size=None,
            point_cnn_dilation=None,

            point_cnn_norm=None,
            point_cnn_activation=None,
            point_cnn_residual=False,
            point_cnn_global_residual=False,

            point_mlp_on_cnn_feats=False,
            
            nano=False,

            down_dim=None,
            down_pool_dim=None,
            down_in_mlp=None,
            down_out_mlp=None,
            down_mlp_drop=None,
            down_num_heads=1,
            down_num_blocks=0,
            down_ffn_ratio=4,
            down_residual_drop=None,
            down_attn_drop=None,
            down_drop_path=None,

            up_dim=None,
            up_in_mlp=None,
            up_out_mlp=None,
            up_mlp_drop=None,
            up_num_heads=1,
            up_num_blocks=0,
            up_ffn_ratio=4,
            up_residual_drop=None,
            up_attn_drop=None,
            up_drop_path=None,

            node_mlp=None,
            h_edge_mlp=None,
            v_edge_mlp=None,
            mlp_activation=nn.LeakyReLU(),
            mlp_norm=BatchNorm,
            qk_dim=8,
            qkv_bias=True,
            qk_scale=None,
            in_rpe_dim=18,
            activation=nn.LeakyReLU(),
            norm=LayerNorm,
            pre_norm=True,
            no_sa=False,
            no_ffn=False,
            k_rpe=False,
            q_rpe=False,
            v_rpe=False,
            k_delta_rpe=False,
            q_delta_rpe=False,
            qk_share_rpe=False,
            q_on_minus_rpe=False,
            share_hf_mlps=False,
            stages_share_rpe=False,
            blocks_share_rpe=False,
            heads_share_rpe=False,

            use_pos=True,
            use_node_hf=True,
            use_diameter=False,
            use_diameter_parent=False,
            pool='max',
            unpool='index',
            fusion='cat',
            norm_mode='graph',
            output_stage_wise=False,
            
            store_features=False):
        super().__init__()

        self.nano = nano
        self.use_pos = use_pos
        self.use_node_hf = use_node_hf
        self.use_diameter = use_diameter
        self.use_diameter_parent = use_diameter_parent
        self.norm_mode = norm_mode
        self.stages_share_rpe = stages_share_rpe
        self.blocks_share_rpe = blocks_share_rpe
        self.heads_share_rpe = heads_share_rpe
        self.output_stage_wise = output_stage_wise
        self.store_features = store_features
        
        self.share_hf_mlps = share_hf_mlps
        
        self.point_mlp_on_cnn_feats = point_mlp_on_cnn_feats
        
        self.point_hf = point_hf
        self.segment_hf = segment_hf
        self.post_cnn_point_hf = post_cnn_point_hf

        # Create version holder
        from src import __version__ # imported here to avoid circular import
        self.version_holder = VersionHolder(__version__)

        # Convert input arguments to nested lists
        (
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path,
            pool
        ) = listify_with_reference(
            down_dim,
            down_pool_dim,
            down_in_mlp,
            down_out_mlp,
            down_mlp_drop,
            down_num_heads,
            down_num_blocks,
            down_ffn_ratio,
            down_residual_drop,
            down_attn_drop,
            down_drop_path,
            pool)

        (
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path
        ) = listify_with_reference(
            up_dim,
            up_in_mlp,
            up_out_mlp,
            up_mlp_drop,
            up_num_heads,
            up_num_blocks,
            up_ffn_ratio,
            up_residual_drop,
            up_attn_drop,
            up_drop_path)

        # Local helper variables describing the architecture
        num_down = len(down_dim) - self.nano
        num_up = len(up_dim)
        needs_h_edge_hf = any(x > 0 for x in down_num_blocks + up_num_blocks)
        needs_v_edge_hf = num_down > 0 and isinstance(
            pool_factory(pool[0], down_pool_dim[0]), BaseAttentivePool)

        # Build MLPs that will be used to process handcrafted segment
        # and edge features. These will be called before each
        # DownNFuseStage and their output will be passed to
        # DownNFuseStage and UpNFuseStage. For the special case of nano
        # models, the first mlps will be run before the first Stage too
        node_mlp = node_mlp if use_node_hf else None
        self.node_mlps = _build_mlps(
            node_mlp,
            num_down + self.nano,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        h_edge_mlp = h_edge_mlp if needs_h_edge_hf else None
        self.h_edge_mlps = _build_mlps(
            h_edge_mlp,
            num_down + self.nano,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        v_edge_mlp = v_edge_mlp if needs_v_edge_hf else None
        self.v_edge_mlps = _build_mlps(
            v_edge_mlp,
            num_down,
            mlp_activation,
            mlp_norm,
            share_hf_mlps)

        # Module operating on Level-0 points in isolation
        if self.nano:
            self.first_stage = Stage(
                down_dim[0],
                num_blocks=down_num_blocks[0],
                in_mlp=down_in_mlp[0],
                out_mlp=down_out_mlp[0],
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=down_mlp_drop[0],
                num_heads=down_num_heads[0],
                qk_dim=qk_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                ffn_ratio=down_ffn_ratio[0],
                residual_drop=down_residual_drop[0],
                attn_drop=down_attn_drop[0],
                drop_path=down_drop_path[0],
                activation=activation,
                norm=norm,
                pre_norm=pre_norm,
                no_sa=no_sa,
                no_ffn=no_ffn,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                use_pos=use_pos,
                use_diameter=use_diameter,
                use_diameter_parent=use_diameter_parent,
                blocks_share_rpe=blocks_share_rpe,
                heads_share_rpe=heads_share_rpe,
                version_holder=self.version_holder)
        else:
            self.first_stage = PointStage(
                point_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=point_drop,
                use_pos=use_pos,
                use_diameter_parent=use_diameter_parent,
                
                cnn_blocks=point_cnn_blocks,
                cnn=point_cnn,
                cnn_kernel_size=point_cnn_kernel_size,
                cnn_dilation=point_cnn_dilation,
                cnn_norm=point_cnn_norm,
                cnn_activation=point_cnn_activation,
                cnn_residual=point_cnn_residual,
                cnn_global_residual=point_cnn_global_residual,
                
                point_mlp_on_cnn_feats= point_mlp_on_cnn_feats,
                version_holder=self.version_holder)

        # Operator to append the features such as the diameter or other 
        # handcrafted features to the NAG's features
        self.feature_fusion = CatFusion()

        # Transformer encoder (down) Stages operating on Level-i data
        if num_down > 0:

            # Build the RPE encoders here if shared across all stages
            down_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_down, 18, qk_dim, stages_share_rpe)

            # If key and query RPEs share the same MLP, only the key MLP
            # is preserved, to limit the number of model parameters
            down_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), num_down, 18, qk_dim,
                stages_share_rpe)

            # Since the first value of each down_ parameter is used for
            # the nano Stage (if self.nano=True), we artificially
            # prepend None values to the rpe lists, so they have the
            # same length as other down_ parameters
            if self.nano:
                down_k_rpe = [None] + down_k_rpe
                down_q_rpe = [None] + down_q_rpe

            self.down_stages = nn.ModuleList([
                DownNFuseStage(
                    dim,
                    num_blocks=num_blocks,
                    in_mlp=in_mlp,
                    out_mlp=out_mlp,
                    mlp_activation=mlp_activation,
                    mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    in_rpe_dim=in_rpe_dim,
                    ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    activation=activation,
                    norm=norm,
                    pre_norm=pre_norm,
                    no_sa=no_sa,
                    no_ffn=no_ffn,
                    k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe,
                    v_rpe=v_rpe,
                    k_delta_rpe=k_delta_rpe,
                    q_delta_rpe=q_delta_rpe,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    pool=pool_factory(_pool, pool_dim),
                    fusion=fusion,
                    use_pos=use_pos,
                    use_diameter=use_diameter,
                    use_diameter_parent=use_diameter_parent,
                    blocks_share_rpe=blocks_share_rpe,
                    heads_share_rpe=heads_share_rpe,
                    version_holder=self.version_holder)
                for
                    i_down,
                    (dim,
                    num_blocks,
                    in_mlp,
                    out_mlp,
                    mlp_drop,
                    num_heads,
                    ffn_ratio,
                    residual_drop,
                    attn_drop,
                    drop_path,
                    stage_k_rpe,
                    stage_q_rpe,
                    pool_dim,
                    _pool)
                in enumerate(zip(
                    down_dim,
                    down_num_blocks,
                    down_in_mlp,
                    down_out_mlp,
                    down_mlp_drop,
                    down_num_heads,
                    down_ffn_ratio,
                    down_residual_drop,
                    down_attn_drop,
                    down_drop_path,
                    down_k_rpe,
                    down_q_rpe,
                    down_pool_dim,
                    pool))
                if i_down >= self.nano])
        else:
            self.down_stages = None

        # Transformer decoder (up) Stages operating on Level-i data
        if num_up > 0:

            # Build the RPE encoder here if shared across all stages
            up_k_rpe = _build_shared_rpe_encoders(
                k_rpe, num_up, 18, qk_dim, stages_share_rpe)

            # If key and query RPEs share the same MLP, only the key MLP
            # is preserved, to limit the number of model parameters
            up_q_rpe = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), num_up, 18, qk_dim,
                stages_share_rpe)

            self.up_stages = nn.ModuleList([
                UpNFuseStage(
                    dim,
                    num_blocks=num_blocks,
                    in_mlp=in_mlp,
                    out_mlp=out_mlp,
                    mlp_activation=mlp_activation,
                    mlp_norm=mlp_norm,
                    mlp_drop=mlp_drop,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    in_rpe_dim=in_rpe_dim,
                    ffn_ratio=ffn_ratio,
                    residual_drop=residual_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    activation=activation,
                    norm=norm,
                    pre_norm=pre_norm,
                    no_sa=no_sa,
                    no_ffn=no_ffn,
                    k_rpe=stage_k_rpe,
                    q_rpe=stage_q_rpe,
                    v_rpe=v_rpe,
                    k_delta_rpe=k_delta_rpe,
                    q_delta_rpe=q_delta_rpe,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    unpool=unpool,
                    fusion=fusion,
                    use_pos=use_pos,
                    use_diameter=use_diameter,
                    use_diameter_parent=use_diameter_parent,
                    blocks_share_rpe=blocks_share_rpe,
                    heads_share_rpe=heads_share_rpe,
                    version_holder=self.version_holder)
                for dim,
                    num_blocks,
                    in_mlp,
                    out_mlp,
                    mlp_drop,
                    num_heads,
                    ffn_ratio,
                    residual_drop,
                    attn_drop,
                    drop_path,
                    stage_k_rpe,
                    stage_q_rpe
                in zip(
                    up_dim,
                    up_num_blocks,
                    up_in_mlp,
                    up_out_mlp,
                    up_mlp_drop,
                    up_num_heads,
                    up_ffn_ratio,
                    up_residual_drop,
                    up_attn_drop,
                    up_drop_path,
                    up_k_rpe,
                    up_q_rpe)])
        else:
            self.up_stages = None

        assert self.num_up_stages > 0 or not self.output_stage_wise, \
            "At least one up stage is needed for output_stage_wise=True"

        assert bool(self.down_stages) != bool(self.up_stages) \
               or self.num_down_stages >= self.num_up_stages, \
            "The number of Up stages should be <= the number of Down " \
            "stages."
        assert self.nano or self.num_down_stages > self.num_up_stages or self.num_down_stages == 0, \
            "The number of Up stages should be < the number of Down " \
            "stages. That is to say, we do not want to output Level-0 " \
            "features but at least Level-1."

    @property
    def num_down_stages(self):
        return len(self.down_stages) if self.down_stages is not None else 0

    @property
    def num_up_stages(self):
        return len(self.up_stages) if self.up_stages is not None else 0

    @property
    def out_dim(self):
        if self.output_stage_wise:
            out_dim = [stage.out_dim for stage in self.up_stages][::-1]
            out_dim += [self.down_stages[-1].out_dim]
            return out_dim
        if self.up_stages is not None:
            return self.up_stages[-1].out_dim
        if self.down_stages is not None:
            return self.down_stages[-1].out_dim
        return self.first_stage.out_dim

    @property
    def version(self):
        return self.version_holder.value

    @version.setter
    def version(self, v):
        self.version_holder.value = v

    def forward(self, nag):
        # assert isinstance(nag, NAG)
        # assert nag.num_levels >= 2
        # assert nag.num_levels > self.num_down_stages

        assert int(self.nano) == nag.start_i_level, \
            "`nano` mode should be consistent between the model and the data.\n"\
                "   If `nano=True`, the first NAG level should be 1.\n" \
                "   If `nano=False`, the first NAG level should be 0."

        # Get the handcrafted features
        if not self.nano:
            nag.add_keys_to(level=0, 
                            keys=self.point_hf, 
                            to='x', 
                            delete_after=False)
            
            nag.add_keys_to(level=0, 
                            keys=self.post_cnn_point_hf, 
                            to='x_mlp', 
                            delete_after=not self.store_features)
        
        nag.add_keys_to(level='1+', 
                        keys=self.segment_hf, 
                        to='x', 
                        delete_after=not self.store_features)

        # Apply the first MLPs on the handcrafted features
        if self.nano:
            if self.node_mlps is not None and self.node_mlps[0] is not None:
                norm_index = nag[1].norm_index(mode=self.norm_mode)
                nag[1].x = self.node_mlps[0](nag[1].x, batch=norm_index)
            if self.h_edge_mlps is not None:
                norm_index = nag[1].norm_index(mode=self.norm_mode)
                norm_index = norm_index[nag[1].edge_index[0]]
                nag[1].edge_attr = self.h_edge_mlps[0](
                    nag[1].edge_attr, batch=norm_index)

        # Forward the first stage
        x, diameter = self._forward_first_stage(nag)

        # Add the diameter to the next level's attributes
        start_i_level = nag.start_i_level
        nag[start_i_level + 1].diameter = diameter

        # Iteratively encode level-1 and above
        down_outputs = []
        if self.nano:
            down_outputs.append(x)
        if self.down_stages is not None:

            enum = enumerate(zip(
                self.down_stages,
                self.node_mlps[int(self.nano):],
                self.h_edge_mlps[int(self.nano):],
                self.v_edge_mlps))

            for i_stage, (stage, node_mlp, h_edge_mlp, v_edge_mlp) in enum:

                # Forward on the down stage and the corresponding NAG
                # level
                i_level = i_stage + 1 + self.nano

                # Process handcrafted node and edge features. We need to
                # do this here before those can be passed to the
                # DownNFuseStage and, later on, to the UpNFuseStage
                if node_mlp is not None:
                    norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                    nag[i_level].x = node_mlp(nag[i_level].x, batch=norm_index)
                if h_edge_mlp is not None:
                    norm_index = nag[i_level].norm_index(mode=self.norm_mode)
                    norm_index = norm_index[nag[i_level].edge_index[0]]
                    edge_attr = getattr(nag[i_level], 'edge_attr', None)
                    if edge_attr is not None:
                        nag[i_level].edge_attr = h_edge_mlp(
                            edge_attr, batch=norm_index)
                if v_edge_mlp is not None:
                    norm_index = nag[i_level - 1].norm_index(mode=self.norm_mode)
                    v_edge_attr = getattr(nag[i_level], 'v_edge_attr', None)
                    if v_edge_attr is not None:
                        nag[i_level - 1].v_edge_attr = v_edge_mlp(
                            v_edge_attr, batch=norm_index)

                # Forward on the DownNFuseStage
                x, diameter = self._forward_down_stage(stage, 
                                                       nag, 
                                                       i_level, 
                                                       x)
                down_outputs.append(x)

                # End here if we reached the last NAG level
                if i_level == nag.absolute_num_levels - 1:
                    continue

                # Add the diameter to the next level's attributes
                nag[i_level + 1].diameter = diameter

        # Iteratively decode level-num_down_stages and below
        up_outputs = []
        if self.up_stages is not None:
            for i_stage, stage in enumerate(self.up_stages):
                i_level = self.num_down_stages - i_stage - 1 + self.nano
                x_skip = down_outputs[-(2 + i_stage)] # Remark : down_outputs[-1] is equal to x 
                x, _ = self._forward_up_stage(stage, 
                                               nag, 
                                               i_level, 
                                               x, 
                                               x_skip)
                up_outputs.append(x)

        # Different types of output signatures. For stage-wise output,
        # return the output for each stage. For the Lmax level, we take
        # the output of the innermost 'down_stage'. Finally, these
        # outputs are sorted by order of increasing NAG level (from low
        # to high)
        if self.output_stage_wise:
            out = [x] + up_outputs[::-1][1:] + [down_outputs[-1]]
            return out

        return x
    
    def _forward_first_stage(self, nag : NAG):
        assert isinstance(nag, NAG)
        
        start_i_level = nag.start_i_level
        data = nag[start_i_level]
        
        return self.forward_first_stage(
            data=data,
            first_stage=self.first_stage,
            use_node_hf=self.use_node_hf,
            norm_mode=self.norm_mode)

    @staticmethod
    def forward_first_stage(data : Data,
                            first_stage,
                            use_node_hf,
                            norm_mode):
        
        assert isinstance(data, Data)

        # Forward the first stage
        return first_stage(
            data.x if use_node_hf else None,
            data.norm_index(mode=norm_mode),
            pos=data.pos,
            diameter=None,
            node_size=getattr(data, 'node_size', None),
            super_index=data.super_index,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            coords=getattr(data, 'coords', None),
            batch=data.batch,
            x_mlp=getattr(data, 'x_mlp', None))

    def _forward_down_stage(self, stage, nag, i_level, x):
        is_last_level = (i_level == nag.end_i_level)
        x_handcrafted = nag[i_level].x if self.use_node_hf else None
        return stage(
            x_handcrafted,
            x,
            nag[i_level].norm_index(mode=self.norm_mode),
            nag[i_level - 1].super_index,
            pos=nag[i_level].pos,
            diameter=nag[i_level].diameter, #TODO : WARNING : Diameter is not used in a `DownNFuseStage`
            node_size=nag[i_level].node_size,
            super_index=nag[i_level].super_index if not is_last_level else None,
            edge_index=nag[i_level].edge_index,
            edge_attr=nag[i_level].edge_attr,
            v_edge_attr=nag[i_level - 1].v_edge_attr,
            num_super=nag[i_level].num_nodes)

    def _forward_up_stage(self, stage, nag, i_level, x, x_skip):
        x_handcrafted = nag[i_level].x if self.use_node_hf else None
        return stage(
            self.feature_fusion(x_skip, x_handcrafted),
            x,
            nag[i_level].norm_index(mode=self.norm_mode),
            nag[i_level].super_index,
            pos=nag[i_level].pos,
            diameter=None, #TODO : WARNING : Diameter is not used in the `UpNFuseStage` (I removed the -self.nano, because as nag[0] is not loaded, it raises an error)
            node_size=nag[i_level].node_size,
            super_index=nag[i_level].super_index,
            edge_index=nag[i_level].edge_index,
            edge_attr=nag[i_level].edge_attr)


def _build_shared_rpe_encoders(
        rpe, num_stages, in_dim, out_dim, stages_share):
    """Local helper to build RPE encoders for spt. The main goal is to
    make shared encoders construction easier.

    Note that setting stages_share=True will make all stages, blocks and
    heads use the same RPE encoder.
    """
    if not isinstance(rpe, bool):
        assert stages_share, \
            "If anything else but a boolean is passed for the RPE encoder, " \
            "this value will be passed to all Stages and `stages_share` " \
            "should be set to True."
        return [rpe] * num_stages

    # If all stages share the same RPE encoder, all blocks and all heads
    # too. We copy the same module instance to be shared across all
    # stages and blocks
    if stages_share and rpe:
        return [nn.Linear(in_dim, out_dim)] * num_stages

    return [rpe] * num_stages


def _build_mlps(layers, num_stage, activation, norm, shared):
    if layers is None:
        return [None] * num_stage

    if shared:
        return nn.ModuleList([
            MLP(layers, activation=activation, norm=norm)] * num_stage)

    return nn.ModuleList([
        MLP(layers, activation=activation, norm=norm)
        for _ in range(num_stage)])
