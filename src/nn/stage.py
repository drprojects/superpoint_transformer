import torch
from torch import nn
from src.nn import MLP, TransformerBlock, BatchNorm, UnitSphereNorm
from src.nn.pool import pool_factory
from src.nn.unpool import *
from src.nn.fusion import CatFusion, fusion_factory


__all__ = ['Stage', 'DownNFuseStage', 'UpNFuseStage', 'PointStage']


class Stage(nn.Module):
    """A Stage has the following structure:

         x  -- PosInjection -- in_MLP -- TransformerBlock -- out_MLP -->
                   |         (optional)   (* num_blocks)   (optional)
        pos -- SphereNorm
    (optional)

    :param dim: int
        Number of channels for the TransformerBlock
    :param num_blocks: int
        Number of TransformerBlocks in the Stage
    :param num_heads: int
        Number of heads in the TransformerBlocks
    :param in_mlp: List, optional
        Channels for the input MLP. The last channel must match
        `dim`
    :param out_mlp: List, optional
        Channels for the output MLP. The first channel must match
        `dim`
    :param mlp_activation: nn.Module
        Activation function for the input and output MLPs
    :param mlp_norm: nn.Module
        Normalization for the input and output MLPs
    :param mlp_drop: float, optional
        Dropout rate for the last layer of the input and output MLPs
    :param use_pos: bool
        Whether the node's normalized position should be concatenated to
        the features before in_mlp
    :param use_diameter: bool
        Whether the node's diameter should be concatenated to the
        features before in_mlp (assumes diameter to be passed in the
        forward)
    :param use_diameter_parent: bool
        Whether the node's parent diameter should be concatenated to the
        features before in_mlp (only if pos is passed in the forward)
    :param qk_dim:
    :param k_rpe:
    :param q_rpe:
    :param k_delta_rpe:
    :param q_delta_rpe:
    :param qk_share_rpe:
    :param q_on_minus_rpe:
    :param blocks_share_rpe:
    :param heads_share_rpe:
    :param transformer_kwargs:
        Keyword arguments for the TransformerBlock
    """

    def __init__(
            self,
            dim,
            num_blocks=1,
            num_heads=1,
            in_mlp=None,
            out_mlp=None,
            mlp_activation=nn.LeakyReLU(),
            mlp_norm=BatchNorm,
            mlp_drop=None,
            use_pos=True,
            use_diameter=False,
            use_diameter_parent=False,
            qk_dim=8,
            k_rpe=False,
            q_rpe=False,
            k_delta_rpe=False,
            q_delta_rpe=False,
            qk_share_rpe=False,
            q_on_minus_rpe=False,
            blocks_share_rpe=False,
            heads_share_rpe=False,
            **transformer_kwargs):

        super().__init__()

        self.dim = dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        # MLP to change input channel size
        if in_mlp is not None:
            assert in_mlp[-1] == dim
            self.in_mlp = MLP(
                in_mlp,
                activation=mlp_activation,
                norm=mlp_norm,
                drop=mlp_drop)
        else:
            self.in_mlp = None

        # MLP to change output channel size
        if out_mlp is not None:
            assert out_mlp[0] == dim
            self.out_mlp = MLP(
                out_mlp,
                activation=mlp_activation,
                norm=mlp_norm,
                drop=mlp_drop)
        else:
            self.out_mlp = None

        # Transformer blocks
        if num_blocks > 0:

            # Build the RPE encoders here if shared across all blocks
            k_rpe_blocks = _build_shared_rpe_encoders(
                k_rpe, num_blocks, num_heads, 18, qk_dim, blocks_share_rpe,
                heads_share_rpe)

            k_delta_rpe_blocks = _build_shared_rpe_encoders(
                k_delta_rpe, num_blocks, num_heads, dim, qk_dim, blocks_share_rpe,
                heads_share_rpe)

            # If key and query RPEs share the same MLP, only the key MLP
            # is preserved, to limit the number of model parameters
            q_rpe_blocks = _build_shared_rpe_encoders(
                q_rpe and not (k_rpe and qk_share_rpe), num_blocks, num_heads,
                18, qk_dim, blocks_share_rpe, heads_share_rpe)

            q_delta_rpe_blocks = _build_shared_rpe_encoders(
                q_delta_rpe and not (k_delta_rpe and qk_share_rpe),
                num_blocks, num_heads, dim, qk_dim, blocks_share_rpe,
                heads_share_rpe)

            self.transformer_blocks = nn.ModuleList(
                TransformerBlock(
                    dim,
                    num_heads=num_heads,
                    qk_dim=qk_dim,
                    k_rpe=k_rpe_block,
                    q_rpe=q_rpe_block,
                    k_delta_rpe=k_delta_rpe_block,
                    q_delta_rpe=q_delta_rpe_block,
                    qk_share_rpe=qk_share_rpe,
                    q_on_minus_rpe=q_on_minus_rpe,
                    heads_share_rpe=heads_share_rpe,
                    **transformer_kwargs)
                for k_rpe_block, q_rpe_block, k_delta_rpe_block, q_delta_rpe_block
                in zip(k_rpe_blocks, q_rpe_blocks, k_delta_rpe_blocks, q_delta_rpe_blocks))
        else:
            self.transformer_blocks = None

        # UnitSphereNorm converts global node coordinates to
        # segment-level coordinates expressed in a unit-sphere. The
        # corresponding scaling factor (diameter) is returned, to be
        # used in potential subsequent stages
        self.pos_norm = UnitSphereNorm()

        # Fusion operator to combine node positions with node features
        self.feature_fusion = CatFusion()
        self.use_pos = use_pos
        self.use_diameter = use_diameter
        self.use_diameter_parent = use_diameter_parent

    @property
    def out_dim(self):
        if self.out_mlp is not None:
            return self.out_mlp.out_dim
        if self.transformer_blocks is not None:
            return self.transformer_blocks[-1].dim
        if self.in_mlp is not None:
            return self.in_mlp.out_dim
        return self.dim

    def forward(
            self,
            x,
            norm_index,
            pos=None,
            diameter=None,
            node_size=None,
            super_index=None,
            edge_index=None,
            edge_attr=None):

        # Recover info from the input
        if x is not None:
            N = x.shape[0]
            dtype = x.dtype
            device = x.device
        elif pos is not None:
            N = pos.shape[0]
            dtype = pos.dtype
            device = pos.device
        elif diameter is not None:
            N = diameter.shape[0]
            dtype = diameter.dtype
            device = diameter.device
        elif super_index is not None:
            N = super_index.shape[0]
            dtype = edge_attr.dtype if edge_attr is not None else torch.float
            device = super_index.device
        else:
            raise ValueError("Could not infer basic info from input arguments")

        # Append normalized coordinates to the node features
        if pos is not None:
            pos, diameter_parent = self.pos_norm(pos, super_index, w=node_size)
            if self.use_pos:
                x = self.feature_fusion(pos, x)
        else:
            diameter_parent = None

        # Inject the parent segment diameter to the node features if
        # need be
        if self.use_diameter:
            diam = diameter if diameter is not None else \
                torch.zeros((N, 1), dtype=dtype, device=device)
            x = self.feature_fusion(diam, x)

        if self.use_diameter_parent:
            if diameter_parent is None:
                diam = torch.zeros((N, 1), dtype=dtype, device=device)
            elif super_index is None:
                diam = diameter_parent.repeat(N, 1)
            else:
                diam = diameter_parent[super_index]
            x = self.feature_fusion(diam, x)

        # MLP on input features to change channel size
        if self.in_mlp is not None:
            x = self.in_mlp(x, batch=norm_index)

        # Transformer blocks
        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                x, norm_index, edge_index = block(
                    x, norm_index, edge_index=edge_index, edge_attr=edge_attr)

        # MLP on output features to change channel size
        if self.out_mlp is not None:
            x = self.out_mlp(x, batch=norm_index)

        return x, diameter_parent


def _build_shared_rpe_encoders(
        rpe, num_blocks, num_heads, in_dim, out_dim, blocks_share, heads_share):
    """Local helper to build RPE encoders for Stage. The main goal is to
    make shared encoders construction easier.

    Note that setting blocks_share=True will make all blocks use the
    same RPE encoder. It is possible to set blocks_share=True and
    heads_share=False to allow heads of different blocks of the Stage to
    share their RPE encoders while allowing heads of the same block to
    rely on different RPE encoders.
    """
    if not isinstance(rpe, bool):
        assert blocks_share, \
            "If anything else but a boolean is passed for the RPE encoder, " \
            "this value will be passed to all blocks and blocks_share should " \
            "be set to True."
        return [rpe] * num_blocks

    if not heads_share:
        out_dim = out_dim * num_heads

    if blocks_share and rpe:
        return [nn.Linear(in_dim, out_dim)] * num_blocks

    return [rpe] * num_blocks


class DownNFuseStage(Stage):
    """A Stage preceded by a pooling operator and a fusion operator to
    aggregate node features from level-i to level-i+1 and fuse them
    with other features from level-i+1. A DownNFuseStage has the
    following structure:

        x1 ------- Fusion -- Stage -->
                     |
        x2 -- Pool --
    """

    def __init__(self, *args, pool='max', fusion='cat', **kwargs):
        super().__init__(*args, **kwargs)

        # Pooling operator
        # IMPORTANT: the `down_pool_block` naming MUST MATCH the one
        # used in `SemanticSegmentationModule.configure_optimizers()` for
        # differential learning rates to work
        self.down_pool_block = pool_factory(pool)

        # Fusion operator
        self.fusion = fusion_factory(fusion)

    def forward(
            self,
            x_parent,
            x_child,
            norm_index,
            pool_index,
            pos=None,
            diameter=None,
            node_size=None,
            super_index=None,
            edge_index=None,
            edge_attr=None,
            v_edge_attr=None,
            num_super=None):

        # Pool the children features
        x_pooled = self.down_pool_block(
            x_child, x_parent, pool_index, edge_attr=v_edge_attr,
            num_pool=num_super)

        # Fuse parent and pooled child features
        x_fused = self.fusion(x_parent, x_pooled)

        # Stage forward
        return super().forward(
            x_fused,
            norm_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr)


class UpNFuseStage(Stage):
    """A Stage preceded by an unpooling operator and a fusion operator
    to expand node features to from level-i+1 to level-i and fuse them
    with other features from level-i. An UpNFuseStage has the following
    structure:

        x1 --------- Fusion -- Stage -->
                       |
        x2 -- Unpool --

    The UpNFuseStage is typically used in a UNet-like decoder.
    """

    def __init__(self, *args, unpool='index', fusion='cat', **kwargs):
        super().__init__(*args, **kwargs)

        # Unpooling operator
        if unpool == 'index':
            self.unpool = IndexUnpool()
        else:
            raise NotImplementedError(f'Unknown unpool={unpool} mode')

        # Fusion operator
        self.fusion = fusion_factory(fusion)

    def forward(
            self,
            x_child,
            x_parent,
            norm_index,
            unpool_index,
            pos=None,
            diameter=None,
            node_size=None,
            super_index=None,
            edge_index=None,
            edge_attr=None):
        # Unpool the parent features
        x_unpool = self.unpool(x_parent, unpool_index)

        # Fuse unpooled parent and child features
        x_fused = self.fusion(x_child, x_unpool)

        # Stage forward
        return super().forward(
            x_fused,
            norm_index,
            pos=pos,
            node_size=node_size,
            super_index=super_index,
            edge_index=edge_index,
            edge_attr=edge_attr)


class PointStage(Stage):
    """A Stage specifically designed for operating on raw points. This
    is similar to the point-wise part of PointNet, operating on Level-1
    segments. A PointStage has the following structure:

         x  -- PosInjection -- in_MLP -->
                   |         (optional)
        pos -- SphereNorm
    (optional)

    :param in_mlp: List, optional
        Channels for the input MLP. The last channel must match
        `dim`
    :param mlp_activation: nn.Module
        Activation function for the input and output MLPs
    :param mlp_norm: nn.Module
        Normalization for the input and output MLPs
    :param mlp_drop: float, optional
        Dropout rate for the last layer of the input and output MLPs
    :param use_pos: bool
        Whether the node's normalized position should be concatenated to
        the features before in_mlp
    :param use_diameter_parent: bool
        Whether the node's parent diameter should be concatenated to the
        features before in_mlp (only if pos is passed in the forward)
    """

    def __init__(
            self,
            in_mlp,
            mlp_activation=nn.LeakyReLU(),
            mlp_norm=BatchNorm,
            mlp_drop=None,
            use_pos=True,
            use_diameter_parent=False):

        assert len(in_mlp) > 1, \
            'in_mlp should be a list of channels of length >= 2'

        super().__init__(
            in_mlp[-1],
            num_blocks=0,
            in_mlp=in_mlp,
            out_mlp=None,
            mlp_activation=mlp_activation,
            mlp_norm=mlp_norm,
            mlp_drop=mlp_drop,
            use_pos=use_pos,
            use_diameter=False,
            use_diameter_parent=use_diameter_parent)
