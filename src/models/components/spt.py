from torch import nn
from src.utils import listify_with_reference
from src.nn import Stage, PointStage, DownNFuseStage, UpNFuseStage, \
    BatchNorm, CatFusion, MLP, LayerNorm
from src.nn.pool import BaseAttentivePool
from src.nn.pool import pool_factory

__all__ = ['SPT']


class SPT(nn.Module):
    """Superpoint Transformer. A UNet-like architecture processing NAG.
    """

    def __init__(
            self,

            point_mlp=None,
            point_drop=None,

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
            output_stage_wise=False):

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
            down_drop_path
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
            down_drop_path)

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
            pool_factory(pool, down_pool_dim[0]), BaseAttentivePool)

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
                heads_share_rpe=heads_share_rpe)
        else:
            self.first_stage = PointStage(
                point_mlp,
                mlp_activation=mlp_activation,
                mlp_norm=mlp_norm,
                mlp_drop=point_drop,
                use_pos=use_pos,
                use_diameter_parent=use_diameter_parent)

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
                    pool=pool_factory(pool, pool_dim),
                    fusion=fusion,
                    use_pos=use_pos,
                    use_diameter=use_diameter,
                    use_diameter_parent=use_diameter_parent,
                    blocks_share_rpe=blocks_share_rpe,
                    heads_share_rpe=heads_share_rpe)
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
                    pool_dim)
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
                    down_pool_dim))
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
                    heads_share_rpe=heads_share_rpe)
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
        assert self.nano or self.num_down_stages > self.num_up_stages, \
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

    def forward(self, nag):
        # assert isinstance(nag, NAG)
        # assert nag.num_levels >= 2
        # assert nag.num_levels > self.num_down_stages

        # TODO: this will need to be changed if we want FAST NANO
        if self.nano:
            nag = nag[1:]

        # Apply the first MLPs on the handcrafted features
        if self.nano:
            if self.node_mlps is not None and self.node_mlps[0] is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                nag[0].x = self.node_mlps[0](nag[0].x, batch=norm_index)
            if self.h_edge_mlps is not None:
                norm_index = nag[0].norm_index(mode=self.norm_mode)
                norm_index = norm_index[nag[0].edge_index[0]]
                nag[0].edge_attr = self.h_edge_mlps[0](
                    nag[0].edge_attr, batch=norm_index)

        # Encode level-0 data
        x, diameter = self.first_stage(
            nag[0].x if self.use_node_hf else None,
            nag[0].norm_index(mode=self.norm_mode),
            pos=nag[0].pos,
            diameter=None,
            node_size=getattr(nag[0], 'node_size', None),
            super_index=nag[0].super_index,
            edge_index=nag[0].edge_index,
            edge_attr=nag[0].edge_attr)

        # Add the diameter to the next level's attributes
        nag[1].diameter = diameter

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
                i_level = i_stage + 1

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
                x, diameter = self._forward_down_stage(stage, nag, i_level, x)
                down_outputs.append(x)

                # End here if we reached the last NAG level
                if i_level == nag.num_levels - 1:
                    continue

                # Add the diameter to the next level's attributes
                nag[i_level + 1].diameter = diameter

        # Iteratively decode level-num_down_stages and below
        up_outputs = []
        if self.up_stages is not None:
            for i_stage, stage in enumerate(self.up_stages):
                i_level = self.num_down_stages - i_stage - 1
                x_skip = down_outputs[-(2 + i_stage)]
                x, _ = self._forward_up_stage(stage, nag, i_level, x, x_skip)
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

    def _forward_down_stage(self, stage, nag, i_level, x):
        is_last_level = (i_level == nag.num_levels - 1)
        x_handcrafted = nag[i_level].x if self.use_node_hf else None
        return stage(
            x_handcrafted,
            x,
            nag[i_level].norm_index(mode=self.norm_mode),
            nag[i_level - 1].super_index,
            pos=nag[i_level].pos,
            diameter=nag[i_level].diameter,
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
            diameter=nag[i_level - self.nano].diameter,
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
