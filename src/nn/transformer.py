from torch import nn
from src.nn import SelfAttentionBlock, FFN, DropPath, LayerNorm, \
    INDEX_BASED_NORMS


__all__ = ['TransformerBlock']


# TODO: Careful with how we define the index for LayerNorm:
#  cluster-wise or cloud-wise ? Maybe cloud-wise, seems more stable...


class TransformerBlock(nn.Module):
    """Base block of the Transformer architecture:

        x ---------------- + ---------------- + -->
            \             |   \              |
             -- N -- SA --     -- N -- FFN --

    Where:
        - N: Normalization
        - SA: Self-Attention
        - FFN: Feed-Forward Network

    Inspired by: https://github.com/microsoft/Swin-Transformer

    :param dim: int
        Dimension of the features space on which the transformer
        block operates
    :param num_heads: int
        Number of attention heads
    :param qkv_bias: bool
        Whether the linear layers producing queries, keys, and
        values should have a bias
    :param qk_dim: int
        Dimension of the queries and keys
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
        `1 / sqrt(g)`
    :param in_rpe_dim: int
        Dimension of the features passed as input for relative
        positional encoding computation (i.e. edge features)
    :param ffn_ratio: int
        Multiplicative factor for computing the dimension of the
        `FFN` inverted bottleneck: `ffn_ratio * dim`
    :param attn_drop: float
        Dropout on the attention weights of the `SelfAttentionBlock`
    :param residual_drop: float
        Dropout on the output features of the `SelfAttentionBlock`
    :param drop_path: float
        Dropout on the `SelfAttentionBlock` and `FFN` paths. Contrary
        to other dropout parameters, here we either keep all or none
        features. This allows training with stochastic depth
    :param activation: nn.Module
        Activation function for the `FFN` module
    :param norm: nn.Module
        Normalization function for the `FFN` module
    :param pre_norm: bool
        Whether the normalization should be applied before or after
        the `SelfAttentionBlock` and `FFN` in the residual branches
    :param no_sa: bool
        Whether a self-attention residual branch should be used at
        all
    :param no_ffn: bool
        Whether a feed-forward residual branch should be used at
        all
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
    :param heads_share_rpe: bool
        whether attention heads should share the same parameters for
        building relative positional encodings. See
        `SelfAttentionBlock`
    """

    def __init__(
            self,
            dim,
            num_heads=1,
            qkv_bias=True,
            qk_dim=8,
            qk_scale=None,
            in_rpe_dim=18,
            ffn_ratio=4,
            attn_drop=None,
            residual_drop=None,
            drop_path=None,
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
            heads_share_rpe=False):
        super().__init__()

        self.dim = dim
        self.pre_norm = pre_norm

        # Self-Attention residual branch
        self.no_sa = no_sa
        if not no_sa:
            self.sa_norm = norm(dim)
            self.sa = SelfAttentionBlock(
                dim,
                num_heads=num_heads,
                in_dim=None,
                out_dim=dim,
                qkv_bias=qkv_bias,
                qk_dim=qk_dim,
                qk_scale=qk_scale,
                in_rpe_dim=in_rpe_dim,
                attn_drop=attn_drop,
                drop=residual_drop,
                k_rpe=k_rpe,
                q_rpe=q_rpe,
                v_rpe=v_rpe,
                k_delta_rpe=k_delta_rpe,
                q_delta_rpe=q_delta_rpe,
                qk_share_rpe=qk_share_rpe,
                q_on_minus_rpe=q_on_minus_rpe,
                heads_share_rpe=heads_share_rpe)

        # Feed-Forward Network residual branch
        self.no_ffn = no_ffn
        if not no_ffn:
            self.ffn_norm = norm(dim)
            self.ffn_ratio = ffn_ratio
            self.ffn = FFN(
                dim,
                hidden_dim=int(dim * ffn_ratio),
                activation=activation,
                drop=residual_drop)

        # Optional DropPath module for stochastic depth
        self.drop_path = DropPath(drop_path) \
            if drop_path is not None and drop_path > 0 else nn.Identity()

    def forward(self, x, norm_index, edge_index=None, edge_attr=None):
        """
        :param x: FloatTensor or shape (N, C)
            Node features
        :param norm_index: LongTensor or shape (N)
            Node indices for the LayerNorm
        :param edge_index: LongTensor of shape (2, E)
            Edges in torch_geometric [[sources], [targets]] format for
            the self-attention module
        :param edge_attr: FloatTensor or shape (E, F)
            Edge attributes in torch_geometric format for relative pose
            encoding in the self-attention module
        :return:
        """
        assert x.dim() == 2, 'x should be a 2D Tensor'
        assert x.is_floating_point(), 'x should be a 2D FloatTensor'
        assert norm_index.dim() == 1 and norm_index.shape[0] == x.shape[0], \
            'norm_index should be a 1D LongTensor'
        assert edge_index is None or \
               (edge_index.dim() == 2 and not edge_index.is_floating_point()), \
            'edge_index should be a 2D LongTensor'
        assert edge_attr is None or \
               (edge_attr.dim() == 2 and edge_attr.shape[0] == edge_index.shape[1]),\
            'edge_attr be a 2D LongTensor'

        # Keep track of x for the residual connection
        shortcut = x

        # Self-Attention residual branch. Skip the SA block if no edges
        # are provided
        if self.no_sa or edge_index is None or edge_index.shape[1] == 0:
            pass
        elif self.pre_norm:
            x = self._forward_norm(self.sa_norm, x, norm_index)
            x = self.sa(x, edge_index, edge_attr=edge_attr)
            x = shortcut + self.drop_path(x)
        else:
            x = self.sa(x, edge_index, edge_attr=edge_attr)
            x = self.drop_path(x)
            x = self._forward_norm(self.sa_norm, shortcut + x, norm_index)

        # Feed-Forward Network residual branch
        if not self.no_ffn and self.pre_norm:
            x = self._forward_norm(self.ffn_norm, x, norm_index)
            x = self.ffn(x)
            x = shortcut + self.drop_path(x)
        if not self.no_ffn and not self.pre_norm:
            x = self.ffn(x)
            x = self.drop_path(x)
            x = self._forward_norm(self.ffn_norm, shortcut + x, norm_index)

        return x, norm_index, edge_index

    @staticmethod
    def _forward_norm(norm, x, norm_index):
        """Simple helper for the forward pass on norm modules. Some
        modules require an index, while others don't.
        """
        if isinstance(norm, INDEX_BASED_NORMS):
            return norm(x, batch=norm_index)
        return norm(x)
