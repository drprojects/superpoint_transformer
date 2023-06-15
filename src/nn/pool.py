import torch
from torch import nn
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.nn.aggr import MaxAggregation
from torch_geometric.nn.aggr import MinAggregation
from torch_scatter import scatter_sum
from torch_geometric.utils import softmax
from src.utils.nn import init_weights, LearnableParameter, build_qk_scale_func


__all__ = [
    'pool_factory', 'SumPool', 'MeanPool', 'MaxPool', 'MinPool',
    'AttentivePool', 'AttentivePoolWithLearntQueries']


def pool_factory(pool, *args, **kwargs):
    """Build a Pool module from string or from an existing module. This
    helper is intended to be used as a helper in spt and Stage
    constructors.
    """
    if isinstance(pool, (AggregationPoolMixIn, BaseAttentivePool)):
        return pool
    if pool == 'max':
        return MaxPool()
    if pool == 'min':
        return MinPool()
    if pool == 'mean':
        return MeanPool()
    if pool == 'sum':
        return SumPool()
    return pool(*args, **kwargs)


class AggregationPoolMixIn:
    """MixIn class to convert torch-geometric Aggregation modules into
    Pool modules with our desired forward signature.

    :param x_child: Tensor of shape (Nc, Cc)
        Node features for the children nodes
    :param x_parent: Any
        Not used for Aggregation
    :param index: LongTensor of shape (Nc)
        Indices indicating the parent of each for each child node
    :param edge_attr: Any
        Not used for Aggregation
    :param num_pool: int
        Number of parent nodes Nc. If not provided, will be inferred
        from `index.max() + 1`
    :return:
    """
    def __call__(self, x_child, x_parent, index, edge_attr=None, num_pool=None):
        return super().__call__(x_child, index=index, dim_size=num_pool)


class SumPool(AggregationPoolMixIn, SumAggregation):
    pass


class MeanPool(AggregationPoolMixIn, MeanAggregation):
    pass


class MaxPool(AggregationPoolMixIn, MaxAggregation):
    pass


class MinPool(AggregationPoolMixIn, MinAggregation):
    pass


class BaseAttentivePool(nn.Module):
    """Base class for attentive pooling classes. This class is not
    intended to be instantiated, but avoids duplicating code between
    similar child classes, which are expected to implement:
      - `_get_query()`
    """

    def __init__(
            self,
            dim=None,
            num_heads=1,
            in_dim=None,
            out_dim=None,
            qkv_bias=True,
            qk_dim=8,
            qk_scale=None,
            attn_drop=None,
            drop=None,
            in_rpe_dim=9,
            k_rpe=False,
            q_rpe=False,
            v_rpe=False,
            heads_share_rpe=False):
        super().__init__()

        assert dim % num_heads == 0, f"dim must be a multiple of num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.qk_scale = build_qk_scale_func(dim, num_heads, qk_scale)
        self.heads_share_rpe = heads_share_rpe

        self.kv = nn.Linear(dim, qk_dim * num_heads + dim, bias=qkv_bias)

        # Build the RPE encoders, with the option of sharing weights
        # across all heads
        rpe_dim = qk_dim if heads_share_rpe else qk_dim * num_heads

        if not isinstance(k_rpe, bool):
            self.k_rpe = k_rpe
        else:
            self.k_rpe = nn.Linear(in_rpe_dim, rpe_dim) if k_rpe else None

        if not isinstance(q_rpe, bool):
            self.q_rpe = q_rpe
        else:
            self.q_rpe = nn.Linear(in_rpe_dim, rpe_dim) if q_rpe else None

        if v_rpe:
            raise NotImplementedError

        self.in_proj = nn.Linear(in_dim, dim) if in_dim is not None else None
        self.out_proj = nn.Linear(dim, out_dim) if out_dim is not None else None

        self.attn_drop = nn.Dropout(attn_drop) \
            if attn_drop is not None and attn_drop > 0 else None
        self.out_drop = nn.Dropout(drop) \
            if drop is not None and drop > 0 else None

    def forward(
            self, x_child, x_parent, index, edge_attr=None, num_pool=None):
        """
        :param x_child: Tensor of shape (Nc, Cc)
            Node features for the children nodes
        :param x_parent: Tensor of shape (Np, Cp)
            Node features for the parent nodes
        :param index: LongTensor of shape (Nc)
            Indices indicating the parent of each for each child node
        :param edge_attr: FloatTensor or shape (Nc, F)
            Edge attributes for relative pose encoding
        :param num_pool: int
            Number of parent nodes Nc. If not provided, will be inferred
            from the shape of x_parent
        :return:
        """
        Nc = x_child.shape[0]
        Np = x_parent.shape[0] if num_pool is None else num_pool
        H = self.num_heads
        D = self.qk_dim
        DH = D * H

        # Optional linear projection of features
        if self.in_proj is not None:
            x_child = self.in_proj(x_child)

        # Compute queries from parent features
        q = self._get_query(x_parent)  # [Np, DH]

        # Compute keys and values from child features
        kv = self.kv(x_child)  # [Nc, DH + C]

        # Expand queries and separate keys and values
        q = q[index].view(Nc, H, D)     # [Nc, H, D]
        k = kv[:, :DH].view(Nc, H, D)   # [Nc, H, D]
        v = kv[:, DH:].view(Nc, H, -1)  # [Nc, H, C // H]

        # Apply scaling on the queries
        q = q * self.qk_scale(index)

        if self.k_rpe is not None:
            rpe = self.k_rpe(edge_attr)

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            k = k + rpe.view(Nc, H, -1)

        if self.q_rpe is not None:
            rpe = self.q_rpe(edge_attr)

            # Expand RPE to all heads if heads share the RPE encoder
            if self.heads_share_rpe:
                rpe = rpe.repeat(1, H)

            q = q + rpe.view(Nc, H, -1)

        # Compute compatibility scores from the query-key products
        compat = torch.einsum('nhd, nhd -> nh', q, k)  # [Nc, H]

        # Compute the attention scores with scaled softmax
        attn = softmax(compat, index=index, dim=0, num_nodes=Np)  # [Nc, H]

        # Optional attention dropout
        if self.attn_drop is not None:
            attn = self.attn_drop(attn)

        # Apply the attention on the values
        x = (v * attn.unsqueeze(-1)).view(Nc, self.dim)  # [Nc, C]
        x = scatter_sum(x, index, dim=0, dim_size=Np)  # [Np, C]

        # Optional linear projection of features
        if self.out_proj is not None:
            x = self.out_proj(x)  # [Np, out_dim]

        # Optional dropout on projection of features
        if self.out_drop is not None:
            x = self.out_drop(x)  # [Np, C] or [Np, out_dim]

        return x

    def _get_query(self, x_parent):
        """Overwrite this method to implement the attentive pooling.

        :param x_parent: Tensor of shape (Np, Cp)
            Node features for the parent nodes

        :returns Tensor of shape (Np, D * H)
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


class AttentivePool(BaseAttentivePool):
    def __init__(
            self,
            dim=None,
            q_in_dim=None,
            num_heads=1,
            in_dim=None,
            out_dim=None,
            qkv_bias=True,
            qk_dim=8,
            qk_scale=None,
            attn_drop=None,
            drop=None,
            in_rpe_dim=9,
            k_rpe=False,
            q_rpe=False,
            v_rpe=False,
            heads_share_rpe=False):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            in_dim=in_dim,
            out_dim=out_dim,
            qkv_bias=qkv_bias,
            qk_dim=qk_dim,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            in_rpe_dim=in_rpe_dim,
            k_rpe=k_rpe,
            q_rpe=q_rpe,
            v_rpe=v_rpe,
            heads_share_rpe=heads_share_rpe)

        # Queries will be built from input parent feature
        self.q = nn.Linear(q_in_dim, qk_dim * num_heads, bias=qkv_bias)

    def _get_query(self, x_parent):
        """Build queries from input parent features

        :param x_parent: Tensor of shape (Np, Cp)
            Node features for the parent nodes

        :returns Tensor of shape (Np, D * H)
        """
        return self.q(x_parent)  # [Np, DH]



class AttentivePoolWithLearntQueries(BaseAttentivePool):
    def __init__(
            self,
            dim=None,
            num_heads=1,
            in_dim=None,
            out_dim=None,
            qkv_bias=True,
            qk_dim=8,
            qk_scale=None,
            attn_drop=None,
            drop=None,
            in_rpe_dim=18,
            k_rpe=False,
            q_rpe=False,
            v_rpe=False,
            heads_share_rpe=False):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            in_dim=in_dim,
            out_dim=out_dim,
            qkv_bias=qkv_bias,
            qk_dim=qk_dim,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            drop=drop,
            in_rpe_dim=in_rpe_dim,
            k_rpe=k_rpe,
            q_rpe=q_rpe,
            v_rpe=v_rpe,
            heads_share_rpe=heads_share_rpe)

        # Each head will learn its own query and all parent nodes will
        # use these same queries.
        self.q = LearnableParameter(torch.zeros(qk_dim * num_heads))

        # `init_weights` initializes the weights with a truncated normal
        # distribution
        init_weights(self.q)

    def _get_query(self, x_parent):
        """Build queries from learnable queries. The parent features are
        simply used to get the number of parent nodes and expand the
        learnt queries accordingly.

        :param x_parent: Tensor of shape (Np, Cp)
            Node features for the parent nodes

        :returns Tensor of shape (Np, D * H)
        """
        Np = x_parent.shape[0]
        return self.q.repeat(Np, 1)
