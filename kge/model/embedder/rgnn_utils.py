from math import floor, sqrt
import random
import torch
from typing import Optional, Tuple

# ---- Message Passing Helper Functions ---- #

# TODO referencen woher (CompGCN special ding)
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:

    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.floor_divide_(count)
    return out


def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError

def scatter_(name, src, index, dim_size=None):
	r"""Aggregates all values from the :attr:`src` tensor at the indices
	specified in the :attr:`index` tensor along the first dimension.
	If multiple indices reference the same location, their contributions
	are aggregated according to :attr:`name` (either :obj:`"add"`,
	:obj:`"mean"` or :obj:`"max"`).

	Args:
		name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
			:obj:`"max"`).
		src (Tensor): The source tensor.
		index (LongTensor): The indices of elements to scatter.
		dim_size (int, optional): Automatically create output tensor with size
			:attr:`dim_size` in the first dimension. If set to :attr:`None`, a
			minimal sized output tensor is returned. (default: :obj:`None`)

	:rtype: :class:`Tensor`
	"""
	if name == 'add': name = 'sum'
	assert name in ['sum', 'mean', 'max']
	out = scatter(src, index, dim=0, out=None, dim_size=dim_size, reduce=name)
	return out[0] if isinstance(out, tuple) else out

# ---- RGCN-specific Weight Initialisation Methods ---- #

# initialisation methods adapted from the R-GCN PyTorch Implementation:
# https://arxiv.org/pdf/2107.10015.pdf

def schlichtkrull_std(tensor, gain, shape=None):
    """
    a = \text{gain} \times \frac{3}{\sqrt{\text{fan\_in} + \text{fan\_out}}}
    """
    if shape: 
        fan_in, fan_out = shape[0], shape[1]
    else:
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    return gain * 3.0 / sqrt(float(fan_in + fan_out))

def schlichtkrull_normal_(tensor, gain=1., shape=None):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a normal distribution."""
    std = schlichtkrull_std(tensor, gain, shape)
    with torch.no_grad():
        return tensor.normal_(0.0, std)

def schlichtkrull_uniform_(tensor, gain=1., shape=None):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a uniform distribution."""
    std = schlichtkrull_std(tensor, gain, shape)
    with torch.no_grad():
        return tensor.uniform_(-std, std)

def wgcn_uniform_(tensor):
    if tensor.dim()==1:
        std = 1./sqrt(tensor.size(0))
    if tensor.dim()==2:
        std = 1./sqrt(tensor.size(1))
    with torch.no_grad():
        return tensor.uniform_(-std, std)

# ---- Composition Functions for Message Passing ---- #

def neighbor(h_i, h_j, h_r, message_weight=None):
    return h_j

def sub(h_i, h_j, h_r, message_weight=None):
    return h_j-h_r   

def sub_weighted(h_i, h_j, h_r, message_weight):
    return h_j*rel_weight-h_r  

def mult(h_i, h_j, h_r, message_weight=None):
    return h_j*h_r   

def mult_weighted(h_i, h_j, h_r, message_weight):
    return h_j*h_r*rel_weight   

def ccorr(h_i, h_j, h_r, message_weight=None):
    return torch.irfft(
        com_mult(conj(torch.rfft(h_j, 1)), torch.rfft(h_r, 1)),
        1, 
        signal_sizes=(h_r.shape[-1],)
    )  

def ccorr_weighted(h_i, h_j, h_r, message_weight):
    weighted_h_j = h_j * rel_weight
    return torch.irfft(
        com_mult(conj(torch.rfft(weighted_h_j, 1)), torch.rfft(h_r, 1)),
        1, 
        signal_sizes=(h_r.shape[-1],)
    )  

def cross(h_i, h_j, h_r, message_weight=None):
    return h_j*h_r+h_j

def cross_weighted(h_i, h_j, h_r, message_weight=None):
    return h_j*h_r*message_weight + h_j*message_weight




# ---- Sparse Grouping of Values from Edges to Entites ---- #

import torch


class Edge2NodeFunction(torch.autograd.Function):
    """Sparse aggregation of values from edges to either messaging or receiving
    nodes.
    Adopted from the RAGAT implementation (original name: SpecialSpmmFunctionFinal)
    https://github.com/liuxiyang641/RAGAT."""

    @staticmethod
    def forward(ctx, edge, edge_w, size1, size2, out_features, dim):
        # assert indices.requires_grad == False
        # assert not torch.isnan(edge).any()
        # assert not torch.isnan(edge_w).any()

        # create tensor with edge indices and scores as values
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([size1, size2, out_features]))
        # sum values over the index values
        b = torch.sparse.sum(a, dim=dim)
        ctx.size1 = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.size2 = size2
        if dim == 0:
            ctx.indices = a._indices()[1, :]
        else:
            ctx.indices = a._indices()[0, :]
        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices
            if torch.cuda.is_available():
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None, None


class Edge2Node(torch.nn.Module):
    def forward(self, edge, edge_w, size1, size2, out_features, dim=1):
        return Edge2NodeFunction.apply(edge, edge_w, size1, size2, out_features, dim)
