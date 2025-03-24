import torch

import mlx
import mlx.core as mx
from tnt.tensor import Environment, Tensor
import llama


def torch_to_mlx(tensor):
  return mx.array(tensor.numpy())


def mlx_to_torch(array):
  return torch.tensor(memoryview(array))


environment = Environment(
  mx.array,
  torch_to_mlx,
  mlx_to_torch
)

def register_op(op):
  def inner(func):
    environment.register_op(op, func)
  return inner

# environment.register_op(
#   op, def_
# )

# environment.register_decomposition(
#   op, def_
# )


args = llama.ModelArgs(
  dim=128,
  n_layers=3,
  n_heads=8,
  n_kv_heads=8,
  vocab_size=32000,
  multiple_of=256,
  norm_eps=1e-5,
  max_batch_size=1,
  max_seq_len=1024
)
llama_model = llama.Transformer(args)

context_length = 128
inputs = torch.randint(0, 1024, (1, context_length))
indexes = torch.arange(0, context_length)
cache_indexes = torch.arange(0, context_length)
mask = torch.full((1, 1, context_length, context_length), float('-inf'))
mask = mask.triu(1)


from torch.ops import aten

@register_op(aten.detach.default)
@register_op(aten.clone.default)
@register_op(aten.view.default)
@register_op(aten._unsafe_view.default)
def aten_detach_default(x):
  return x



@register_op(aten.mm.default)
@register_op(aten.bmm.default)
def aten_mm_default(x, y):
  return x @ y


@register_op(aten.view_as_complex.default)
def aten_view_as_complex_default(x):
  real = x[..., 0]
  imag = x[..., 1]
  return real + 1j * imag



@register_op(aten.div.Tensor)
def aten_div_Tensor(x, y):
  return x / y


@register_op(aten.t.default)
def aten_t_default(x):
  return x.T


@register_op(aten.expand.default)
def aten_expand_default(x, shape):
    """
    Implements torch.expand using MLX.

    Args:
        x (mx.array): Input array.
        shape (tuple): Desired output shape.

    Returns:
        mx.array: Expanded array.
    """
    x_shape = x.shape
    x_ndim = len(x_shape)
    shape_ndim = len(shape)

    if shape_ndim < x_ndim:
        raise ValueError("The number of dimensions of the expanded tensor must be greater or equal to the number of dimensions of the input tensor.")

    new_shape = list(shape)
    for i in range(x_ndim):
        dim_diff = shape_ndim - x_ndim
        if new_shape[dim_diff + i] == -1 or new_shape[dim_diff + i] == x_shape[i]:
            new_shape[dim_diff + i] = x_shape[i]
        elif x_shape[i] != 1:
            raise ValueError(f"shape[{dim_diff + i}] is invalid for input of size {x_shape[i]}")

    repeats = []
    for i in range(shape_ndim):
        if i < shape_ndim - x_ndim:
            repeats.append(shape[i])
        else:
            if x_shape[i - (shape_ndim - x_ndim)] == 1:
                repeats.append(shape[i])
            else:
                repeats.append(1)
    return mx.broadcast_to(x, shape)


@register_op(aten.rsqrt.default)
def aten_rsqrt_default(x):
  return mx.rsqrt(x)


@register_op(aten.add.Tensor)
def aten_add_Tensor(a, b):
  breakpoint()
  return a + b


@register_op(aten._softmax.default)
def aten__softmax_default(x, dim):
  return mlx.nn.softmax(x, dim)


@register_op(aten.embedding.default)
def aten_embedding(weight: mx.array, indices: mx.array, padding_idx: int = None, max_norm: float = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False) -> mx.array:
    """
    Implements torch.embedding using MLX.

    Args:
        weight (mx.array): Embedding weight matrix (num_embeddings, embedding_dim).
        indices (mx.array): Indices to look up.
        padding_idx (int, optional): Index to pad with zeros. Defaults to None.
        max_norm (float, optional): Max norm of embeddings. Defaults to None.
        norm_type (float, optional): Norm type for max norm. Defaults to 2.0.
        scale_grad_by_freq (bool, optional): Not implemented. Defaults to False.
        sparse (bool, optional): Not implemented. Defaults to False.

    Returns:
        mx.array: Embedded output.
    """

    if scale_grad_by_freq:
        raise NotImplementedError("scale_grad_by_freq is not implemented.")
    if sparse:
        raise NotImplementedError("sparse is not implemented.")

    embedded = weight[indices]

    if padding_idx is not None:
        mask = indices == padding_idx
        embedded = mx.where(mx.expand_dims(mask, axis=-1), mx.zeros_like(embedded), embedded)

    if max_norm is not None:
        norms = mx.linalg.norm(embedded, ord=norm_type, axis=-1, keepdims=True)
        clipped_norms = mx.clip(norms, a_max=max_norm)
        embedded = embedded * (clipped_norms / norms)

    return embedded


@register_op(aten.silu.default)
def aten_silu_default(x):
  return mlx.nn.silu(x)


@register_op(aten.mean.dim)
def aten_mean_dim(x, dim, keepdim):
  return x.mean(dim, keepdims=keepdim)


@register_op(aten.mul.Tensor)
def aten_mul_Tensor(a, b):
  return a * b


@register_op(aten.view_as_real.default)
def aten_view_as_real_default(x):
    real = mx.real(x)
    imag = mx.imag(x)
    return mx.stack([real, imag], axis=-1)

@register_op(aten.index.Tensor)
def aten_index_Tensor(x, index):
  return x[index]


@register_op(aten.transpose.int)
def aten_transpose_int(x, dim0, dim1):
  return x.swapaxes(dim0, dim1)



@register_op(aten.index_select.default)
def aten_index_select_default(x, dim, index):
  return mx.take(x, index, axis=dim)


@register_op(aten.pow.Tensor_Scalar)
def aten_pow_Tensor_Scalar(x, a):
  return mx.power(x, a)


@register_op(aten.index_put_.default)
def aten_index_put__default(inputs, indices, values):
    inputs[indices] = values


environment.enable_torch_modes()


def mlx_tensor(torch_t):
  data = torch_to_mlx(torch_t)
  meta = torch_t.to('meta')
  return Tensor(meta, data, environment)

  

a = torch.tensor([1,2,3])
b = torch.tensor([1,2,3])
print(a + b)

