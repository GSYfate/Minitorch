from typing import Tuple
from numba import cuda
import numba
from .autodiff import Context
from .tensor_data import (
    to_index,
    index_to_position,
    broadcast_index,
    MAX_DIMS,
    Shape,
    Strides
)
from .tensor_functions import Function
from .tensor import Tensor
to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


def _tensor_conv1d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """
    1D Cuda Convolution implementation.
    Given input tensor of
       `batch, in_channels, width`
    and weight tensor
       `out_channels, in_channels, k_width`
    Computes padded output of
       `batch, out_channels, width`
    `reverse` decides if weight is anchored left (False) or right.
    (See diagrams)
    Args:
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        out_size (int): size of the `out` tensor.
        input (array): storage for `input` tensor.
        input_shape (array): shape for `input` tensor.
        input_strides (array): strides for `input` tensor.
        weight (array): storage for `input` tensor.
        weight_shape (array): shape for `input` tensor.
        weight_strides (array): strides for `input` tensor.
        reverse (bool): anchor weight at left or right
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = weight_strides
    s2 = input_strides
   
    BLOCK_DIM = 32
    cache = cuda.shared.array((MAX_DIMS, MAX_DIMS), numba.float64)
    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    out_pos = cuda.blockIdx.x
    pos_x = cuda.threadIdx.x
    pos_y = cuda.threadIdx.y

    if out_pos < out_size:
        to_index(out_pos, out_shape, out_index)
        b = out_index[0]
        o = out_index[1]
        w = out_index[2]

        channel = pos_x
        idx = pos_y

        if channel < in_channels and idx < kw:
            if reverse:
                idx -= (kw - 1)
            if w + idx < 0 or w + idx >= width:
                input_val = 0.0
                weight_val = 0.0 
            else:
                input_val = input[b * s1[0] + channel * s1[1]  + (w + idx) * s1[2]]
                weight_val = weight[o*s2[0] + channel*s2[1] + idx * s2[2]]  
            cache[channel, idx] = input_val * weight_val
            cuda.syncthreads()

            if channel == 0 and idx == 0:
                for c in range(in_channels):
                    for wi in range(kw):
                        out[out_pos] += cache[c, wi]
            

tensor_conv1d = cuda.jit()(_tensor_conv1d)

class CudaConv1dFun(Function):

    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        # Instantiate and run the cuda kernel.
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        blockspergrid = output.size
        tensor_conv1d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)

        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        blockspergrid = grad_weight.size
        tensor_conv1d[blockspergrid, threadsperblock](
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)
        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        # Instantiate and run the cuda kernel.
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        blockspergrid = grad_input.size
        tensor_conv1d[blockspergrid, threadsperblock](
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


def _tensor_conv2d(
    out: Tensor,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Tensor,
    input_shape: Shape,
    input_strides: Strides,
    weight: Tensor,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:

    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    # inners
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]
   
    BLOCK_DIM = 32
    cache = cuda.shared.array((4, 16,16), numba.float64)
    out_index = cuda.local.array(MAX_DIMS, numba.int32)
    out_pos = cuda.blockIdx.x
    pos_x = cuda.threadIdx.x
    pos_y = cuda.threadIdx.y
    pos_z = cuda.threadIdx.z

    if out_pos < out_size:
        to_index(out_pos, out_shape, out_index)
        b = out_index[0]
        o = out_index[1]
        h = out_index[2]
        w = out_index[3]

        channel = pos_x
        h_i = pos_y
        w_i = pos_z

        if channel < in_channels and w_i < kw and h_i < kh:
            if reverse:
                w_i -= (kw - 1)
                h_i -= (kh - 1)
            if w + w_i < 0 or w + w_i >= width or h + h_i < 0 or h + h_i >=height:
                input_val, weight_val = 0.0,0.0 
            else:
                input_val = input[b * s1[0] + channel * s1[1]  + (h + h_i) * s1[2] + (w + w_i) * s1[3]]
                weight_val = weight[o * s2[0] + channel * s2[1] + h_i * s2[2] + w_i * s2[3]]  
                cache[channel, h_i, w_i] = input_val * weight_val
            cuda.syncthreads()

            if channel == 0 and w_i == 0 and h_i==0:
                for c in range(in_channels):
                    for hi in range(kh):
                      for wi in range(kw):
                        out[out_pos] += cache[c, h_i, wi]
            

tensor_conv2d = cuda.jit()(_tensor_conv2d)

class CudaConv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input : batch x in_channel x h x w
            weight  : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        threadsperblock = (4, 16, 16)
        blockspergrid = output.size
        tensor_conv2d[blockspergrid, threadsperblock](
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape
        
        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        threadsperblock = (4, 16, 16)
        blockspergrid = grad_weight.size
        tensor_conv2d[blockspergrid, threadsperblock](
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        threadsperblock = (4, 16, 16)
        blockspergrid = grad_input.size
        tensor_conv2d[blockspergrid, threadsperblock](
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight