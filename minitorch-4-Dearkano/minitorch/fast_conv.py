import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
)
from .tensor_functions import Function
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
count = njit(inline="always")(count)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


@njit(parallel=True)
def tensor_conv1d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    1D Convolution implementation.

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

    for p in prange(out_size):
        out_index = np.zeros(3, np.int32)
        pos1 = np.zeros(3, np.int32)
        pos2 = np.zeros(3, np.int32)
        count(p, out_shape, out_index)
        cur_batch, cur_out_channels, cur_n = out_index
        # input matrix - out_tensor[cur_batch, cur_out_channels: cur_out_channels + in_channels, cur_n: cur_n + kw]
        # shape of input matrix: (1, in_channels, kw)
        # weight matrix - (1, in_channels, kw)
        v = 0
        for i in range(in_channels):
            for j in range(kw):
                if not reverse:
                    pos1[0] = cur_out_channels
                    pos1[1] = i
                    pos1[2] = j
                    pos2[0] = cur_batch
                    pos2[1] = i
                    pos2[2] = cur_n + j
                    if cur_n + j >= width:
                        v += 0
                    else:
                        v += weight[index_to_position(pos1, weight_strides)] * input[index_to_position(
                            pos2, input_strides)]
                else:
                    if cur_n - j < 0:
                        v += 0
                    else:
                        pos1[0] = cur_out_channels
                        pos1[1] = i
                        pos1[2] = j
                        pos2[0] = cur_batch
                        pos2[1] = i
                        pos2[2] = cur_n - j
                        v += weight[index_to_position(pos1, weight_strides)] * input[index_to_position(
                            pos2, input_strides)]
        out[index_to_position(out_index, out_strides)] = v


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 1D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Run convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


@njit(parallel=True)
def tensor_conv2d(
    out,
    out_shape,
    out_strides,
    out_size,
    input,
    input_shape,
    input_strides,
    weight,
    weight_shape,
    weight_strides,
    reverse,
):
    """
    2D Convolution implementation.

    Given input tensor of

       `batch, in_channels, height, width`

    and weight tensor

       `out_channels, in_channels, k_height, k_width`

    Computes padded output of

       `batch, out_channels, height, width`

    `Reverse` decides if weight is anchored top-left (False) or bottom-right.
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
        reverse (bool): anchor weight at top-left or bottom-right
    """
    batch_, out_channels, _, _ = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    for p in prange(out_size):
        out_index = np.zeros(4, np.int32)
        pos1 = np.zeros(4, np.int32)
        pos2 = np.zeros(4, np.int32)
        count(p, out_shape, out_index)
        cur_batch, cur_out_channels, cur_h, cur_w = out_index
        v = 0

        for i in range(in_channels):
            for j in range(kh):
                for k in range(kw):
                    if not reverse:
                        pos1[0] = cur_out_channels
                        pos1[1] = i
                        pos1[2] = j
                        pos1[3] = k
                        pos2[0] = cur_batch
                        pos2[1] = i
                        pos2[2] = cur_h + j
                        pos2[3] = cur_w + k
                        if cur_h + j >= height or cur_w + k >= width:
                            v += 0
                        else:
                            v += weight[index_to_position(pos1, weight_strides)] * input[index_to_position(
                                pos2, input_strides)]
                    else:
                        pos1[0] = cur_out_channels
                        pos1[1] = i
                        pos1[2] = j
                        pos1[3] = k
                        pos2[0] = cur_batch
                        pos2[1] = i
                        pos2[2] = cur_h - j
                        pos2[3] = cur_w - k
                        if cur_h - j < 0 or cur_w - k < 0:
                            v += 0
                        else:
                            v += weight[index_to_position(pos1, weight_strides)] * input[index_to_position(
                                pos2, input_strides)]
        out[index_to_position(out_index, out_strides)] = v


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        Compute a 2D Convolution

        Args:
            ctx : Context
            input (:class:`Tensor`) : batch x in_channel x h x w
            weight (:class:`Tensor`) : out_channel x in_channel x kh x kw

        Returns:
            (:class:`Tensor`) : batch x out_channel x h x w
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2
        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
