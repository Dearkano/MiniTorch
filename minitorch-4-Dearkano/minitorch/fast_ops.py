import numpy as np
from .tensor_data import (
    count,
    index_to_position,
    broadcast_index,
    shape_broadcast,
    MAX_DIMS,
)
from numba import njit, prange


# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
count = njit(inline="always")(count)
index_to_position = njit(inline="always")(index_to_position)
broadcast_index = njit(inline="always")(broadcast_index)


def tensor_map(fn):
    """
    NUMBA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.
        out (array): storage for out tensor.
        out_shape (array): shape for out tensor.
        out_strides (array): strides for out tensor.
        in_storage (array): storage for in tensor.
        in_shape (array): shape for in tensor.
        in_strides (array): strides for in tensor.

    Returns:
        None : Fills in `out`
    """

    def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
        # TODO: Implement for Task 2.2.

        for i in prange(len(out)):
            out_index = np.zeros(MAX_DIMS, np.int32)
            in_index = np.zeros(MAX_DIMS, np.int32)
            count(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            data = in_storage[index_to_position(in_index, in_strides)]
            map_data = fn(data)
            out[index_to_position(out_index, out_strides)] = map_data

    return njit(parallel=True)(_map)


def map(fn):
    """
    Higher-order tensor map function ::

      fn_map = map(fn)
      b = fn_map(a)


    Args:
        fn: function from float-to-float to apply.
        a (:class:`Tensor`): tensor to map over
        out (:class:`Tensor`): optional, tensor data to fill in,
               should broadcast with `a`

    Returns:
        :class:`Tensor` : new tensor
    """

    # This line JIT compiles your tensor_map
    f = tensor_map(njit()(fn))

    def ret(a, out=None):
        if out is None:
            out = a.zeros(a.shape)
        f(*out.tuple(), *a.tuple())
        return out

    return ret


def tensor_zip(fn):
    """
    NUMBA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function maps two floats to float to apply.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        b_storage (array): storage for `b` tensor.
        b_shape (array): shape for `b` tensor.
        b_strides (array): strides for `b` tensor.

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        b_storage,
        b_shape,
        b_strides,
    ):
        for i in prange(len(out)):
            a_index = np.zeros(MAX_DIMS, np.int32)
            b_index = np.zeros(MAX_DIMS, np.int32)
            o_index = np.zeros(MAX_DIMS, np.int32)
            count(i, out_shape, o_index)
            broadcast_index(o_index, out_shape, a_shape, a_index)
            broadcast_index(o_index, out_shape, b_shape, b_index)
            a_data = a_storage[index_to_position(a_index, a_strides)]
            b_data = b_storage[index_to_position(b_index, b_strides)]
            map_data = fn(a_data, b_data)
            out[index_to_position(o_index, out_strides)] = map_data

    return njit(parallel=True)(_zip)


def zip(fn):
    """
    Higher-order tensor zip function ::

      fn_zip = zip(fn)
      c = fn_zip(a, b)

    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to zip over
        b (:class:`Tensor`): tensor to zip over

    Returns:
        :class:`Tensor` : new tensor
    """
    f = tensor_zip(njit()(fn))

    def ret(a, b):
        c_shape = shape_broadcast(a.shape, b.shape)
        out = a.zeros(c_shape)
        f(*out.tuple(), *a.tuple(), *b.tuple())
        return out

    return ret


def tensor_reduce(fn):
    """
    NUMBA higher-order tensor reduce function.

    Args:
        fn: reduction function mapping two floats to float.
        out (array): storage for `out` tensor.
        out_shape (array): shape for `out` tensor.
        out_strides (array): strides for `out` tensor.
        a_storage (array): storage for `a` tensor.
        a_shape (array): shape for `a` tensor.
        a_strides (array): strides for `a` tensor.
        reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
        reduce_size (int): size of reduce shape

    Returns:
        None : Fills in `out`

    """

    def _reduce(
        out,
        out_shape,
        out_strides,
        a_storage,
        a_shape,
        a_strides,
        reduce_shape,
        reduce_size,
    ):
        for p in prange(len(out)):
            index = np.zeros(MAX_DIMS, np.int32)
            offset = np.zeros(MAX_DIMS, np.int32)
            count(p, out_shape, index)
            k = index_to_position(index, out_strides)
            for s in prange(reduce_size):
                count(s, reduce_shape, offset)
                a_index = index + offset
                out[k] = fn(
                    out[k], a_storage[index_to_position(a_index, a_strides)])

    return njit(parallel=True)(_reduce)


def reduce(fn, start=0.0):
    """
    Higher-order tensor reduce function. ::

      fn_reduce = reduce(fn)
      reduced = fn_reduce(a, dims)


    Args:
        fn: function from two floats-to-float to apply
        a (:class:`Tensor`): tensor to reduce over
        dims (list, optional): list of dims to reduce
        out (:class:`Tensor`, optional): tensor to reduce into

    Returns:
        :class:`Tensor` : new tensor
    """

    f = tensor_reduce(njit()(fn))

    def ret(a, dims=None, out=None):
        old_shape = None
        if out is None:
            out_shape = list(a.shape)
            for d in dims:
                out_shape[d] = 1
            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start
        else:
            old_shape = out.shape
            diff = len(a.shape) - len(out.shape)
            out = out.view(*([1] * diff + list(old_shape)))

        # Assume they are the same dim
        assert len(out.shape) == len(a.shape)

        # Create a reduce shape / reduce size
        reduce_shape = []
        reduce_size = 1
        for i, s in enumerate(a.shape):
            if out.shape[i] == 1:
                reduce_shape.append(s)
                reduce_size *= s
            else:
                reduce_shape.append(1)

        # Apply
        f(*out.tuple(), *a.tuple(), np.array(reduce_shape), reduce_size)

        if old_shape is not None:
            out = out.view(*old_shape)
        return out

    return ret


@njit(parallel=True)
def tensor_matrix_multiply(
    out,
    out_shape,
    out_strides,
    a_storage,
    a_shape,
    a_strides,
    b_storage,
    b_shape,
    b_strides,
):
    """
    NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as ::

        assert a_shape[-1] == b_shape[-2]

    Args:
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """
    inner = a_shape[-1]
    for i in prange(len(out)):
        out_index = np.empty(MAX_DIMS, np.int32)
        a_start_index = np.empty(MAX_DIMS, np.int32)
        b_start_index = np.empty(MAX_DIMS, np.int32)
        count(i, out_shape, out_index)

        count(i, out_shape, a_start_index)
        a_start_index[len(out_shape) - 1] = 0
        a_index = np.empty(MAX_DIMS, np.int32)
        broadcast_index(a_start_index, out_shape, a_shape, a_index)
        a_position = index_to_position(a_index, a_strides)

        count(i, out_shape, b_start_index)
        b_start_index[len(out_shape) - 2] = 0
        b_index = np.empty(MAX_DIMS, np.int32)
        broadcast_index(b_start_index, out_shape, b_shape, b_index)
        b_position = index_to_position(b_index, b_strides)

        # Reduce over a and b to fill in out_position
        sum_out = 0
        out_position = index_to_position(out_index, out_strides)
        for k in range(inner):
            # increment
            new_a_position = a_position + k * a_strides[-1]
            new_b_position = b_position + k * b_strides[-2]
            sum_out += a_storage[new_a_position] * b_storage[new_b_position]
        out[out_position] = sum_out


def matrix_multiply(a, b):
    """
    Tensor matrix multiply

    Should work for any tensor shapes that broadcast in the first n-2 dims and
    have ::

        assert a.shape[-1] == b.shape[-2]

    Args:
        a (:class:`Tensor`): tensor a
        b (:class:`Tensor`): tensor b

    Returns:
        :class:`Tensor` : new tensor
    """

    # Create out shape
    # START CODE CHANGE
    ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
    ls.append(a.shape[-2])
    ls.append(b.shape[-1])
    assert a.shape[-1] == b.shape[-2]
    # END CODE CHANGE
    out = a.zeros(tuple(ls))

    # Call main function
    tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
    return out


class FastOps:
    map = map
    zip = zip
    reduce = reduce
    matrix_multiply = matrix_multiply
# import numpy as np
# from .tensor_data import (
#     count,
#     index_to_position,
#     broadcast_index,
#     shape_broadcast,
#     MAX_DIMS,
# )
# from numba import njit, prange


# # This code will JIT compile fast versions your tensor_data functions.
# # If you get an error, read the docs for NUMBA as to what is allowed
# # in these functions.
# count = njit(inline="always")(count)
# index_to_position = njit(inline="always")(index_to_position)
# broadcast_index = njit(inline="always")(broadcast_index)


# def tensor_map(fn):
#     """
#     NUMBA higher-order tensor map function. ::

#       fn_map = tensor_map(fn)
#       fn_map(out, ... )

#     Args:
#         fn: function mappings floats-to-floats to apply.
#         out (array): storage for out tensor.
#         out_shape (array): shape for out tensor.
#         out_strides (array): strides for out tensor.
#         in_storage (array): storage for in tensor.
#         in_shape (array): shape for in tensor.
#         in_strides (array): strides for in tensor.

#     Returns:
#         None : Fills in `out`
#     """

#     def _map(out, out_shape, out_strides, in_storage, in_shape, in_strides):
#         # TODO: Implement for Task 3.1.
#         if (
#             len(out_strides) != len(in_strides)
#             or (out_strides != in_strides).any()
#             or (out_shape != in_shape).any()
#         ):
#             for i in prange(len(out)):
#                 # index for out tenser
#                 out_index = np.empty(MAX_DIMS, np.int32)
#                 in_index = np.empty(MAX_DIMS, np.int32)
#                 count(i, out_shape, out_index)
#                 broadcast_index(out_index, out_shape, in_shape, in_index)
#                 in_position = index_to_position(in_index, in_strides)
#                 out_position = index_to_position(out_index, out_strides)
#                 out[out_position] = fn(in_storage[in_position])
#         else:
#             for i in prange(len(out)):
#                 out[i] = fn(in_storage[i])

#     return njit(parallel=True)(_map)


# def map(fn):
#     """
#     Higher-order tensor map function ::

#       fn_map = map(fn)
#       b = fn_map(a)


#     Args:
#         fn: function from float-to-float to apply.
#         a (:class:`Tensor`): tensor to map over
#         out (:class:`Tensor`): optional, tensor data to fill in,
#                should broadcast with `a`

#     Returns:
#         :class:`Tensor` : new tensor
#     """

#     # This line JIT compiles your tensor_map
#     f = tensor_map(njit()(fn))

#     def ret(a, out=None):
#         if out is None:
#             out = a.zeros(a.shape)
#         f(*out.tuple(), *a.tuple())
#         return out

#     return ret


# def tensor_zip(fn):
#     """
#     NUMBA higher-order tensor zipWith (or map2) function ::

#       fn_zip = tensor_zip(fn)
#       fn_zip(out, ...)

#     Args:
#         fn: function maps two floats to float to apply.
#         out (array): storage for `out` tensor.
#         out_shape (array): shape for `out` tensor.
#         out_strides (array): strides for `out` tensor.
#         a_storage (array): storage for `a` tensor.
#         a_shape (array): shape for `a` tensor.
#         a_strides (array): strides for `a` tensor.
#         b_storage (array): storage for `b` tensor.
#         b_shape (array): shape for `b` tensor.
#         b_strides (array): strides for `b` tensor.

#     Returns:
#         None : Fills in `out`
#     """

#     def _zip(
#         out,
#         out_shape,
#         out_strides,
#         a_storage,
#         a_shape,
#         a_strides,
#         b_storage,
#         b_shape,
#         b_strides,
#     ):
#         # TODO: Implement for Task 3.1.
#         if (
#             len(out_strides) != len(a_strides)
#             or (out_strides != a_strides).any()
#             or (out_shape != a_shape).any()
#             or len(out_strides) != len(b_strides)
#             or (out_strides != b_strides).any()
#             or (out_shape != b_shape).any()
#         ):
#             for i in prange(len(out)):
#                 # index for out tenser
#                 out_index = np.empty(MAX_DIMS, np.int32)
#                 a_index = np.empty(MAX_DIMS, np.int32)
#                 b_index = np.empty(MAX_DIMS, np.int32)
#                 count(i, out_shape, out_index)
#                 out_position = index_to_position(out_index, out_strides)
#                 broadcast_index(out_index, out_shape, a_shape, a_index)
#                 a_position = index_to_position(a_index, a_strides)
#                 broadcast_index(out_index, out_shape, b_shape, b_index)
#                 b_position = index_to_position(b_index, b_strides)
#                 out[out_position] = fn(
#                     a_storage[a_position], b_storage[b_position])
#         else:
#             for i in prange(len(out)):
#                 out[i] = fn(a_storage[i], b_storage[i])

#     return njit(parallel=True)(_zip)


# def zip(fn):
#     """
#     Higher-order tensor zip function ::

#       fn_zip = zip(fn)
#       c = fn_zip(a, b)

#     Args:
#         fn: function from two floats-to-float to apply
#         a (:class:`Tensor`): tensor to zip over
#         b (:class:`Tensor`): tensor to zip over

#     Returns:
#         :class:`Tensor` : new tensor
#     """
#     f = tensor_zip(njit()(fn))

#     def ret(a, b):
#         c_shape = shape_broadcast(a.shape, b.shape)
#         out = a.zeros(c_shape)
#         f(*out.tuple(), *a.tuple(), *b.tuple())
#         return out

#     return ret


# def tensor_reduce(fn):
#     """
#     NUMBA higher-order tensor reduce function.

#     Args:
#         fn: reduction function mapping two floats to float.
#         out (array): storage for `out` tensor.
#         out_shape (array): shape for `out` tensor.
#         out_strides (array): strides for `out` tensor.
#         a_storage (array): storage for `a` tensor.
#         a_shape (array): shape for `a` tensor.
#         a_strides (array): strides for `a` tensor.
#         reduce_shape (array): shape of reduction (1 for dimension kept, shape value for dimensions summed out)
#         reduce_size (int): size of reduce shape

#     Returns:
#         None : Fills in `out`

#     """

#     def _reduce(
#         out,
#         out_shape,
#         out_strides,
#         a_storage,
#         a_shape,
#         a_strides,
#         reduce_shape,
#         reduce_size,
#     ):

#         # TODO: Implement for Task 3.1.
#         for i in prange(len(out)):
#             out_index = np.empty(MAX_DIMS, np.int32)
#             a_index = np.empty(MAX_DIMS, np.int32)

#             count(i, out_shape, out_index)
#             out_position = index_to_position(out_index, out_strides)

#             for j in range(reduce_size):
#                 count(j, reduce_shape, a_index)
#                 for k in range(len(reduce_shape)):
#                     if reduce_shape[k] != 1:
#                         out_index[k] = a_index[k]
#                 a_position = index_to_position(out_index, a_strides)

#                 out[out_position] = fn(
#                     out[out_position], a_storage[a_position])

#     return njit(parallel=True)(_reduce)


# def reduce(fn, start=0.0):
#     """
#     Higher-order tensor reduce function. ::

#       fn_reduce = reduce(fn)
#       reduced = fn_reduce(a, dims)


#     Args:
#         fn: function from two floats-to-float to apply
#         a (:class:`Tensor`): tensor to reduce over
#         dims (list, optional): list of dims to reduce
#         out (:class:`Tensor`, optional): tensor to reduce into

#     Returns:
#         :class:`Tensor` : new tensor
#     """

#     f = tensor_reduce(njit()(fn))

#     def ret(a, dims=None, out=None):
#         old_shape = None
#         if out is None:
#             out_shape = list(a.shape)
#             for d in dims:
#                 out_shape[d] = 1
#             # Other values when not sum.
#             out = a.zeros(tuple(out_shape))
#             out._tensor._storage[:] = start
#         else:
#             old_shape = out.shape
#             diff = len(a.shape) - len(out.shape)
#             out = out.view(*([1] * diff + list(old_shape)))

#         # Assume they are the same dim
#         assert len(out.shape) == len(a.shape)

#         # Create a reduce shape / reduce size
#         reduce_shape = []
#         reduce_size = 1
#         for i, s in enumerate(a.shape):
#             if out.shape[i] == 1:
#                 reduce_shape.append(s)
#                 reduce_size *= s
#             else:
#                 reduce_shape.append(1)

#         # Apply
#         f(*out.tuple(), *a.tuple(), np.array(reduce_shape), reduce_size)

#         if old_shape is not None:
#             out = out.view(*old_shape)
#         return out

#     return ret


# @njit(parallel=True)
# def tensor_matrix_multiply(
#     out,
#     out_shape,
#     out_strides,
#     a_storage,
#     a_shape,
#     a_strides,
#     b_storage,
#     b_shape,
#     b_strides,
# ):
#     """
#     NUMBA tensor matrix multiply function.

#     Should work for any tensor shapes that broadcast as long as ::

#         assert a_shape[-1] == b_shape[-2]

#     Args:
#         out (array): storage for `out` tensor
#         out_shape (array): shape for `out` tensor
#         out_strides (array): strides for `out` tensor
#         a_storage (array): storage for `a` tensor
#         a_shape (array): shape for `a` tensor
#         a_strides (array): strides for `a` tensor
#         b_storage (array): storage for `b` tensor
#         b_shape (array): shape for `b` tensor
#         b_strides (array): strides for `b` tensor

#     Returns:
#         None : Fills in `out`
#     """

#     # TODO: Implement for Task 3.2.
#     inner = a_shape[-1]
#     for i in prange(len(out)):
#         out_index = np.empty(MAX_DIMS, np.int32)
#         a_start_index = np.empty(MAX_DIMS, np.int32)
#         b_start_index = np.empty(MAX_DIMS, np.int32)
#         count(i, out_shape, out_index)

#         count(i, out_shape, a_start_index)
#         a_start_index[len(out_shape) - 1] = 0
#         a_index = np.empty(MAX_DIMS, np.int32)
#         broadcast_index(a_start_index, out_shape, a_shape, a_index)
#         a_position = index_to_position(a_index, a_strides)

#         count(i, out_shape, b_start_index)
#         b_start_index[len(out_shape) - 2] = 0
#         b_index = np.empty(MAX_DIMS, np.int32)
#         broadcast_index(b_start_index, out_shape, b_shape, b_index)
#         b_position = index_to_position(b_index, b_strides)

#         # Reduce over a and b to fill in out_position
#         sum_out = 0
#         out_position = index_to_position(out_index, out_strides)
#         for k in range(inner):
#             # increment
#             new_a_position = a_position + k * a_strides[-1]
#             new_b_position = b_position + k * b_strides[-2]
#             sum_out += a_storage[new_a_position] * b_storage[new_b_position]
#         out[out_position] = sum_out


# def matrix_multiply(a, b):
#     """
#     Tensor matrix multiply

#     Should work for any tensor shapes that broadcast in the first n-2 dims and
#     have ::

#         assert a.shape[-1] == b.shape[-2]

#     Args:
#         a (:class:`Tensor`): tensor a
#         b (:class:`Tensor`): tensor b

#     Returns:
#         :class:`Tensor` : new tensor
#     """

#     # Create out shape
#     # START CODE CHANGE
#     ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
#     ls.append(a.shape[-2])
#     ls.append(b.shape[-1])
#     assert a.shape[-1] == b.shape[-2]
#     # END CODE CHANGE
#     out = a.zeros(tuple(ls))

#     # Call main function
#     tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())
#     return out


# class FastOps:
#     map = map
#     zip = zip
#     reduce = reduce
#     matrix_multiply = matrix_multiply
