import minitorch
from hypothesis import given
from .strategies import tensors
import pytest


@pytest.mark.task4_1
def test_conv1d_simple():
    t = minitorch.tensor_fromlist([0, 1, 2, 3]).view(1, 1, 4)
    t.requires_grad_(True)
    t2 = minitorch.tensor_fromlist([[1, 2, 3]]).view(1, 1, 3)
    out = minitorch.Conv1dFun.apply(t, t2)

    assert out[0, 0, 0] == 0 * 1 + 1 * 2 + 2 * 3
    assert out[0, 0, 1] == 1 * 1 + 2 * 2 + 3 * 3
    assert out[0, 0, 2] == 2 * 1 + 3 * 2
    assert out[0, 0, 3] == 3 * 1


@pytest.mark.task4_1
def test_conv1d_simple2():
    t = minitorch.tensor_fromlist([[1, 2, 1, 2, 3, 4],
                                   [2, 1, 3, 2, 3, 2.]]).view(1, 2, 6)
    t.requires_grad_(True)
    t2 = minitorch.tensor_fromlist([[[3, 2, 1],
                                     [1, 2, 3.]], [[3, 2, 1],
                                                   [1, 2, 3.]]]).view(2, 2, 3)
    out = minitorch.Conv1dFun.apply(t, t2)
    print(out)
    assert out[0, 1, 2] == 26
    minitorch.grad_check(minitorch.Conv1dFun.apply, t, t2)


@pytest.mark.task4_1
@given(tensors(shape=(1, 2, 6)), tensors(shape=(2, 2, 4)))
def test_conv1d(input, weight):
    out = minitorch.Conv1dFun.apply(input, weight)
    minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)
    print('=============')
    print(input, weight)
    print(out)


@pytest.mark.task4_1
@given(tensors(shape=(2, 2, 6)), tensors(shape=(3, 2, 2)))
def test_conv1d_channel(input, weight):
    # Run several times for random seed
    for _ in range(5):
        minitorch.grad_check(minitorch.Conv1dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv(input, weight):
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 4)))
def test_conv_batch(input, weight):
    minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 4)))
def test_conv_channel(input, weight):
    # Run several times for random seed
    for _ in range(5):
        minitorch.grad_check(minitorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_2
def test_conv2():
    t = minitorch.tensor_fromlist(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    ).view(1, 1, 4, 4)
    t.requires_grad_(True)

    t2 = minitorch.tensor_fromlist([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = minitorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    minitorch.grad_check(minitorch.Conv2dFun.apply, t, t2)
