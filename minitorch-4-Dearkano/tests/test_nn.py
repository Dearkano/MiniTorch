import minitorch
from hypothesis import given
from .strategies import tensors, assert_close
import pytest


def exp_sum(f1, f2):
    return f1 + minitorch.operators.exp(f2)


exp_reduce = minitorch.FastOps.reduce(exp_sum)
exp_map = minitorch.FastOps.map(minitorch.operators.exp)


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = minitorch.avgpool2d(t, (2, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t):
    out = minitorch.nn.max(t, 2)
    assert out[0, 0, 0] == max(t[0, 0, i] for i in range(4))
    out = minitorch.nn.max(t, 1)
    assert out[0, 0, 0] == max(t[0, i, 0] for i in range(3))
    out = minitorch.nn.max(t, 0)
    assert out[0, 0, 0] == max(t[i, 0, 0] for i in range(2))
    rand_tensor = minitorch.rand(t.shape) * 1e-5
    t = t + rand_tensor
    minitorch.grad_check(lambda t: minitorch.nn.max(t, 2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t):
    out = minitorch.maxpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j]
                              for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j]
                              for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j]
                              for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t):
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    t = minitorch.tensor_fromlist([
        [
            [
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00],
                [0.00, 0.00, 0.00, 0.00]]]])
    q = minitorch.softmax(t, 2)
    print('=====t')
    print(t)
    print('=====q')
    print(q)
    x = q.sum(dim=3)
    print('=====x')
    print(x)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)
