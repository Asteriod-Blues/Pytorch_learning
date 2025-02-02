import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor

a = torch.tensor([1, 2, 3])
print(type(a))


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from Answer.py!')


def create_sample_tensor() -> Tensor:

    x = torch.zeros(3, 2)
    x[0, 1] = 10
    x[1, 0] = 100
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    for index, value in zip(indices, values):
        x[index] = value
    return x


def count_tensor_elements(x: Tensor) -> int:
    num_elements = 1
    shape = x.shape
    for element in shape:
        num_elements *= element
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    x = torch.full((M, N), 3.14)
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    assert start <= stop
    start1 = start
    while start1 % 10 != 0 and start1 <= stop:
        start1 += 1
    if start == start1:
        x = torch.tensor(0,)
    else:
        x = torch.arange(start1, stop + 1, 10, dtype=torch.float64)
    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    last_row = x[-1, :]
    third_col = x[:, 2:3]
    first_two_rows_three_cols = x[0:2, 0:3]
    even_rows_odd_cols = x[::2, 1::2]
    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    x[0:1, :6] = torch.tensor([0, 1, 2, 2, 2, 2])
    x[1:2, :6] = torch.tensor([0, 1, 2, 2, 2, 2])
    x[2:3, :6] = torch.tensor([3, 4, 3, 4, 5, 5])
    x[3:4, :6] = torch.tensor([3, 4, 3, 4, 5, 5])
    return x


def shuffle_cols(x: Tensor) -> Tensor:
    m = x.shape[0]
    y = torch.zeros((m, 4))
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 0]
    y[:, 2] = x[:, 2]
    y[:, 3] = x[:, 1]
    return y


def reverse_rows(x: Tensor) -> Tensor:
    m = x.shape[0]
    n = x.shape[1]
    y = torch.zeros(m, n)
    i = 0
    m -= 1
    while m >= 0:
        y[i] = x[m]
        m -= 1
        i += 1
    return y


def take_one_elem_per_col(x: Tensor) -> Tensor:
    y = torch.tensor([0, 0, 0])
    y[0] = x[1, 0]
    y[1] = x[0, 1]
    y[2] = x[3, 2]
    return y


def make_one_hot(x: List[int]) -> Tensor:
    n = len(x)
    c = max(x) + 1
    y = torch.zeros(n, c, dtype=torch.float32)
    idx1 = list(range(n))
    y[idx1, x] = 1
    return y


def sum_positive_entries(x: Tensor) -> Tensor:
    mask = x > 0
    pos_sum = torch.sum(x[mask])
    return pos_sum


def reshape_practice(x: Tensor) -> Tensor:
    x1 = x.view(2, 3, 4)
    x2 = x1.transpose(0, 1)
    y = x2.reshape(3, 8)
    return y


def zero_row_min(x: Tensor) -> Tensor:
    #  这里用布尔掩码也可以，建议尝试下
    y = x.clone()
    indexOfminimum = y.argmin(dim=1)
    rows = torch.arange(x.shape[0])
    y[rows, indexOfminimum] = 0
    return y


def batched_matrix_multiply(
    x: Tensor, y: Tensor, use_loop: bool = True
) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    Depending on the value of use_loop, this calls to either
    batched_matrix_multiply_loop or batched_matrix_multiply_noloop to perform
    the actual computation. You don't need to implement anything here.

    Args:
        x: Tensor of shape (B, N, M)
        y: Tensor of shape (B, M, P)
        use_loop: Whether to use an explicit Python loop.

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)


def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    b = x.shape[0]
    n = x.shape[1]
    p = y.shape[2]
    z = torch.zeros((b, n, p), dtype=x.dtype)
    for i in range(b):
        z[i] = x[i].matmul(y[i])
    return z


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    z = torch.bmm(x, y)
    return z


def normalize_columns(x: Tensor) -> Tensor:
    m, n = x.shape
    column_sums = x.sum(dim=0)
    column_means = column_sums / m
    squared_column_sums = torch.sum(x ** 2, dim=0)
    variances = (squared_column_sums - m * column_means ** 2) / (m - 1)  # 要用样本方差，而不是总体方差
    stds = torch.sqrt(variances)
    y = (x - column_means) / stds
    return y


def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    y = x.mm(w)
    return y


def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    x_gpu = x.cuda()
    w_gpu = w.cuda()
    y_gpu = torch.matmul(x_gpu, w_gpu)
    y = y_gpu.cpu()
    return y
