# 导入必要的库
import torch
# 导入类型提示作用的库（可省略）
from typing import List, Tuple
from torch import Tensor
# 检查环境是否安装成功
print(torch.cuda.is_available())

# Tensor

# Creating
a = torch.tensor([1, 2, 3])
print("Here is a : ", end='')
print(a)
print("Rank of a : ", a.dim())  # dimension details
print("Shape of a : ", a.shape)
# Accessing
print("First element : ", a[0])
# Mutate
a[0] = 4
print("The first one after mutating : ", a[0])
# Multidimensional
b = torch.tensor([[1, 2], [3, 4]])
print("Rank of b : ", b.dim())  # dimension details
print("Shape of b : ", b.shape)
print("b[0, 1] : ", b[0, 1])
b[0, 1] = 4
print("b[0, 1] after mutate", b[0, 1])
# Print
print("a is:", a)
# Complete the implementation of the functions: create_sample_tensor, mutate_tensor, and count_tensor_elements

# Tensor constructors
c = torch.zeros(3, 2)
print('tensor of zeros:')
print(c)
d = torch.ones(3, 2)
print('tensor of ones:')
print(d)
e = torch.eye(3)  # 特征向量
print('\nidentity matrix:')
print(e)
f = torch.rand(4, 5)
print('\nrandom tensor:')
print(f)
# Complete the implementation of create_tensor_of_pi to practice using a tensor constructor.

# Data type
# Force a particular datatype
y0 = torch.tensor([1, 2], dtype=torch.float32)  # 32-bit float
y1 = torch.tensor([1, 2], dtype=torch.int32)    # 32-bit (signed) integer
y2 = torch.tensor([1, 2], dtype=torch.int64)    # 64-bit (signed) integer
print('\ndtype when we force a datatype:')
print('32-bit float: ', y0.dtype)
print('32-bit integer: ', y1.dtype)
print('64-bit integer: ', y2.dtype)
# Change datatype
y3 = y0.double()
y4 = y1.float()
y5 = y2.to(torch.float64)
print("y3:", y3.dtype)
print("y4:", y4.dtype)
print("y5:", y5.dtype)
#  Create a tensor with the same datatype as another tensor
x0 = torch.eye(3, dtype=torch.float32)
x1 = torch.zeros_like(x0)  # 类型和形状都相同的x1
x2 = x0.new_zeros([3, 2])  # 类型相同形状自定义
x3 = x0.to(y1)  # 将x0的数据类型变得和y1相同
print("x0:", x0.dtype)
print("x1:", x1.dtype)
print("x2:", x2.dtype)
print("x3:", x3.dtype)
#  创建一个等差数列
g = torch.arange(1, 11, 5)   # start,end,step
print(g)
#  Implement the function multiples_of_ten
