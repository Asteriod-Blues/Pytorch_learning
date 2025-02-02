import torch
# Swapping axes
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original:", a)
print("Trying to use view:", a.view(3, 2))
print("Transposed matrix:")
print(a.t())
print(torch.t(a))
x0 = torch.tensor([
     [[1,  2,  3,  4],
      [5,  6,  7,  8],
      [9, 10, 11, 12]],
     [[13, 14, 15, 16],
      [17, 18, 19, 20],
      [21, 22, 23, 24]]])
print("Original matrix:", x0)
print("Original shape:", x0.shape)
x1 = x0.transpose(1, 2)
print("Swap axes 1 and 2:", x1)
print("Shape of tensor after swap axes 1 and 2:", x1.shape)
x2 = x0.permute(2, 1, 0)
print("Swap multiple axes at the same time:", x2)
print("Shape of tensor x2:", x2.shape)
#  Implement the function reshape_practice

# Tensor computation
x = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
y = torch.tensor([[5, 6, 7, 8]], dtype=torch.float32)

# Elementwise sum; all give the same result
print('Elementwise sum:')
print(x + y)
print(torch.add(x, y))
print(x.add(y))

# Elementwise difference
print('\nElementwise difference:')
print(x - y)
print(torch.sub(x, y))
print(x.sub(y))

# Elementwise product
print('\nElementwise product:')
print(x * y)
print(torch.mul(x, y))
print(x.mul(y))

# Elementwise division
print('\nElementwise division')
print(x / y)
print(torch.div(x, y))
print(x.div(y))

# Elementwise power
print('\nElementwise power')
print(x ** y)
print(torch.pow(x, y))
print(x.pow(y))

print('Square root:')
print(torch.sqrt(x))
print(x.sqrt())

print('\nTrig functions:')
print(torch.sin(x))
print(x.sin())
print(torch.cos(x))
print(x.cos())
# Reduction operation
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Original:", b)
print("Sum of all elements in b:", b.sum())
print("Sum up in one dimension:", b.sum(dim=0))  # 对每一列求和

# Min
c = torch.tensor([[2, 3, 4, 2], [3, 3, 3, 2]], dtype=torch.float32)
print("Minimum number in all dimensions:", c.min())
col_min_vals, col_min_idx = c.min(dim=0)
print("Minimum number in each column:")
print("Values:", col_min_vals)
print("Indexes:", col_min_idx)
print("Only indexes of minimum in one dimension", c.argmin(dim=0))

# Mean
print("Mean of c in dimension 1:", c.mean(dim=1, keepdim=True))
# Clone
d = c.clone()
print("This is c:", c)
print("This is d:", d)
print("Nothing is different.")
#  Implement the function zero_row_min

# Matrix operations
v = torch.tensor([6, 7], dtype=torch.float32)
w = torch.tensor([9, 10], dtype=torch.float32)
print("Inner product of vectors:", v.dot(w))
d = torch.tensor([[2, 3], [4, 5]], dtype=torch.float32)
e = torch.tensor([[6, 7], [8, 9]], dtype=torch.float32)
print("Matrix-matrix product:", d.mm(e))
print("Here is vector v:", v)
print("Here is matrix d:", d)
print("Matrix-vector product with mv:", d.mv(v))
print("Matrix-vector product with view and mm:", d.mm(v.view(2, 1)))
print("Auto calculate:", d.matmul(v))
f = torch.randn(2, 3, 4)
g = torch.randn(2, 4, 6)
h = torch.bmm(f, g)
print("Batched matrix multiply:", h)
# Implement function batched_matrix_multiply

# 运行vectorization.py去比较向量化代码和非向量化代码运行速度的差距

# Implement the function normalize_columns
