import torch
# Tensor indexing
a = torch.tensor([0, 11, 22, 33, 44, 55])
print("Start and stop:", a[2:4])
print("Only with start:", a[2:])
print("Only with stop:", a[:5])
print("With slice:", a[0:5:2])
print("Last one:", a[:-1])
# Access the single row or column
b = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print("Single row:", b[1])  # b[1:2, 0:]
print("Single column:", b[0, 1:2])
print(b.shape)
# Make a copy
c = b[0, 1:]
d = b[0, 1:].clone()
print("Before mutating:")
print(b)
print(c)
print(d)
b[0, 1:] = 20
c[1] = 30
d[2] = 40
print("After mutating:")
print(b)
print(c)
print(d)
# Modifying
e = torch.zeros(3, 4, dtype=torch.float64)
print("Original:", e)
e[:, :2] = 1
e[:, 2:] = torch.tensor([[2, 3], [5, 6], [8, 9]])
print(e)
# Implement the function slice_indexing_practice, slice_assignment_practice

# Using index arrays to index tensors
f = torch.tensor([[1, 2, 3], [5, 6, 7], [9, 10, 11]])
print("Original:", f)
# Reordering rows
idx1 = [0, 0, 1, 2, 2]
print("After reordering rows:", f[idx1])
# Reordering column
idx2 = [2, 1, 0]
print("After reordering col:", f[:, idx2])
# Get or modify diagonal
idx3 = [0, 1, 2]
print("Original diagonal:", f[idx3, idx3])
# Same value
f[idx3, idx3] = 0
# Different value
f[idx3, idx3] = torch.tensor([99, 99, 99])
print("Modified diagonal:", f)
# Implement the functions shuffle_cols, reverse_rows, and take_one_elem_per_col, and make_one_hot

# Boolean tensor indexing
g = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original tensor:", g)
# Boolean mask,select the elements greater than 3
mask = g > 3
print("Boolean mask tensor:", mask)
# Access the elements selected by mask
print("After selected:", g[mask])
# To modify the target elements
g[mask] = 0
g[g <= 3] = 0
print("Modified one:", g)
# Sum up all elements in tensor
print("Sum:", torch.sum(g))
# implement the function sum_positive_entries

# Reshaping operations
# VIEW function
x0 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
print("Original tensor:", x0)
print("Original shape:", x0.shape)
x1 = x0.view(8)
print("Flattened tensor:", x1)
print("Flattened shape:", x1.shape)
x2 = x0.view(2, 2, 2)
print("Rank 3 tensor:", x2)
print("Rank 3 shape:", x2.shape)
x3 = x0.view(-1, 8)
print("Using -1 to auto compute the number needed:", x3)
print("Shape:", x3.shape)


def flatten(x):
    return x.view(-1)


print("Using flatten function:", flatten(x0))
print("Flatten function shape:", x0.shape)
x0[0, 0] = 99
x1[0] = 1
print("They share the memory:", x0)
print("NOTHING GONNA CHANGE")
