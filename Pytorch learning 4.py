import torch
# Out-of-place vs In-place operators
# Out-of-place addition creates and returns a new tensor without modifying the inputs:
x = torch.tensor([1, 2, 3])
y = torch.tensor([3, 4, 5])
print('Out-of-place addition:')
print('Before addition:')
print('x: ', x)
print('y: ', y)
z = x.add(y)  # Same as z = x + y or z = torch.add(x, y)
print('\nAfter addition (x and y unchanged):')
print('x: ', x)
print('y: ', y)
print('z: ', z)
print('z is x: ', z is x)
print('z is y: ', z is y)

# In-place addition modifies the input tensor:
print('\n\nIn-place Addition:')
print('Before addition:')
print('x: ', x)
print('y: ', y)
x.add_(y)  # Same as x += y or torch.add(x, y, out=x)
print('\nAfter addition (x is modified):')
print('x: ', x)
print('y: ', y)
print('z: ', z)
print('z is x: ', z is x)
print('z is y: ', z is y)

# Avoid using In-place operators,avoid unexpected mistakes!

# Running on GPU
# Check it is available or not
if torch.cuda.is_available():
  print('PyTorch can use GPUs!')
else:
  print('PyTorch cannot use GPUs.')
# Change devices between CPU and GPU
x0 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
x1 = x0.cuda()
x2 = x0.cpu()
print("Original device:", x0.device)
print("Device of X1:", x1.device)
print("Device of X2:", x2.device)
y = torch.tensor([1, 2, 3], dtype=torch.float64, device='cuda')
x3 = x0.to(y)
print("X3 will have the same dtype and device with y:")
print("Dtype:", x3.dtype)
print("Device:", x3.device)
#  Check CPU VS GPU
#  Implement function mm_on_Gpu

