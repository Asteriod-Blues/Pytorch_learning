import torch
from Answer import create_tensor_of_pi


x = create_tensor_of_pi(4, 5)

print('x is a tensor:', torch.is_tensor(x))
print('x has correct shape: ', x.shape == (4, 5))
print('x is filled with pi: ', (x == 3.14).all().item() == 1)