import torch
from Answer import sum_positive_entries
# Make a few test cases
torch.manual_seed(598)
x0 = torch.tensor([[-1, -1, 0], [0, 1, 2], [3, 4, 5]])
x1 = torch.tensor([-100, 0, 1, 2, 3])
x2 = torch.randn(100, 100).long()
print('Correct for x0: ', sum_positive_entries(x0) == 15)
print('Correct for x1: ', sum_positive_entries(x1) == 6)
print('Correct for x2: ', sum_positive_entries(x2) == 1871)