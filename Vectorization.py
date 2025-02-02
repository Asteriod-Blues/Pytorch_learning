import time
import torch
import matplotlib.pyplot as plt
from Answer import batched_matrix_multiply

N, M, P = 64, 64, 64
loop_times = []
no_loop_times = []
with_loop_slowdown = []
Bs = list(range(4, 128, 4))
num_trials = 20
for B in Bs:
    loop_trials = []
    no_loop_trials = []
    for trial in range(num_trials):
        x = torch.randn(B, N, M)
        y = torch.randn(B, M, P)
        t0 = time.time()
        z1 = batched_matrix_multiply(x, y, use_loop=True)
        t1 = time.time()
        z2 = batched_matrix_multiply(x, y, use_loop=False)
        t2 = time.time()
        loop_trials.append(t1 - t0)
        no_loop_trials.append(t2 - t1)
    loop_mean = torch.tensor(loop_trials).mean().item()
    no_loop_mean = torch.tensor(no_loop_trials).mean().item()
    loop_times.append(loop_mean)
    no_loop_times.append(no_loop_mean)
    with_loop_slowdown.append(no_loop_mean / loop_mean)

plt.subplot(1, 2, 1)
plt.plot(Bs, loop_times, 'o-', label='use_loop=True')
plt.plot(Bs, no_loop_times, 'o-', label='use_loop=False')
plt.xlabel('Batch size B')
plt.ylabel('Runtime (s)')
plt.legend(fontsize=14)
plt.title('Loop vs Vectorized speeds')

plt.subplot(1, 2, 2)
plt.plot(Bs, with_loop_slowdown, '-o')
plt.title('No Vectorized slowdown')
plt.xlabel('Batch size B')
plt.ylabel('No Vectorized slowdown')

plt.gcf().set_size_inches(12, 4)
plt.show()