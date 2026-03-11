"""Quick MLP forward+backward benchmark: dense vs sparse24 on WSL.

Tests the full MLP (c_fc + relu^2 + c_proj) to measure realistic speedup.
No torch.compile, no training loop — just raw MLP throughput.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/mnt/c/Users/Apoorv/projects/nanochat')

torch.set_float32_matmul_precision('high')

class DenseMLP(nn.Module):
    def __init__(self, n_embd, hidden):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

def bench(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000
    return elapsed

def main():
    device = 'cuda'
    dtype = torch.bfloat16

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Compute capability: {torch.cuda.get_device_capability()}")
    print(f"cuSPARSELt: {torch.backends.cusparselt.is_available()}")
    print()

    from nanochat.sparse24 import Sparse24MLP

    configs = [
        # (n_embd, hidden, M, label)
        (384,  1536, 2048, "d6  (M=2048)"),
        (768,  3072, 4096, "d12 (M=4096)"),
        (1536, 6144, 4096, "d24 (M=4096)"),
    ]

    for n_embd, hidden, M, label in configs:
        print(f"=== {label}: embd={n_embd}, hidden={hidden} ===")

        dense_mlp = DenseMLP(n_embd, hidden).to(device, dtype)
        sparse_mlp = Sparse24MLP(
            dense_mlp.c_fc, dense_mlp.c_proj, emulate=False
        ).to(device, dtype)

        x = torch.randn(M, n_embd, device=device, dtype=dtype, requires_grad=True)

        # Forward only
        dense_fwd = bench(lambda: dense_mlp(x))
        sparse_fwd = bench(lambda: sparse_mlp(x))
        print(f"  Fwd:  dense={dense_fwd:.3f}ms  sparse={sparse_fwd:.3f}ms  -> {dense_fwd/sparse_fwd:.2f}x")

        # Forward + backward
        def run_fwd_bwd(mlp):
            y = mlp(x)
            loss = y.sum()
            loss.backward()

        dense_fb = bench(lambda: run_fwd_bwd(dense_mlp))
        sparse_fb = bench(lambda: run_fwd_bwd(sparse_mlp))
        print(f"  F+B:  dense={dense_fb:.3f}ms  sparse={sparse_fb:.3f}ms  -> {dense_fb/sparse_fb:.2f}x")
        print()

if __name__ == "__main__":
    main()
