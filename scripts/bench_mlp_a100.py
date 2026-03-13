"""Benchmark MLP forward+backward: dense vs sparse24, compiled vs not."""
import torch
import torch.nn as nn
import time

torch.set_float32_matmul_precision("high")

from nanochat.sparse24 import Sparse24MLP, _sparse24_forward

device = "cuda"
dtype = torch.bfloat16
B, T = 32, 2048


def bench(fn, label, warmup=5, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:40s} {dt:8.3f} ms")
    return dt


for depth, dim in [(12, 768), (26, 1664)]:
    print(f"\n{'='*60}")
    print(f"  depth={depth}, dim={dim}, B={B}, T={T}")
    print(f"{'='*60}")

    c_fc = nn.Linear(dim, 4*dim, bias=False, device=device, dtype=dtype)
    c_proj = nn.Linear(4*dim, dim, bias=False, device=device, dtype=dtype)

    def dense_step():
        x = torch.randn(B*T, dim, device=device, dtype=dtype, requires_grad=True)
        h = torch.nn.functional.relu(c_fc(x)).square()
        y = c_proj(h)
        y.sum().backward()
        return y

    def sparse_step():
        x = torch.randn(B*T, dim, device=device, dtype=dtype, requires_grad=True)
        h = torch.nn.functional.relu(c_fc(x)).square()
        K = h.shape[-1]
        x_2d = h.reshape(-1, K)
        y_2d = _sparse24_forward(x_2d, c_proj.weight)
        y = y_2d.reshape(B*T, -1)
        y.sum().backward()
        return y

    dense_compiled = torch.compile(dense_step, dynamic=False)
    sparse_compiled = torch.compile(sparse_step, dynamic=False)

    d_eager = bench(dense_step, "dense (eager)")
    s_eager = bench(sparse_step, "sparse24 (eager)")
    ratio_eager = s_eager / d_eager
    print(f"  {'sparse/dense ratio (eager)':40s} {ratio_eager:8.2f}x")

    print("  --- compiling (may take a minute) ---")
    d_comp = bench(dense_compiled, "dense (compiled)")
    s_comp = bench(sparse_compiled, "sparse24 (compiled)")
    ratio_comp = s_comp / d_comp
    print(f"  {'sparse/dense ratio (compiled)':40s} {ratio_comp:8.2f}x")
