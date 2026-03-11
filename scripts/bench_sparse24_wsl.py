"""Benchmark dense vs sparse24 MLP matmul on RTX 3060 via WSL.

Measures the c_proj matmul (hidden -> embed) which is the target for
2:4 activation sparsity speedup.
"""
import torch
import torch.nn.functional as F
import time

torch.set_float32_matmul_precision('high')

def apply_24(x):
    flat = x.reshape(-1, 4)
    _, idx = torch.topk(flat.abs(), k=2, dim=-1)
    mask = torch.zeros_like(flat)
    mask.scatter_(-1, idx, 1.0)
    return (flat * mask).reshape(x.shape)

def bench(fn, warmup=20, iters=200):
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
    print(f"cuSPARSELt available: {torch.backends.cusparselt.is_available()}")
    print()

    # Model dimensions matching d6 config
    # c_proj: (B*T, 4*n_embd) @ (n_embd, 4*n_embd).T -> (B*T, n_embd)
    configs = [
        # (M, K, N, label)
        (2048, 1536, 384, "d6: B*T=2048, hidden=1536, embd=384"),
        (4096, 3072, 768, "d12: B*T=4096, hidden=3072, embd=768"),
        (4096, 6144, 1536, "d24: B*T=4096, hidden=6144, embd=1536"),
    ]

    from torch.sparse import to_sparse_semi_structured

    for M, K, N, label in configs:
        print(f"--- {label} ---")
        # Activation after ReLU^2 (naturally ~50% sparse)
        x = F.relu(torch.randn(M, K, device=device, dtype=dtype)).square()
        x_24 = apply_24(x)
        weight = torch.randn(N, K, device=device, dtype=dtype)

        # Dense matmul
        dense_ms = bench(lambda: x @ weight.t())

        # Sparse matmul via cuSPARSELt
        x_sparse = to_sparse_semi_structured(x_24)
        sparse_ms = bench(lambda: torch.mm(x_sparse, weight.t()))

        speedup = dense_ms / sparse_ms
        print(f"  Dense:    {dense_ms:.3f} ms")
        print(f"  Sparse24: {sparse_ms:.3f} ms")
        print(f"  Speedup:  {speedup:.2f}x")
        print()

if __name__ == "__main__":
    main()
