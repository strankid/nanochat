"""Benchmark dense vs sparse24 with our Triton fused kernel on WSL.

Tests the full pipeline: Triton compress -> CUTLASS sparse matmul.
Also tests more M sizes to find the crossover point.
"""
import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, '/mnt/c/Users/Apoorv/projects/nanochat')

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
    print(f"CUTLASS sparse mm: {hasattr(torch, '_sparse_semi_structured_mm')}")
    print(f"cuSPARSELt: {torch.backends.cusparselt.is_available()}")
    print()

    from nanochat.sparse24 import fast_sparse24_compress
    from torch.sparse import to_sparse_semi_structured

    # Test our d6 hidden dim (1536) at various batch sizes
    K = 1536  # hidden dim (4 * n_embd for d6 with embd=384)
    N = 384   # output dim (n_embd)

    print(f"=== Varying M (batch*seq), K={K}, N={N} ===")
    print(f"{'M':>6} | {'Dense':>8} | {'cuSPARSELt':>10} | {'CUTLASS':>8} | {'Speedup(cslt)':>13} | {'Speedup(cut)':>12}")
    print("-" * 75)

    for M in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
        x = F.relu(torch.randn(M, K, device=device, dtype=dtype)).square()
        x_24 = apply_24(x)
        weight = torch.randn(N, K, device=device, dtype=dtype)

        # Dense
        dense_ms = bench(lambda: x @ weight.t())

        # cuSPARSELt path
        try:
            x_sparse = to_sparse_semi_structured(x_24)
            cslt_ms = bench(lambda: torch.mm(x_sparse, weight.t()))
            cslt_speedup = dense_ms / cslt_ms
        except Exception as e:
            cslt_ms = float('nan')
            cslt_speedup = float('nan')

        # CUTLASS path (our Triton compress + _sparse_semi_structured_mm)
        try:
            if M >= 32 and K >= 64 and K % 16 == 0 and M % 32 == 0:
                packed, meta = fast_sparse24_compress(x)
                cut_ms = bench(lambda: torch._sparse_semi_structured_mm(
                    *fast_sparse24_compress(x), weight.t()))
                cut_speedup = dense_ms / cut_ms
            else:
                cut_ms = float('nan')
                cut_speedup = float('nan')
        except Exception as e:
            cut_ms = float('nan')
            cut_speedup = float('nan')
            print(f"  CUTLASS error at M={M}: {e}")

        print(f"{M:>6} | {dense_ms:>7.3f}ms | {cslt_ms:>9.3f}ms | {cut_ms:>7.3f}ms | {cslt_speedup:>12.2f}x | {cut_speedup:>11.2f}x")

    # Also test d12 dims where we saw speedup
    print()
    K2 = 3072
    N2 = 768
    print(f"=== Varying M, K={K2}, N={N2} (d12 scale) ===")
    print(f"{'M':>6} | {'Dense':>8} | {'cuSPARSELt':>10} | {'Speedup':>8}")
    print("-" * 45)

    for M in [256, 512, 1024, 2048, 4096]:
        x = F.relu(torch.randn(M, K2, device=device, dtype=dtype)).square()
        x_24 = apply_24(x)
        weight = torch.randn(N2, K2, device=device, dtype=dtype)
        dense_ms = bench(lambda: x @ weight.t())
        x_sparse = to_sparse_semi_structured(x_24)
        cslt_ms = bench(lambda: torch.mm(x_sparse, weight.t()))
        print(f"{M:>6} | {dense_ms:>7.3f}ms | {cslt_ms:>9.3f}ms | {dense_ms/cslt_ms:>7.2f}x")

if __name__ == "__main__":
    main()
