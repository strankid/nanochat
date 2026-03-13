"""Profile sparse24 components on A100 to find the bottleneck."""
import torch
import time

torch.set_float32_matmul_precision("high")

def bench(fn, warmup=10, iters=100, label=""):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:45s} {dt:8.3f} ms")
    return dt

device = "cuda"
dtype = torch.bfloat16

# d12 sizes: c_proj is (M, 3072) @ (3072, 768)
# M = batch * seq = 32 * 2048 = 65536
configs = [
    ("d12 M=65536", 65536, 3072, 768),
    ("d12 M=8192",  8192,  3072, 768),
    ("d26 M=65536", 65536, 6656, 1664),
    ("d26 M=8192",  8192,  6656, 1664),
]

from nanochat.sparse24 import (
    fast_sparse24_compress, apply_24_sparsity,
    _sparse24_compress_and_matmul,
    _CUTLASS_SPARSE_AVAILABLE, _CUSPARSELT_AVAILABLE, _TRITON_AVAILABLE,
)
print(f"CUTLASS sparse: {_CUTLASS_SPARSE_AVAILABLE}")
print(f"cuSPARSELt: {_CUSPARSELT_AVAILABLE}")
print(f"Triton: {_TRITON_AVAILABLE}")
print()

for name, M, K, N in configs:
    print(f"=== {name} (M={M}, K={K}, N={N}) ===")
    x = torch.randn(M, K, device=device, dtype=dtype)
    w = torch.randn(N, K, device=device, dtype=dtype)

    # 1. Dense matmul baseline
    bench(lambda: x @ w.t(), label="dense matmul (cuBLAS)")

    # 2. apply_24_sparsity + dense matmul (emulate)
    def emulate():
        xm = apply_24_sparsity(x)
        return xm @ w.t()
    bench(emulate, label="emulate (mask + dense matmul)")

    # 3. Just Triton compression kernel
    bench(lambda: fast_sparse24_compress(x), label="Triton compress only")

    # 4. Triton compress + CUTLASS sparse mm (our custom_op)
    if _CUTLASS_SPARSE_AVAILABLE:
        # Direct call without custom_op wrapper
        def cutlass_path():
            packed, meta = fast_sparse24_compress(x)
            return torch._sparse_semi_structured_mm(packed, meta, w.t())
        bench(cutlass_path, label="Triton compress + CUTLASS sparse mm")

        # Through custom_op
        bench(lambda: _sparse24_compress_and_matmul(x, w), label="custom_op (full path)")

        # Just CUTLASS sparse mm (pre-compressed)
        packed, meta = fast_sparse24_compress(x)
        bench(lambda: torch._sparse_semi_structured_mm(packed, meta, w.t()),
              label="CUTLASS sparse mm only (pre-compressed)")

    if _CUSPARSELT_AVAILABLE:
        from torch.sparse import to_sparse_semi_structured
        xm = apply_24_sparsity(x)
        xs = to_sparse_semi_structured(xm)
        bench(lambda: torch.mm(xs, w.t()), label="cuSPARSELt sparse mm (pre-converted)")

    print()
