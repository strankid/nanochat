"""Test 2:4 sparse matmul on WSL (Linux PyTorch build)."""
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "No CUDA"
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute capability: {torch.cuda.get_device_capability()}")
print()

has_cutlass = hasattr(torch, '_sparse_semi_structured_mm')
has_cusparselt = hasattr(torch.backends, 'cusparselt') and torch.backends.cusparselt.is_available()
print(f"CUTLASS sparse mm available: {has_cutlass}")
print(f"cuSPARSELt available: {has_cusparselt}")
print()

def apply_24(x):
    flat = x.reshape(-1, 4)
    _, idx = torch.topk(flat.abs(), k=2, dim=-1)
    mask = torch.zeros_like(flat)
    mask.scatter_(-1, idx, 1.0)
    return (flat * mask).reshape(x.shape)

x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
x_sparse = apply_24(x)

# Test 1: to_sparse_semi_structured
try:
    from torch.sparse import to_sparse_semi_structured
    result = to_sparse_semi_structured(x_sparse)
    print(f"to_sparse_semi_structured: SUCCESS! type={type(result).__name__}")
except Exception as e:
    print(f"to_sparse_semi_structured: FAILED: {e}")
    result = None

# Test 2: sparse matmul
if result is not None:
    try:
        weight = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        y_dense = x_sparse @ weight.t()
        y_sparse = torch.mm(result, weight.t())
        diff = (y_dense - y_sparse).abs().max().item()
        print(f"Sparse matmul: SUCCESS! max diff vs dense: {diff:.6f}")
    except Exception as e:
        print(f"Sparse matmul: FAILED: {e}")
