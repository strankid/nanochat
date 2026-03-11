"""
2:4 Activation Sparsity for nanochat.

Exploits the natural sparsity of ReLU^2 activations to accelerate FFN matmuls
using NVIDIA's semi-structured (2:4) sparse tensor cores.

Architecture:
  1. Triton kernel fuses 2:4 mask + CUTLASS compression in one pass
  2. Construct SparseSemiStructuredTensorCUTLASS from pre-computed data
  3. CUTLASS sparse matmul uses hardware sparse tensor cores (~2x speedup)

The compression is bandwidth-limited at ~0.1ms (H100) vs 5-7ms for PyTorch's
Python-based compression, making activation sparsity viable for the first time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

# Check for CUTLASS sparse matmul support
_CUTLASS_SPARSE_AVAILABLE = False
_CUSPARSELT_AVAILABLE = False
try:
    if torch.cuda.is_available():
        # Test if CUTLASS sparse mm kernel is compiled
        _CUTLASS_SPARSE_AVAILABLE = hasattr(torch, '_sparse_semi_structured_mm')
        # Test if cuSPARSELt is available
        if hasattr(torch.backends, 'cusparselt'):
            _CUSPARSELT_AVAILABLE = torch.backends.cusparselt.is_available()
except Exception:
    pass


# ---------------------------------------------------------------------------
# Triton kernel: fused 2:4 mask + CUTLASS compression
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _fused_mask_compress_kernel(
        X_ptr,        # (M * K,) bf16 flat — input activation
        PACKED_ptr,   # (M * K//2,) bf16 flat — compressed values
        META_ptr,     # (M * meta_ncols,) int16 flat — metadata (natural order)
        total_groups,  # M * K // 4
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fused 2:4 masking + CUTLASS format compression.

        1D grid over groups of 4 elements in flat memory order.
        Each program handles BLOCK groups. Adjacent programs read adjacent memory.
        This maximizes memory coalescing since adjacent threads read adjacent addresses.
        """
        pid = tl.program_id(0)
        gids = pid * BLOCK + tl.arange(0, BLOCK)  # group indices
        mask = gids < total_groups

        # Load 4 contiguous bf16 values per group
        base = gids * 4
        v0 = tl.load(X_ptr + base, mask=mask, other=0.0)
        v1 = tl.load(X_ptr + base + 1, mask=mask, other=0.0)
        v2 = tl.load(X_ptr + base + 2, mask=mask, other=0.0)
        v3 = tl.load(X_ptr + base + 3, mask=mask, other=0.0)

        a0 = tl.abs(v0)
        a1 = tl.abs(v1)
        a2 = tl.abs(v2)
        a3 = tl.abs(v3)

        # Ranking: lower index breaks ties
        w0 = ((a0 >= a1).to(tl.int32) + (a0 >= a2).to(tl.int32)
               + (a0 >= a3).to(tl.int32))
        w1 = ((a1 > a0).to(tl.int32) + (a1 >= a2).to(tl.int32)
               + (a1 >= a3).to(tl.int32))
        w2 = ((a2 > a0).to(tl.int32) + (a2 > a1).to(tl.int32)
               + (a2 >= a3).to(tl.int32))
        w3 = 6 - w0 - w1 - w2

        k0 = w0 >= 2
        k1 = w1 >= 2
        k2 = w2 >= 2
        k3 = w3 >= 2

        z = tl.full(k0.shape, 0, tl.int32)
        i0 = tl.where(k0, z, tl.where(k1, z + 1, tl.where(k2, z + 2, z + 3)))
        i1 = tl.where(k3, z + 3, tl.where(k2, z + 2, tl.where(k1, z + 1, z)))

        val0 = tl.where(k0, v0, tl.where(k1, v1, tl.where(k2, v2, v3)))
        val1 = tl.where(k3, v3, tl.where(k2, v2, tl.where(k1, v1, v0)))

        # packed output: 2 values per group, contiguous within a row
        # group gid corresponds to row = gid // (K//4), group_in_row = gid % (K//4)
        groups_per_row: tl.constexpr = K // 4
        row = gids // groups_per_row
        gir = gids % groups_per_row  # group index within row
        packed_base = row * (K // 2) + gir * 2
        tl.store(PACKED_ptr + packed_base, val0, mask=mask)
        tl.store(PACKED_ptr + packed_base + 1, val1, mask=mask)

        # metadata: 4 bits per group, 4 groups per int16 meta element
        # meta column = gir // 4, bit position = (gir % 4) * 4
        meta_ncols: tl.constexpr = K // 16
        meta_col = gir // 4
        bit_pos = (gir % 4) * 4
        bits = (i0 | (i1 << 2)) << bit_pos

        # Atomic OR into meta (since 4 groups contribute to each meta element)
        meta_offset = row * meta_ncols + meta_col
        tl.atomic_or(META_ptr + meta_offset, bits.to(tl.int32), mask=mask)


# ---------------------------------------------------------------------------
# Meta reordering for CUTLASS layout
# ---------------------------------------------------------------------------

def _compute_meta_reorder_offsets(M, meta_ncols, device):
    """Pre-compute scatter offsets for CUTLASS metadata reordering.

    Returns a (M * meta_ncols,) int64 tensor. Only depends on shape,
    so compute once and cache.
    """
    dst_rows = torch.arange(0, M, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(M, 1)

    # Reorder rows, then swizzle 2x2 blocks (int16 meta → group=32, interweave=4)
    group, interweave = 32, 4
    dst_rows = (
        dst_rows // group * group
        + (dst_rows % 8) * interweave
        + (dst_rows % group) // 8
    )

    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * M * interleave + dst_rows * interleave + cols_min).view(-1)


# Global cache for reorder offsets (keyed by (M, meta_ncols, device))
_reorder_cache = {}


def _get_reorder_offsets(M, meta_ncols, device):
    key = (M, meta_ncols, device)
    if key not in _reorder_cache:
        _reorder_cache[key] = _compute_meta_reorder_offsets(M, meta_ncols, device)
    return _reorder_cache[key]


# ---------------------------------------------------------------------------
# Public API: fast_sparse24_compress
# ---------------------------------------------------------------------------

def fast_sparse24_compress(x):
    """Fused 2:4 mask + CUTLASS compression via Triton.

    Args:
        x: (M, K) tensor, bf16 or fp16. K must be divisible by 16.

    Returns:
        (packed, meta_reordered): ready for SparseSemiStructuredTensorCUTLASS
    """
    assert x.ndim == 2 and x.is_contiguous()
    M, K = x.shape
    assert K % 16 == 0, f"K={K} must be divisible by 16"

    meta_ncols = K // 16
    packed = torch.empty((M, K // 2), dtype=x.dtype, device=x.device)
    # Use int32 for atomic OR accumulation, convert to int16 after
    meta_i32 = torch.zeros((M, meta_ncols), dtype=torch.int32, device=x.device)

    total_groups = M * K // 4
    BLOCK = 1024
    grid = ((total_groups + BLOCK - 1) // BLOCK,)

    _fused_mask_compress_kernel[grid](
        x.view(-1), packed.view(-1), meta_i32.view(-1),
        total_groups,
        K=K,
        BLOCK=BLOCK,
    )

    meta = meta_i32.to(torch.int16)

    # Reorder metadata for CUTLASS layout
    offsets = _get_reorder_offsets(M, meta_ncols, x.device)
    meta_reordered = torch.empty_like(meta).view(-1)
    meta_reordered.scatter_(0, offsets, meta.view(-1))
    meta_reordered = meta_reordered.view(M, meta_ncols)

    return packed, meta_reordered


def apply_24_sparsity(x):
    """Enforce 2:4 sparsity: keep top-2 magnitude values per group of 4 along last dim."""
    shape = x.shape
    x = x.reshape(-1, 4)
    _, idx = torch.topk(x.abs(), k=2, dim=-1)
    mask = torch.zeros_like(x)
    mask.scatter_(-1, idx, 1.0)
    return (x * mask).reshape(shape)


# ---------------------------------------------------------------------------
# Sparse24MLP: drop-in MLP replacement
# ---------------------------------------------------------------------------

# Register as a custom op so torch.compile can include it in the graph
# without graph breaks. Dynamo treats it as an opaque leaf op.
@torch.library.custom_op("sparse24::compress_and_matmul", mutates_args=())
def _sparse24_compress_and_matmul(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    packed, meta = fast_sparse24_compress(x)
    return torch._sparse_semi_structured_mm(packed, meta, weight.t())

@_sparse24_compress_and_matmul.register_fake
def _sparse24_compress_and_matmul_fake(x, weight):
    M = x.shape[0]
    N = weight.shape[0]
    return torch.empty(M, N, dtype=x.dtype, device=x.device)

def _sparse24_setup_backward(ctx, inputs, output):
    x, weight = inputs
    ctx.save_for_backward(x, weight)

def _sparse24_backward(ctx, grad):
    x, weight = ctx.saved_tensors
    # STE: backward through sparsification as if it weren't there
    grad_x = grad @ weight
    grad_w = grad.t() @ x
    return grad_x, grad_w

_sparse24_compress_and_matmul.register_autograd(
    _sparse24_backward, setup_context=_sparse24_setup_backward
)


def _sparse24_forward(x_2d, weight):
    return _sparse24_compress_and_matmul(x_2d, weight)


class Sparse24MLP(nn.Module):
    """Drop-in MLP replacement with 2:4 activation sparsity on c_proj.

    Uses a fused Triton kernel for compression and CUTLASS sparse tensor
    cores for ~2x matmul speedup.

    Shares weight parameters with the original MLP — no copies.
    """

    def __init__(self, c_fc, c_proj, emulate=False):
        super().__init__()
        self.c_fc = c_fc
        self.c_proj = c_proj
        self.emulate = emulate

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()

        # Reuse MLP's class-level sparsity tracking so diagnostics work
        # even after MLP modules are replaced with Sparse24MLP
        from nanochat.gpt import MLP
        if MLP.track_sparsity:
            with torch.no_grad():
                n_total = x.numel()
                n_nz = (x != 0).sum().item()
                sparsity = 1.0 - n_nz / n_total
                flat = x.view(-1, 4)
                _, idx = torch.topk(flat.abs(), k=2, dim=-1)
                mask = torch.zeros_like(flat)
                mask.scatter_(-1, idx, 1.0)
                n_nz_after = ((flat * mask) != 0).sum().item()
                drop = (n_nz - n_nz_after) / max(n_nz, 1)
                MLP.sparsity_log.append((sparsity, drop))

        leading = x.shape[:-1]
        K = x.shape[-1]
        x_2d = x.reshape(-1, K)
        M = x_2d.shape[0]

        # During eval, skip hardware sparse path — the Triton kernel + CUTLASS
        # overhead isn't amortized for the many small forward passes in eval.
        # Use dense matmul with the 2:4 mask applied (mathematically equivalent).
        if not self.training:
            x_masked = apply_24_sparsity(x_2d)
            y_2d = x_masked @ self.c_proj.weight.t()
        else:
            use_hw = (
                not self.emulate
                and _TRITON_AVAILABLE
                and _CUTLASS_SPARSE_AVAILABLE
                and x_2d.dtype in (torch.float16, torch.bfloat16)
                and K % 16 == 0
                and M % 32 == 0  # CUTLASS minimum
                and K >= 64      # CUTLASS minimum
            )

            if use_hw:
                y_2d = _sparse24_forward(x_2d, self.c_proj.weight)
            elif not self.emulate and _CUSPARSELT_AVAILABLE and x_2d.dtype in (torch.float16, torch.bfloat16):
                x_masked = apply_24_sparsity(x_2d)
                from torch.sparse import to_sparse_semi_structured
                x_sparse = to_sparse_semi_structured(x_masked)
                y_2d = torch.mm(x_sparse, self.c_proj.weight.t())
            else:
                x_masked = apply_24_sparsity(x_2d)
                y_2d = x_masked @ self.c_proj.weight.t()

        return y_2d.reshape(*leading, -1)


def convert_model_to_sparse24(model, emulate=False):
    """Replace MLP modules with Sparse24MLP. Shares weights, no copies.

    Returns the number of layers converted.
    """
    from nanochat.gpt import MLP
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, MLP):
            sparse_mlp = Sparse24MLP(module.c_fc, module.c_proj, emulate=emulate)
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], sparse_mlp)
            count += 1
    return count
