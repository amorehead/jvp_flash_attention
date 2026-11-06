"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team
Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

Plus modifications to support Jacobian-vector products (JVPs) and Hessian-vector products (HVPs):
- Formulation of flash JVP, by Cheng Lu and Yang Song in https://arxiv.org/abs/2410.11081.
- Reference Triton implementation, by Sofian Mejjoute.
- Reimplementing reference implementation as an autograd function with latest Triton tutorial optimizations, by Alex Birch.
- Support for forward to receive tangents, so as to compute fwd and jvp together; autograd workaround, by Emily (nshepperd).
- Support for function transforms (e.g., torch.func.jvp) via the use of setup_context, by Shih-Ying Yeh.
- Support for sequence lengths 32 & 64; float32 & bfloat16 precision; comprehensive, length and dtype-stratified unit tests;
    working backward hook w.r.t. tensor contiguity; HVP stress testing; standardized docstrings/packaging; and masking/dropout, by Alex Morehead.
"""

from __future__ import annotations

import os
from typing import Any, Literal, NamedTuple

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

from .naive_pytorch import (
    MASK_CONST,
    exists,
    prepare_attn_mask,
    tiled_attention,
    tiled_attention_backward,
)

# NOTE: The Triton kernels require the sequence length to be a multiple of this
# value (and at least this large); short or mismatched sequences are padded up.
MIN_SEQUENCE_LENGTH = 32

# NOTE: Uncomment to turn warnings into errors for debugging
# import warnings
# warnings.filterwarnings("error", category=UserWarning)
# warnings.filterwarnings("error", category=RuntimeWarning)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

    class _TritonConfig(NamedTuple):
        """Minimal stand-in for `triton.Config`."""

        kwargs: dict[str, Any]
        num_stages: int = 3
        num_warps: int = 4
        pre_hook: Any = None

    class _TritonStub:
        """Minimal stand-in for the `triton` module, allowing import without Triton."""

        Config = _TritonConfig

        @staticmethod
        def jit(fn=None, **kwargs):
            """Return the decorated function unchanged."""
            if fn is None:
                return lambda f: f
            return fn

        @staticmethod
        def cdiv(x: int, y: int) -> int:
            """Ceiling division."""
            return -(-x // y)

        @staticmethod
        def set_allocator(*args, **kwargs) -> None:
            """No-op stand-in for `triton.set_allocator`."""
            return None

    triton = _TritonStub()
    tl = _TritonStub()

try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    HAS_TENSOR_DESC = True
except ModuleNotFoundError:
    HAS_TENSOR_DESC = False


def use_naive_attention(
    device: torch.device | str | Tensor, USE_NAIVE: bool | None = None
) -> bool:
    """Whether to use the pure-PyTorch tiled attention instead of the Triton kernels.

    Args:
        device: The device (or a tensor on the device) the attention will run on.
        USE_NAIVE: Whether to force the pure-PyTorch implementation (True), force the
            Triton kernels (False), or choose automatically (None).

    Returns:
        True if the pure-PyTorch implementation should be used, otherwise False.
    """
    assert not exists(USE_NAIVE) or isinstance(
        USE_NAIVE, bool
    ), f"USE_NAIVE must be None, True, or False, but got {USE_NAIVE!r}"
    if isinstance(device, Tensor):
        device = device.device
    device = torch.device(device)

    triton_available = HAS_TRITON and device.type == "cuda" and torch.cuda.is_available()
    if exists(USE_NAIVE):
        assert USE_NAIVE or triton_available, (
            "Cannot force the Triton backend (USE_NAIVE=False) because "
            f"Triton is {'installed' if HAS_TRITON else 'not installed'}, "
            f"CUDA is {'available' if torch.cuda.is_available() else 'unavailable'}, and the "
            f"device is {device!r}; pass USE_NAIVE=True or USE_NAIVE=None instead."
        )
        return USE_NAIVE

    return not triton_available


def is_hip():
    """Check if the current device is HIP."""
    try:
        return torch.cuda.is_available() and torch.version.hip is not None
    except Exception:
        return False


def is_cuda():
    """Check if the current device is CUDA."""
    try:
        return torch.cuda.is_available() and torch.version.cuda is not None
    except Exception:
        return False


def supports_host_descriptor():
    """Check if the current device supports host tensor descriptors."""
    try:
        return is_cuda() and torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


def supports_tma():
    """Check if the current device supports Tensor Memory Access (TMA)."""
    try:
        return HAS_TENSOR_DESC and is_cuda() and torch.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


def is_blackwell():
    """Check if the current device is Blackwell architecture."""
    try:
        return is_cuda() and torch.cuda.get_device_capability()[0] == 10
    except Exception:
        return False


def _is_compiling() -> bool:
    """Check whether code is currently being traced by torch.compile."""
    try:
        return torch.compiler.is_compiling()
    except Exception:
        try:
            return torch._dynamo.is_compiling()
        except Exception:
            return False


# padding function
def pad_to_length(t: Tensor, length: int, left: bool = False) -> Tensor:
    """-> (..., length, head_dim), zero-padding the sequence dimension of `t`."""
    pad = length - t.shape[-2]
    if pad <= 0:
        return t
    return F.pad(t, (0, 0, pad, 0) if left else (0, 0, 0, pad))


def min_padded_length(seq_len: int, min_sequence_length: int = MIN_SEQUENCE_LENGTH) -> int:
    """-> the smallest multiple of min_sequence_length that is >= seq_len."""
    return (
        (seq_len + min_sequence_length - 1) // min_sequence_length
    ) * min_sequence_length


def pad_mode_for(
    n_q: int, n_kv: int, has_mask: bool, min_sequence_length: int = MIN_SEQUENCE_LENGTH
) -> str | None:
    """-> "left", "right", or None, depending on whether/where padding is needed."""
    needs_pad = (
        n_q != n_kv or n_q < min_sequence_length or n_q % min_sequence_length != 0
    )
    if not needs_pad:
        return None
    return "right" if has_mask else "left"


def build_internal_mask(
    n_q: int, n_kv: int, length: int, causal: bool, device: torch.device
) -> Tensor:
    """-> (length, length) boolean mask for left-padded, query-as-tail attention."""
    row = torch.arange(length, device=device)
    col = torch.arange(length, device=device)
    keep = (row >= length - n_q)[:, None] & (col >= length - n_kv)[None, :]
    if causal:
        keep = keep & (row[:, None] >= col[None, :])
    return keep


def pad_attn_mask(mask_tensor: Tensor, mask_type: int, length: int) -> Tensor:
    """-> (..., length, length) mask, masking out padded keys/values."""
    pad = length - mask_tensor.shape[-1]
    if pad <= 0:
        return mask_tensor
    if mask_type == 1:
        return F.pad(mask_tensor, (0, pad, 0, pad), value=False)
    # NOTE: Additive masks need -inf in padded columns so they contribute exactly
    # nothing to the softmax; padded query rows use 0 to remain numerically safe.
    mask_tensor = F.pad(mask_tensor, (0, pad, 0, 0), value=float("-inf"))
    return F.pad(mask_tensor, (0, 0, 0, pad), value=0.0)


def excise_padding(t: Tensor, n: int, pad_mode: str | None) -> Tensor:
    """-> (..., n, head_dim), dropping the padded rows from `t`."""
    if pad_mode is None:
        return t
    return t[..., -n:, :] if pad_mode == "left" else t[..., :n, :]


@triton.jit
def create_dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    """Generate dropout mask using Philox RNG.

    Args:
        philox_seed: Seed for Philox RNG.
        philox_offset: Offset for Philox RNG.
        dropout_p: Dropout probability.
        m: Number of rows.
        n: Number of columns.
        stride: Stride for the output mask.

    Returns:
        dropout_mask: A boolean mask indicating which elements to keep (1.0) or drop (0.0).
        dropout_scale: Scale factor to apply after dropout.
    """
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    offs = ms[:, None] * stride + ns[None, :]
    rng_offs = philox_offset + offs

    # Generate random values using Philox
    rand_vals = tl.rand(philox_seed, rng_offs)

    # Create dropout mask (1.0 = keep, 0.0 = drop)
    dropout_mask = rand_vals > dropout_p
    dropout_scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 0.0

    return dropout_mask, dropout_scale


@triton.jit
def _attn_fwd_inner(
    acc,
    g_acc,  #
    l_i,
    m_i,  #
    mu_i,
    p_tv_acc,  #
    q,
    t_q,  #
    K_block_ptr,
    V_block_ptr,  #
    T_K_block_ptr,
    T_V_block_ptr,  #
    # Mask and dropout parameters
    mask_block_ptr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    # Other parameters
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    sm_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,  #
    ENABLE_JVP: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_TYPE: tl.constexpr,  # 0: no mask, 1: boolean, 2: additive
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    """Inner forward pass for attention mechanism.

    Args:
        acc: Accumulator tensor.
        g_acc: Gradient accumulator tensor.
        l_i: Tensor for storing intermediate results.
        m_i: Tensor for storing intermediate results.
        mu_i: Tensor for storing intermediate results.
        p_tv_acc: Tensor for storing intermediate results.
        q: Query tensor.
        t_q: Tangent of the query tensor.
        K_block_ptr: Pointer to the key block.
        V_block_ptr: Pointer to the value block.
        T_K_block_ptr: Pointer to the tangent key block.
        T_V_block_ptr: Pointer to the tangent value block.
        mask_block_ptr: Pointer to the attention mask block.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        dtype: Data type of the tensors.
        start_m: Starting index for the current block.
        qk_scale: Scale factor for the query-key dot product.
        sm_scale: Scale factor for the softmax.
        BLOCK_M: Block size for the M dimension.
        HEAD_DIM: Dimension of the attention heads.
        BLOCK_N: Block size for the N dimension.
        STAGE: Current stage of the computation.
        offs_m: Offsets for the M dimension.
        offs_n: Offsets for the N dimension.
        N_CTX: Number of context tokens.
        warp_specialize: Whether to apply warp specialization.
        ENABLE_JVP: Whether to enable JVP (Jacobian-vector product).
        ENABLE_DROPOUT: Whether to enable dropout.
        MASK_TYPE: Type of attention mask (0: no mask, 1: boolean, 2: additive).
        MASK_CONST: Constant value used for masking.

    Returns:
        The output tensors as a tuple.
    """
    # Range of values handled by this stage
    if STAGE == 1:
        # NOTE: From 0 to the left of the diagonal
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # NOTE: Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # NOTE: Only used for non-causal attention
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # NOTE: In fp8 mode, we may want to advance the V_block_ptr differently.
    # I did try advancing by (0, lo) instead for fp8, but I got an illegal memory access.
    # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    if ENABLE_JVP:
        T_K_block_ptr = tl.advance(T_K_block_ptr, (0, lo))
        T_V_block_ptr = tl.advance(T_V_block_ptr, (lo, 0))

    if MASK_TYPE > 0:
        mask_block_ptr = tl.advance(mask_block_ptr, (0, lo))

    # Loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        # Let the compiler know that start_n is a multiple
        # of BLOCK_N, so the compiler can do optimizations
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- Compute qk --
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if ENABLE_JVP:
            t_k = tl.load(T_K_block_ptr)
            t_qk = tl.dot(t_q, k) + tl.dot(q, t_k)

        # Load and apply attention mask if provided (before scaling for STAGE != 2)
        if MASK_TYPE > 0:
            mask = tl.load(mask_block_ptr)
            if MASK_TYPE == 1:  # Boolean mask
                # Convert boolean to additive mask: True (attend) -> 0, False (ignore) -> -inf
                qk = qk + tl.where(mask == 1, 0.0, MASK_CONST)
                if ENABLE_JVP:
                    t_qk = tl.where(mask == 1, t_qk, 0.0)

            elif MASK_TYPE == 2:  # Additive mask
                qk = qk + mask

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # For causal attention (STAGE == 2)
        elif STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, MASK_CONST)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        # No masking case (MASK_TYPE == 0 and STAGE != 2)
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)

        if MASK_TYPE == 1 or STAGE == 2:
            # Account for fully masked sequence blocks
            p = tl.where(mask == 1, p, 0.0)

        # Apply dropout if enabled
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX
            )
            p = p * dropout_mask.to(dtype) * dropout_scale

        l_ij = tl.sum(p, 1)

        # -- Update m_i and l_i --
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # -- Update output accumulator --
        if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        # NOTE: We may need to transpose v if dtype == tl.float8e5
        # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
        p = p.to(dtype)

        if ENABLE_JVP:
            p_tqk = p * (t_qk * sm_scale)

            if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
                BM: tl.constexpr = g_acc.shape[0]
                BN: tl.constexpr = g_acc.shape[1]
                g_acc0, g_acc1 = g_acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                g_acc0 = g_acc0 * alpha[:, None]
                g_acc1 = g_acc1 * alpha[:, None]
                g_acc = tl.join(g_acc0, g_acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                g_acc = g_acc * alpha[:, None]

            g_acc = tl.dot(p_tqk.to(v.dtype), v, g_acc)
            mu_ij = tl.sum(p_tqk, 1)
            mu_i = mu_i * alpha + mu_ij
            t_v = tl.load(T_V_block_ptr)
            p_tv_acc = p_tv_acc * alpha[:, None] + tl.dot(p.to(t_v.dtype), t_v).to(t_v.dtype)
            T_V_block_ptr = tl.advance(T_V_block_ptr, (BLOCK_N, 0))
            T_K_block_ptr = tl.advance(T_K_block_ptr, (0, BLOCK_N))

        acc = tl.dot(p.to(v.dtype), v, acc).to(acc.dtype)

        # -- Update m_i --
        m_i = m_ij

        # -- Move to the next block of K, V, and maybe the mask --
        # NOTE: The fp8 PR made a change to how K and V are advanced here but I believe we already have that.
        # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

        if MASK_TYPE > 0:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))

    return acc, g_acc, l_i, m_i, mu_i, p_tv_acc


@triton.jit
def _attn_fwd_inner_tma(
    acc,
    g_acc,  #
    l_i,
    m_i,  #
    mu_i,
    p_tv_acc,  #
    q,
    t_q,  #
    desc_k,
    desc_v,  #
    desc_k_t,
    desc_v_t,  #
    offset_y,
    # Mask and dropout parameters
    mask_block_ptr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    # Other parameters
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    sm_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    ENABLE_JVP: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_TYPE: tl.constexpr,  # 0: no mask, 1: boolean, 2: additive
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    """Inner forward pass for attention mechanism with TMA (Tensor Memory Access) support.

    Args:
        acc: Accumulator tensor.
        g_acc: Gradient accumulator tensor.
        l_i: Tensor for layer normalization.
        m_i: Tensor for masking.
        mu_i: Tensor for mean.
        p_tv_acc: Tensor for TV attention.
        q: Query tensor.
        t_q: Transposed query tensor.
        desc_k: Descriptor for key tensor.
        desc_v: Descriptor for value tensor.
        desc_k_t: Descriptor for transposed key tensor.
        desc_v_t: Descriptor for transposed value tensor.
        offset_y: Offset for y dimension.
        mask_block_ptr: Pointer to the attention mask block.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        dtype: Data type.
        start_m: Start index for m dimension.
        qk_scale: Scale factor for qk.
        sm_scale: Scale factor for sm.
        BLOCK_M: Block size for m dimension.
        HEAD_DIM: Head dimension size.
        BLOCK_N: Block size for n dimension.
        STAGE: Stage of computation.
        offs_m: Offset for m dimension.
        offs_n: Offset for n dimension.
        N_CTX: Context size.
        warp_specialize: Flag for warp specialization.
        ENABLE_JVP: Flag for enabling JVP.
        ENABLE_DROPOUT: Flag for enabling dropout.
        MASK_TYPE: Type of attention mask (0: no mask, 1: boolean, 2: additive).
        MASK_CONST: Constant value used for masking.

    Returns:
        The output tensors as a tuple.
    """
    # Range of values handled by this stage
    if STAGE == 1:
        # NOTE: From 0 to the left of the diagonal
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        # NOTE: Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        # NOTE: Only used for non-causal attention
        lo, hi = 0, N_CTX

    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    if MASK_TYPE > 0:
        mask_block_ptr = tl.advance(mask_block_ptr, (0, lo))

    # Loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        # Let the compiler know that start_n is a multiple
        # of BLOCK_N, so the compiler can do optimizations
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # -- Compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if ENABLE_JVP:
            t_k = desc_k_t.load([offsetk_y, 0]).T
            t_qk = tl.dot(t_q, k) + tl.dot(q, t_k)

        # Load and apply attention mask if provided (before scaling for STAGE != 2)
        if MASK_TYPE > 0:
            mask = tl.load(mask_block_ptr)
            if MASK_TYPE == 1:  # Boolean mask
                # Convert boolean to additive mask: True (attend) -> 0, False (ignore) -> -inf
                qk = qk + tl.where(mask == 1, 0.0, MASK_CONST)
                if ENABLE_JVP:
                    t_qk = tl.where(mask == 1, t_qk, 0.0)

            elif MASK_TYPE == 2:  # Additive mask
                qk = qk + mask

            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        # For causal attention (STAGE == 2)
        elif STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, MASK_CONST)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]

        # No masking case (MASK_TYPE == 0 and STAGE != 2)
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)

        if MASK_TYPE == 1 or STAGE == 2:
            # Account for fully masked sequence blocks
            p = tl.where(mask == 1, p, 0.0)

        # Apply dropout if enabled
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX
            )
            p = p * dropout_mask.to(dtype) * dropout_scale

        # -- Compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # -- Update output accumulator --
        if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]

        # Prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])

        p = p.to(dtype)

        if ENABLE_JVP:
            p_tqk = p * (t_qk * sm_scale)

            # NOT: This non-transposed v for FP8 is presumably only supported on Blackwell
            if warp_specialize and (BLOCK_M == 128 and HEAD_DIM == 128):
                BM: tl.constexpr = g_acc.shape[0]
                BN: tl.constexpr = g_acc.shape[1]
                g_acc0, g_acc1 = g_acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
                g_acc0 = g_acc0 * alpha[:, None]
                g_acc1 = g_acc1 * alpha[:, None]
                g_acc = tl.join(g_acc0, g_acc1).permute(0, 2, 1).reshape([BM, BN])
            else:
                g_acc = g_acc * alpha[:, None]

            g_acc = tl.dot(p_tqk.to(v.dtype), v, g_acc)
            mu_ij = tl.sum(p_tqk, 1)
            mu_i = mu_i * alpha + mu_ij
            t_v = desc_v_t.load([offsetv_y, 0])
            p_tv_acc = p_tv_acc * alpha[:, None] + tl.dot(p.to(t_v.dtype), t_v).to(t_v.dtype)

        # NOTE: This non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p.to(v.dtype), v, acc).to(acc.dtype)

        # Update m_i and l_i
        # Place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N

        if MASK_TYPE > 0:
            mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_N))

    return acc, g_acc, l_i, m_i, mu_i, p_tv_acc


def _host_descriptor_pre_hook(nargs):
    """Pre-hook to set up tensor descriptors for the attention kernel.

    Args:
        nargs: A dictionary of kernel arguments.
    """
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if not supports_tma() or not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s,
        num_warps=w,
        pre_hook=_host_descriptor_pre_hook,
    )
    for BM in [MIN_SEQUENCE_LENGTH, 64, 128]
    for BN in [MIN_SEQUENCE_LENGTH, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(
            dict(BLOCK_M=128, BLOCK_N=64),
            num_stages=2,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        ),
    ]


def keep(conf):
    """Keep configurations that meet certain criteria.

    Args:
        conf: A configuration object.
    """
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    """Prune configurations that are invalid based on certain criteria.

    Args:
        configs: A list of configuration objects.
        named_args: A dictionary of named arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        A list of valid configuration objects.
    """
    N_CTX = kwargs["N_CTX"]

    if N_CTX == MIN_SEQUENCE_LENGTH:
        # Filter out configs where BLOCK_M > MIN_SEQUENCE_LENGTH
        return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= MIN_SEQUENCE_LENGTH]

    # Filter out configs where BLOCK_M > N_CTX or BLOCK_M <= MIN_SEQUENCE_LENGTH, as
    # BLOCK_M = MIN_SEQUENCE_LENGTH often leads to reduced numerical accuracy for longer sequences
    # TODO: Find out why this occurs
    return [
        conf for conf in configs if MIN_SEQUENCE_LENGTH < conf.kwargs.get("BLOCK_M", 0) <= N_CTX
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    """Maybe make a tensor descriptor from a pointer.

    Args:
        desc_or_ptr: The input tensor or pointer.
        shape: The shape of the tensor.
        strides: The strides of the tensor.
        block_shape: The block shape of the tensor.

    Returns:
        A tensor descriptor.
    """
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# @triton.autotune(
#     configs=list(filter(keep, configs)),
#     key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
#     prune_configs_by={"early_config_prune": prune_invalid_configs},
# )
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    T_Q,
    T_K,
    T_V,  #
    sm_scale,
    M,
    Out,
    T_Out,  #
    Mask,  # Mask tensor
    dropout_p,  # Dropout probability
    philox_seed,  # RNG seed for dropout
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_tqz,
    stride_tqh,
    stride_tqm,
    stride_tqk,  #
    stride_tkz,
    stride_tkh,
    stride_tkn,
    stride_tkk,  #
    stride_tvz,
    stride_tvh,
    stride_tvk,
    stride_tvn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    stride_toz,
    stride_toh,
    stride_tom,
    stride_ton,  #
    stride_mz,  # Mask stride
    stride_mh,  # Mask stride
    stride_mm,  # Mask stride
    stride_mn,  # Mask stride
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    ENABLE_JVP: tl.constexpr,  #
    ENABLE_DROPOUT: tl.constexpr,  #
    MASK_TYPE: tl.constexpr,  #
):
    """Forward attention computation.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        T_Q: Tensor for query.
        T_K: Tensor for key.
        T_V: Tensor for value.
        sm_scale: Scale factor.
        M: Number of rows.
        Out: Output tensor.
        T_Out: Tensor for output.
        Mask: Attention mask tensor.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        stride_qz: Stride for query z dimension.
        stride_qh: Stride for query h dimension.
        stride_qm: Stride for query m dimension.
        stride_qk: Stride for query k dimension.
        stride_kz: Stride for key z dimension.
        stride_kh: Stride for key h dimension.
        stride_kn: Stride for key n dimension.
        stride_kk: Stride for key k dimension.
        stride_vz: Stride for value z dimension.
        stride_vh: Stride for value h dimension.
        stride_vk: Stride for value k dimension.
        stride_vn: Stride for value n dimension.
        stride_tqz: Stride for tensor query z dimension.
        stride_tqh: Stride for tensor query h dimension.
        stride_tqm: Stride for tensor query m dimension.
        stride_tqk: Stride for tensor query k dimension.
        stride_tkz: Stride for tensor key z dimension.
        stride_tkh: Stride for tensor key h dimension.
        stride_tkn: Stride for tensor key n dimension.
        stride_tkk: Stride for tensor key k dimension.
        stride_tvz: Stride for tensor value z dimension.
        stride_tvh: Stride for tensor value h dimension.
        stride_tvk: Stride for tensor value k dimension.
        stride_tvn: Stride for tensor value n dimension.
        stride_oz: Stride for output z dimension.
        stride_oh: Stride for output h dimension.
        stride_om: Stride for output m dimension.
        stride_on: Stride for output n dimension.
        stride_toz: Stride for tensor output z dimension.
        stride_toh: Stride for tensor output h dimension.
        stride_tom: Stride for tensor output m dimension.
        stride_ton: Stride for tensor output n dimension.
        stride_mz: Stride for mask z dimension.
        stride_mh: Stride for mask h dimension.
        stride_mm: Stride for mask m dimension.
        stride_mn: Stride for mask n dimension.
        Z: Number of z dimensions.
        H: Number of h dimensions.
        N_CTX: Number of context dimensions.
        HEAD_DIM: Head dimension.
        BLOCK_M: Block size for the queries.
        BLOCK_N: Block size for the keys/values.
        FP8_OUTPUT: FP8 output flag.
        STAGE: Stage.
        warp_specialize: Warp specialization flag.
        ENABLE_JVP: Enable JVP flag.
        ENABLE_DROPOUT: Enable dropout flag.
        MASK_TYPE: Mask type (0: no mask, 1: boolean, 2: additive).
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)  # N = KV

    # Prepare metadata and indices
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float32  # For dot products
    start_m = tl.program_id(0)  # Which block (in the input query sequence) to process
    off_hz = tl.program_id(
        1
    )  # Which head and batch element to process, with a program being a single head of a single batch element
    off_z = (
        off_hz // H
    )  # Which batch element this program is assigned to (n.b., each batch element has H heads)
    off_h = off_hz % H  # The position of the head to process in the batch

    # NOTE: This allows one to get the (N_CTX, HEAD_DIM) block in Q, K, V by indexing it by batch and head
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Initialize block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),  # M = Q
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX),
        strides=(
            stride_kk,
            stride_kn,
        ),  # NOTE: We invert the strides of K to get its matrix transpose K^T
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # Initialize block pointer for the mask, if provided
    if MASK_TYPE > 0:
        mask_offset = off_z.to(tl.int64) * stride_mz + off_h.to(tl.int64) * stride_mh
        mask_block_ptr = tl.make_block_ptr(
            base=Mask + mask_offset,
            shape=(N_CTX, N_CTX),
            strides=(stride_mm, stride_mn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        mask_block_ptr = None

    # Initialize dropout offset for this block
    philox_offset_base = off_hz * N_CTX * N_CTX

    # Initialize offsets for the query tokens to process
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulator pointers:
    # m, the running maximum (one for each query)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l, the running sum (one for each query as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc, the output accumulator (one vector for each query)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if ENABLE_JVP:
        # NOTE: It's extremely likely we could just reuse qvk_offset, but this seems cheap so whatever
        t_qvk_offset = off_z.to(tl.int64) * stride_tqz + off_h.to(tl.int64) * stride_tqh
        T_Q_block_ptr = tl.make_block_ptr(
            base=T_Q + t_qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_tqm, stride_tqk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # NOTE: Could probably just reuse v_order here
        t_v_order: tl.constexpr = (0, 1) if T_V.dtype.element_ty == tl.float8e5 else (1, 0)
        T_V_block_ptr = tl.make_block_ptr(
            base=T_V + t_qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_tvk, stride_tvn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=t_v_order,
        )
        T_K_block_ptr = tl.make_block_ptr(
            base=T_K + t_qvk_offset,
            shape=(HEAD_DIM, N_CTX),
            strides=(
                stride_tkk,
                stride_tkn,
            ),  # NOTE: We invert the strides of tangent K (k_t) to get its matrix transpose K^T
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        T_O_block_ptr = tl.make_block_ptr(
            base=T_Out + t_qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_tom, stride_ton),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # Load q_t: It will stay in SRAM throughout.
        t_q = tl.load(T_Q_block_ptr)
        g_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        mu_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        p_tv_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    else:
        t_q = None
        T_V_block_ptr = None
        T_K_block_ptr = None
        # Allocate minimal dummy tensors to keep consistent the return signature of _attn_fwd_inner
        g_acc = tl.zeros([1, 1], dtype=tl.float32)
        mu_i = tl.zeros([1], dtype=tl.float32)
        p_tv_acc = tl.zeros([1, 1], dtype=tl.float32)

    # Prepare scales
    qk_scale = sm_scale
    qk_scale = qk_scale * 1.44269504  # 1/log(2)

    # Load q: It will stay in SRAM throughout.
    q = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1
    if STAGE == 1 or STAGE == 3:
        # NOTE: This step runs for non-causal attention or for the
        # blocks to the left of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner(
            acc,
            g_acc,
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            K_block_ptr,
            V_block_ptr,  #
            T_K_block_ptr,
            T_V_block_ptr,  #
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    if STAGE == 3:
        # NOTE: This step runs for the blocks to the
        # right of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner(
            acc,
            g_acc,  #
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            K_block_ptr,
            V_block_ptr,  #
            T_K_block_ptr,
            T_V_block_ptr,  #
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    # Epilogue
    empty_mask = l_i == 0.0
    if empty_mask.sum() > 0:
        l_i = tl.where(
            empty_mask, 1.0, l_i
        )  # NOTE: This happens if the entire block is masked out.

    m_i = m_i + tl.where(
        # NOTE: This is needed to compute the logsumexp for the backward pass.
        empty_mask,
        0.0,
        tl.math.log2(l_i),
    )

    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

    # If JVP is enabled, compute and store the output tangent
    if ENABLE_JVP:
        t_p_v = g_acc / l_i[:, None] - (mu_i / l_i)[:, None] * acc
        t_y_out = t_p_v + p_tv_acc / l_i[:, None]
        tl.store(T_O_block_ptr, t_y_out.to(T_Out.type.element_ty))


def _tma_pre_hook(nargs):
    """Pre-hook for TMA (Tensor Memory Access) optimization.

    Args:
        nargs: A dictionary containing the kernel arguments.
    """
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs_tma = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s,
        num_warps=w,
        pre_hook=_tma_pre_hook,
    )
    for BM in [MIN_SEQUENCE_LENGTH, 64, 128, 256]
    for BN in [MIN_SEQUENCE_LENGTH, 64, 128]
    for s in [3, 4, 5]
    for w in [4, 8]
]


def keep_tma(conf):
    """Check if TMA (Tensor Memory Access) optimization should be kept for the given configuration.

    Args:
        conf: The configuration to check.
    """
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


# @triton.autotune(
#     configs=list(filter(keep_tma, configs_tma)),
#     key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
#     prune_configs_by={"early_config_prune": prune_invalid_configs},
# )
@triton.jit
def _attn_fwd_tma(
    sm_scale,
    M,  #
    Z,
    H,  #
    desc_q,
    desc_k,
    desc_v,  #
    desc_q_t,
    desc_k_t,
    desc_v_t,  #
    desc_o,
    desc_o_t,  #
    Mask,  # Mask tensor
    dropout_p,  # Dropout probability
    philox_seed,  # RNG seed for dropout
    stride_mz,  # Mask stride
    stride_mh,  # Mask stride
    stride_mm,  # Mask stride
    stride_mn,  # Mask stride
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    ENABLE_JVP: tl.constexpr,  #
    ENABLE_DROPOUT: tl.constexpr,  #
    MASK_TYPE: tl.constexpr,  #
):
    """Forward attention computation with TMA (Tensor Memory Access) support.

    Args:
        sm_scale: Scale factor for the softmax.
        M: Number of rows in the input.
        Z: Number of channels in the input.
        H: Number of heads in the multi-head attention.
        desc_q: Descriptor for the query tensor.
        desc_k: Descriptor for the key tensor.
        desc_v: Descriptor for the value tensor.
        desc_q_t: Descriptor for the transposed query tensor.
        desc_k_t: Descriptor for the transposed key tensor.
        desc_v_t: Descriptor for the transposed value tensor.
        desc_o: Descriptor for the output tensor.
        desc_o_t: Descriptor for the transposed output tensor.
        Mask: Attention mask tensor.
        dropout_p: Dropout probability.
        philox_seed: Seed for the Philox random number generator.
        stride_mz: Stride for the mask in the z dimension.
        stride_mh: Stride for the mask in the h dimension.
        stride_mm: Stride for the mask in the m dimension.
        stride_mn: Stride for the mask in the n dimension.
        N_CTX: Context length.
        HEAD_DIM: Dimension of each head.
        BLOCK_M: Block size for the queries.
        BLOCK_N: Block size for the keys/values.
        FP8_OUTPUT: Flag indicating if FP8 output is used.
        STAGE: Stage of the computation.
        warp_specialize: Flag indicating if warp specialization is used.
        ENABLE_JVP: Flag indicating if JVP (Jacobian-vector product) is enabled.
        ENABLE_DROPOUT: Flag indicating if dropout is enabled.
        MASK_TYPE: Type of mask used (0: no mask, 1: boolean, 2: additive).
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)  # N = KV

    # Prepare metadata and indices
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float32  # For dot products
    start_m = tl.program_id(0)  # Which block (in the input query sequence) to process
    off_hz = tl.program_id(
        1
    )  # Which head and batch element to process, with a program being a single head of a single batch element
    off_z = (
        off_hz // H
    )  # Which batch element this program is assigned to (n.b., each batch element has H heads)
    off_h = off_hz % H  # The position of the head to process in the batch

    # Initialize tensor descriptors
    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],  # M = Q
    )
    if FP8_OUTPUT:
        v_shape = [HEAD_DIM, y_dim]
        v_strides = [N_CTX, 1]
        v_block_shape = [HEAD_DIM, BLOCK_N]
    else:
        v_shape = [y_dim, HEAD_DIM]
        v_strides = [HEAD_DIM, 1]
        v_block_shape = [BLOCK_N, HEAD_DIM]
    desc_v = _maybe_make_tensor_desc(
        desc_v, shape=v_shape, strides=v_strides, block_shape=v_block_shape
    )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M

    # Initialize block pointer for the mask, if provided
    if MASK_TYPE > 0:
        mask_offset = off_z * stride_mz + off_h * stride_mh
        mask_block_ptr = tl.make_block_ptr(
            base=Mask + mask_offset,
            shape=(N_CTX, N_CTX),
            strides=(stride_mm, stride_mn),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
    else:
        mask_block_ptr = None

    # Initialize dropout offset for this block
    philox_offset_base = off_hz * N_CTX * N_CTX

    # Initialize offsets for the query tokens to process
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize accumulator pointers:
    # m, the running maximum (one for each query)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    # l, the running sum (one for each query as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    # acc, the output accumulator (one vector for each query)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if ENABLE_JVP:
        desc_q_t = _maybe_make_tensor_desc(
            desc_q_t,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        if FP8_OUTPUT:
            t_v_shape = [HEAD_DIM, y_dim]
            t_v_strides = [N_CTX, 1]
            t_v_block_shape = [HEAD_DIM, BLOCK_N]
        else:
            t_v_shape = [y_dim, HEAD_DIM]
            t_v_strides = [HEAD_DIM, 1]
            t_v_block_shape = [BLOCK_N, HEAD_DIM]
        desc_v_t = _maybe_make_tensor_desc(
            desc_v_t, shape=t_v_shape, strides=t_v_strides, block_shape=t_v_block_shape
        )
        desc_k_t = _maybe_make_tensor_desc(
            desc_k_t,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
        desc_o_t = _maybe_make_tensor_desc(
            desc_o_t,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_M, HEAD_DIM],
        )
        # Load t_q: It will stay in SRAM throughout.
        t_q = desc_q_t.load([qo_offset_y, 0])
        g_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        mu_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        p_tv_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    else:
        t_q = None
        desc_k_t = None
        desc_v_t = None
        # Allocate minimal dummy tensors to keep consistent the return signature of _attn_fwd_inner_tma
        g_acc = tl.zeros([1, 1], dtype=tl.float32)
        mu_i = tl.zeros([1], dtype=tl.float32)
        p_tv_acc = tl.zeros([1, 1], dtype=tl.float32)

    # Prepare scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    # Load q: It will stay in SRAM throughout.
    q = desc_q.load([qo_offset_y, 0])

    # Stage: 3 if causal, else 1
    if STAGE == 1 or STAGE == 3:
        # NOTE: This step runs for non-causal attention or for the
        # blocks to the left of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner_tma(
            acc,
            g_acc,  #
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            desc_k,
            desc_v,  #
            desc_k_t,
            desc_v_t,  #
            offset_y,
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    if STAGE == 3:
        # NOTE: This step runs for the blocks to the
        # right of the diagonal for causal attention
        acc, g_acc, l_i, m_i, mu_i, p_tv_acc = _attn_fwd_inner_tma(
            acc,
            g_acc,  #
            l_i,
            m_i,  #
            mu_i,
            p_tv_acc,  #
            q,
            t_q,  #
            desc_k,
            desc_v,  #
            desc_k_t,
            desc_v_t,  #
            offset_y,
            mask_block_ptr,
            dropout_p,
            philox_seed,
            philox_offset_base,
            dtype,
            start_m,
            qk_scale,
            sm_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            N_CTX,  #
            warp_specialize,
            ENABLE_JVP,
            ENABLE_DROPOUT,
            MASK_TYPE,
        )

    # Epilogue
    empty_mask = l_i == 0.0
    if empty_mask.sum() > 0:
        l_i = tl.where(
            empty_mask, 1.0, l_i
        )  # NOTE: This happens if the entire block is masked out.

    m_i = m_i + tl.where(
        # NOTE: This is needed to compute the logsumexp for the backward pass.
        empty_mask,
        0.0,
        tl.math.log2(l_i),
    )

    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(desc_o.dtype))

    if ENABLE_JVP:
        t_p_v = g_acc / l_i[:, None] - (mu_i / l_i)[:, None] * acc
        t_y_out = t_p_v + p_tv_acc / l_i[:, None]
        desc_o_t.store([qo_offset_y, 0], t_y_out.to(desc_o_t.dtype))


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  # noqa: E741
):
    """Preprocess output deltas for the backward attention pass.

    Args:
        O: Output tensor.
        DO: Gradient of the output tensor.
        Delta: Accumulated gradients.
        N_CTX: Context length.
        BLOCK_M: Block size for M dimension.
        HEAD_DIM: Head dimension size.
    """
    # Collect sequence, batch, and head indices
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)

    # Load outputs and gradients
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(
        tl.float32
    )
    delta = tl.sum(o * do, axis=1)

    # Write-back the intermediate delta results
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    CAUSAL_MASKING: tl.constexpr,
    # Args for masking/dropout
    mask_ptr,
    mask_stride_tok1,
    mask_stride_tok2,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    """The main inner-loop logic for computing dK and dV.

    Args:
        dk: Gradient of the key tensor.
        dv: Gradient of the value tensor.
        Q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        DO: Gradient of the output tensor.
        M: Memory tensor.
        D: Delta tensor.
        stride_tok: Stride for the token dimension.
        stride_d: Stride for the head dimension.
        N_CTX: Context length.
        BLOCK_M1: Block size for M dimension.
        BLOCK_N1: Block size for N dimension.
        HEAD_DIM: Head dimension size.
        start_n: Starting index for N dimension.
        start_m: Starting index for M dimension.
        num_steps: Number of steps to unroll.
        CAUSAL_MASKING: Flag for causal masking.
        mask_ptr: Pointer to the mask tensor.
        mask_stride_tok1: Stride for the third (row) dimension of the mask tensor.
        mask_stride_tok2: Stride for the fourth (column) dimension of the mask tensor.
        MASK_TYPE: Type of masking (0: no mask, 1: boolean mask,
                     2: additive mask).
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        ENABLE_DROPOUT: Flag to enable dropout.
        MASK_CONST: Constant used for masking.

    Returns:
        dk: Gradient of the key tensor.
        dv: Gradient of the value tensor.
    """
    # Initialize pointers for Q and DO
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_h = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_h[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d

    if MASK_TYPE > 0:
        mask_ptr = (
            mask_ptr + offs_m[None, :] * mask_stride_tok1 + offs_n[:, None] * mask_stride_tok2
        )

    # NOTE: BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    # Compute fp8 gradients in fp32; other dtypes keep their tensor cores
    dot_dtype: tl.constexpr = (
        tl.float32 if do_ptrs.dtype.element_ty == tl.float8e5 else do_ptrs.dtype.element_ty
    )

    # Iteratively compute dK and dV over the M dimension
    for _ in range(num_steps):
        qT = tl.load(qT_ptrs)

        # Load m before computing qk to reduce pipeline stall
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)

        # Exponentiation
        pT = tl.math.exp2(qkT - m[None, :])

        # External masking after exponentiation
        if MASK_TYPE > 0:
            mask = tl.load(mask_ptr)
            if MASK_TYPE == 1:  # Boolean mask
                pT = tl.where(mask == 1, pT, 0.0)
            elif MASK_TYPE == 2:  # Additive mask
                # NOTE: Padded columns use -inf, so they are excluded exactly.
                attend = (mask != MASK_CONST) & (mask > -float("inf"))
                pT = tl.where(attend, pT, 0.0)

        # (or) Causal masking after exponentiation
        elif CAUSAL_MASKING:
            causal_mask = offs_m[None, :] >= offs_n[:, None]
            pT = tl.where(causal_mask, pT, 0.0)

        # Dropout after exponentiation
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + curr_m * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M1, BLOCK_N1, N_CTX
            )
            pT = pT * dropout_mask.to(pT.dtype) * dropout_scale

        # Compute dV
        ppT = pT
        do = tl.load(do_ptrs)
        dv += tl.dot(ppT.to(dot_dtype), do.to(dot_dtype)).to(do.dtype)
        # NOTE: D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)

        # Compute dP and dS to derive dK
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)

        if ENABLE_DROPOUT:  # This derivative should be masked with the same dropout mask
            dpT = dpT * dropout_mask.to(dpT.dtype) * dropout_scale

        dsT = pT * (dpT - Di[None, :])
        dk += tl.dot(dsT.to(dot_dtype), tl.trans(qT).to(dot_dtype)).to(qT.dtype)

        # Increment pointers
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok

        if MASK_TYPE > 0:
            mask_ptr += step_m * mask_stride_tok1

    return dk, dv


# The main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    K,
    V,  #
    do,
    m,
    D,
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    CAUSAL_MASKING: tl.constexpr,
    # Args for masking/dropout
    mask_ptr,
    mask_stride_tok1,
    mask_stride_tok2,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    ENABLE_DROPOUT: tl.constexpr,
    MASK_CONST: tl.constexpr = MASK_CONST,
):
    """The main inner-loop logic for computing dQ.

    Args:
        dq: Gradient of the query tensor.
        q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        do: Gradient of the output tensor.
        m: Memory tensor.
        D: Delta tensor.
        stride_tok: Stride for the token dimension.
        stride_d: Stride for the head dimension.
        N_CTX: Context length.
        BLOCK_M2: Block size for M dimension.
        BLOCK_N2: Block size for N dimension.
        HEAD_DIM: Head dimension size.
        start_m: Starting index for M dimension.
        start_n: Starting index for N dimension.
        num_steps: Number of steps to unroll.
        CAUSAL_MASKING: Flag for causal masking.
        mask_ptr: Pointer to the mask tensor.
        mask_stride_tok1: Stride for the third (row) dimension of the mask tensor.
        mask_stride_tok2: Stride for the fourth (column) dimension of the mask tensor.
        MASK_TYPE: Type of masking (0: no mask, 1: boolean mask,
                        2: additive mask).
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        ENABLE_DROPOUT: Flag to enable dropout.
        MASK_CONST: Constant used for masking.

    Returns:
        dq: Gradient of the query tensor.
    """
    # Initialize pointers for K, V, and DO
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_h = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_h[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_h[:, None] * stride_d

    if MASK_TYPE > 0:
        mask_ptr = (
            mask_ptr + offs_m[:, None] * mask_stride_tok1 + offs_n[None, :] * mask_stride_tok2
        )

    # NOTE: D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)

    # NOTE: BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    # Compute fp8 gradients in fp32; other dtypes keep their tensor cores
    dot_dtype: tl.constexpr = (
        tl.float32 if kT_ptrs.dtype.element_ty == tl.float8e5 else kT_ptrs.dtype.element_ty
    )

    # Iteratively compute dQ over the N dimension
    for _ in range(num_steps):
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)

        # Exponentiation
        p = tl.math.exp2(qk - m)

        # External masking after exponentiation
        if MASK_TYPE > 0:
            mask = tl.load(mask_ptr)
            if MASK_TYPE == 1:  # Boolean mask
                p = tl.where(mask == 1, p, 0.0)
            elif MASK_TYPE == 2:  # Additive mask
                # NOTE: Padded columns use -inf, so they are excluded exactly.
                attend = (mask != MASK_CONST) & (mask > -float("inf"))
                p = tl.where(attend, p, 0.0)

        # (or) Causal masking after exponentiation
        elif CAUSAL_MASKING:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            p = tl.where(causal_mask, p, 0.0)

        # Dropout after exponentiation
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX + curr_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M2, BLOCK_N2, N_CTX
            )
            p = p * dropout_mask.to(p.dtype) * dropout_scale

        # Compute dP and dS
        dp = tl.dot(do, vT).to(tl.float32)

        if ENABLE_DROPOUT:  # NOTE: This derivative should be masked with the same dropout mask.
            dp = dp * dropout_mask.to(dp.dtype) * dropout_scale

        ds = p * (dp - Di[:, None])

        # Compute dQ
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds.to(dot_dtype), tl.trans(kT).to(dot_dtype)).to(kT.dtype)

        # Increment pointers
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok

        if MASK_TYPE > 0:
            mask_ptr += step_n * mask_stride_tok2

    return dq


@triton.jit
def _attn_bwd_causal(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # Shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    # Used for the mask.
    mask_stride_z,
    mask_stride_h,
    mask_stride_tok1,
    mask_stride_tok2,
    # Dimensions and sizes.
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Args for masking/dropout.
    mask_ptr,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    ENABLE_DROPOUT: tl.constexpr,
):
    """The main backward pass for the (causal) attention mechanism.

    This computes gradients for only ~N²/2 pairwise token interactions,
    since causal attention already masks out half of the interactions.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        sm_scale: Scale factor for the softmax.
        DO: Gradient of the output tensor.
        DQ: Gradient of the query tensor.
        DK: Gradient of the key tensor.
        DV: Gradient of the value tensor.
        M: Memory tensor.
        D: Delta tensor.
        stride_z: Stride for the z dimension.
        stride_h: Stride for the head dimension.
        stride_tok: Stride for the token dimension.
        stride_d: Stride for the head dimension.
        mask_stride_z: Stride for the z dimension in the mask tensor.
        mask_stride_h: Stride for the head dimension in the mask tensor.
        mask_stride_tok1: Stride for the first token (row) dimension in the mask tensor.
        mask_stride_tok2: Stride for the second token (column) dimension in the mask tensor.
        H: Head dimension.
        N_CTX: Context length.
        BLOCK_M1: Block size for M dimension.
        BLOCK_N1: Block size for N dimension.
        BLOCK_M2: Block size for M dimension.
        BLOCK_N2: Block size for N dimension.
        BLK_SLICE_FACTOR: Block slice factor.
        HEAD_DIM: Head dimension size.
        mask_ptr: Pointer to the mask tensor.
        MASK_TYPE: Type of masking (0: no mask, 1: boolean mask,
                        2: additive mask).
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        ENABLE_DROPOUT: Flag to enable dropout.
    """
    # Constants
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # Collect sequence, batch, and head indices
    start_block_id = tl.program_id(0)  # Which block (in the input query sequence) to process
    off_hz = tl.program_id(
        1
    )  # Which head and batch element to process, with a program being a single head of a single batch element
    off_z = (
        off_hz // H
    )  # Which batch element this program is assigned to (n.b., each batch element has H heads)
    off_h = off_hz % H  # The position of the head to process in the batch

    # NOTE: This allows one to get the (N_CTX, HEAD_DIM) block in Q, K, V, etc. by indexing it by batch and head
    delta_shared_offset = (off_hz * N_CTX).to(tl.int64)
    qkv_shared_offset = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h

    # Offset pointers for batch elements and heads
    Q += qkv_shared_offset
    K += qkv_shared_offset
    V += qkv_shared_offset
    DO += qkv_shared_offset
    DQ += qkv_shared_offset
    DK += qkv_shared_offset
    DV += qkv_shared_offset

    M += delta_shared_offset  # NOTE: These tensors have fewer dimensions.
    D += delta_shared_offset

    # Initialize pointer for the mask, if provided
    if MASK_TYPE > 0:
        mask_offset = off_z.to(tl.int64) * mask_stride_z + off_h.to(tl.int64) * mask_stride_h
        mask_ptr += mask_offset

    # Generate philox offset for this block
    philox_offset_base = off_hz * N_CTX * N_CTX

    # ====== COMPUTE dK and dV ======
    # Determine step size for dK and dV computation
    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR

    # Prepare offsets for loading Q/K/V/DO
    start_n = start_block_id * BLOCK_N1
    start_m = start_n

    # Load K and V: They will stay in SRAM throughout.
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_h = tl.arange(0, HEAD_DIM)

    k = tl.load(K + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)

    # Initialize dK and dV accumulators
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # Compute dK and dV for (causally) masked blocks
    num_steps = BLOCK_N1 // MASK_BLOCK_M1
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        CAUSAL_MASKING=True,  #
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for (causally) non-masked blocks
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        CAUSAL_MASKING=False,  #
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    # Write-back dV
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write-back dK (scaled)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # ====== COMPUTE dQ ======
    # Determine step size for dQ computation
    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR

    # Prepare offsets for dQ computation
    start_m = start_block_id * BLOCK_M2
    end_n = start_m + BLOCK_M2

    offs_m = start_m + tl.arange(0, BLOCK_M2)

    # Load Q, DO, and M: They will stay in SRAM throughout.
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

    # Compute dQ for (causally) masked blocks
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
    dq = _attn_bwd_dq(
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important. It's just to reuse the loop structure
        # for dK and dV above as much as possible.
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,  #
        CAUSAL_MASKING=True,  #
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    end_n -= num_steps * MASK_BLOCK_N2

    # Compute dQ for (causally) non-masked blocks
    num_steps = end_n // BLOCK_N2
    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,  #
        CAUSAL_MASKING=False,  #
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    # Write-back dQ (scaled)
    dq *= LN2
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dq_ptrs, dq)


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,  #
    DO,  #
    DQ,
    DK,
    DV,  #
    M,
    D,
    # Shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    # Used for the mask.
    mask_stride_z,
    mask_stride_h,
    mask_stride_tok1,
    mask_stride_tok2,
    # Dimensions and sizes.
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Args for masking/dropout.
    mask_ptr,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    ENABLE_DROPOUT: tl.constexpr,
):
    """The main backward pass for the (non-causal) attention mechanism.

    This computes gradients for all N² pairwise token interactions,
    unlike the causal version which only computes ~N²/2.

    Args:
        Q: Query tensor.
        K: Key tensor.
        V: Value tensor.
        sm_scale: Scale factor for the softmax.
        DO: Gradient of the output tensor.
        DQ: Gradient of the query tensor.
        DK: Gradient of the key tensor.
        DV: Gradient of the value tensor.
        M: Memory tensor.
        D: Delta tensor.
        stride_z: Stride for the z dimension.
        stride_h: Stride for the head dimension.
        stride_tok: Stride for the token dimension.
        stride_d: Stride for the head dimension.
        mask_stride_z: Stride for the z dimension in the mask tensor.
        mask_stride_h: Stride for the head dimension in the mask tensor.
        mask_stride_tok1: Stride for the first token (row) dimension in the mask tensor.
        mask_stride_tok2: Stride for the second token (column) dimension in the mask tensor.
        H: Head dimension.
        N_CTX: Context length.
        BLOCK_M1: Block size for M dimension.
        BLOCK_N1: Block size for N dimension.
        BLOCK_M2: Block size for M dimension.
        BLOCK_N2: Block size for N dimension.
        BLK_SLICE_FACTOR: Block slice factor.
        HEAD_DIM: Head dimension size.
        mask_ptr: Pointer to the mask tensor.
        MASK_TYPE: Type of masking (0: no mask, 1: boolean mask,
                        2: additive mask).
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        ENABLE_DROPOUT: Flag to enable dropout.
    """
    # Constants
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # Collect sequence, batch, and head indices
    start_block_id = tl.program_id(0)  # Which block (in the input query sequence) to process
    off_hz = tl.program_id(
        1
    )  # Which head and batch element to process, with a program being a single head of a single batch element
    off_z = (
        off_hz // H
    )  # Which batch element this program is assigned to (n.b., each batch element has H heads)
    off_h = off_hz % H  # The position of the head to process in the batch

    # NOTE: This allows one to get the (N_CTX, HEAD_DIM) block in Q, K, V, etc. by indexing it by batch and head
    delta_shared_offset = (off_hz * N_CTX).to(tl.int64)
    qkv_shared_offset = off_z.to(tl.int64) * stride_z + off_h.to(tl.int64) * stride_h

    # Offset pointers for batch elements and heads
    Q += qkv_shared_offset
    K += qkv_shared_offset
    V += qkv_shared_offset
    DO += qkv_shared_offset
    DQ += qkv_shared_offset
    DK += qkv_shared_offset
    DV += qkv_shared_offset

    M += delta_shared_offset  # NOTE: These tensors have fewer dimensions.
    D += delta_shared_offset

    # Initialize pointer for the mask, if provided
    if MASK_TYPE > 0:
        mask_offset = off_z.to(tl.int64) * mask_stride_z + off_h.to(tl.int64) * mask_stride_h
        mask_ptr += mask_offset

    # Generate philox offset for this block
    philox_offset_base = off_hz * N_CTX * N_CTX

    # ====== COMPUTE dK and dV ======
    # For non-causal attention, we process ALL query blocks (the entire sequence)
    # This is the key difference from causal: we iterate through all Q positions

    # Prepare offsets for loading Q/K/V/DO
    start_n = start_block_id * BLOCK_N1

    # Load K and V: They will stay in SRAM throughout.
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_h = tl.arange(0, HEAD_DIM)

    k = tl.load(K + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d)

    # Initialize dK and dV accumulators
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    start_m = 0  # Start from the beginning of the sequence
    num_steps = N_CTX // BLOCK_M1  # Process the entire sequence

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        CAUSAL_MASKING=False,
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    # Write-back dV
    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write-back dK (scaled)
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # ====== COMPUTE dQ ======
    # Prepare offsets for dQ computation
    start_m = start_block_id * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    # Load Q, DO, and M: They will stay in SRAM throughout.
    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Initialize dQ accumulator
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)

    # For non-causal attention, we process ALL key/value blocks (the entire sequence)
    # This means each query position can attend to ALL key/value positions

    start_n = 0  # Start from the beginning of the sequence
    num_steps = N_CTX // BLOCK_N2  # Process the entire sequence

    dq = _attn_bwd_dq(
        dq,
        q,
        K,
        V,  #
        do,
        m,
        D,  #
        stride_tok,
        stride_d,  #
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        start_n,
        num_steps,  #
        CAUSAL_MASKING=False,  #
        mask_ptr=mask_ptr,
        mask_stride_tok1=mask_stride_tok1,
        mask_stride_tok2=mask_stride_tok2,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    # Write-back dQ (scaled)
    dq *= LN2
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_h[None, :] * stride_d
    tl.store(dq_ptrs, dq)


@torch.library.custom_op("jvp_flash_attention::attn_fwd_dual_triton", mutates_args=())
def _attn_fwd_dual_triton(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    mask_tensor: Tensor,
    sm_scale: float,
    dropout_p: float,
    philox_seed: int,
    causal: bool,
    warp_specialize: bool,
    mask_type: int,
    min_sequence_length: int = MIN_SEQUENCE_LENGTH,
) -> tuple[Tensor, Tensor, Tensor]:
    """Opaque custom op wrapper for the dual Triton forward launch used under torch.compile.

    Args:
        q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
        v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
        q_t: Tangent query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        k_t: Tangent key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
        v_t: Tangent value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
        mask_tensor: Mask tensor for attention masking.
        sm_scale: Scaling factor for the softmax.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG used in dropout.
        causal: Whether to apply causal masking.
        warp_specialize: Whether to enable warp specialization in the Triton kernel.
        mask_type: Type of masking (0: no mask, 1: boolean mask,
                        2: additive mask).
        min_sequence_length: Minimum block size for the kernels; see `JVPAttn.forward`.

    Returns:
        A tuple containing:
        - Output tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        - Tangent output tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        - Memory tensor M of shape (Z, H, N_CTX) used for the backward pass.
    """
    Z, H, N_CTX, HEAD_DIM_Q = q.shape
    HEAD_DIM_K = k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    if HEAD_DIM_Q != HEAD_DIM_K or HEAD_DIM_K != HEAD_DIM_V:
        raise ValueError(
            "JVP attention requires HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V, "
            f"got ({HEAD_DIM_Q}, {HEAD_DIM_K}, {HEAD_DIM_V})."
        )

    STAGE = 3 if causal else 1
    ENABLE_DROPOUT = dropout_p > 0.0
    o = torch.empty_like(q)
    o_t = torch.empty_like(q_t)
    M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

    extra_kern_args = {}
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    if is_cuda() and warp_specialize:
        extra_kern_args["maxnreg"] = 168

    if hasattr(triton, "set_allocator") and is_cuda():

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

    def strides_zhnd(t: Tensor) -> tuple[int, int, int, int]:
        return (t.stride(0), t.stride(1), t.stride(2), t.stride(3))

    if mask_type == 0:
        mask_strides = (0, 0, 0, 0)
    else:
        mask_strides = (
            mask_tensor.stride(0),
            mask_tensor.stride(1),
            mask_tensor.stride(2),
            mask_tensor.stride(3),
        )

    Z_H = Z * H

    def grid(META: dict[str, Any]) -> tuple[int, int, int]:
        return (triton.cdiv(N_CTX, META["BLOCK_M"]), Z_H, 1)

    _attn_fwd[grid](
        q,
        k,
        v,
        q_t,
        k_t,
        v_t,  #
        sm_scale,
        M,
        o,
        o_t,  #
        mask_tensor,  #
        dropout_p,  #
        philox_seed,  #
        *strides_zhnd(q),  #
        *strides_zhnd(k),  #
        *strides_zhnd(v),  #
        *strides_zhnd(q_t),  #
        *strides_zhnd(k_t),  #
        *strides_zhnd(v_t),  #
        *strides_zhnd(o),  #
        *strides_zhnd(o_t),  #
        *mask_strides,  #
        Z,
        H,  #
        N_CTX=N_CTX,  #
        HEAD_DIM=HEAD_DIM_K,  #
        FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
        STAGE=STAGE,  #
        warp_specialize=warp_specialize,  #
        ENABLE_JVP=True,  #
        ENABLE_DROPOUT=ENABLE_DROPOUT,
        MASK_TYPE=mask_type,
        # NOTE: The following are safe (unit-tested) default values
        BLOCK_M=min_sequence_length,  #
        BLOCK_N=min_sequence_length,  #
        num_stages=NUM_STAGES_OPTIONS[0],  #
        num_warps=4,  #
        **extra_kern_args,
    )
    return o, o_t, M


@_attn_fwd_dual_triton.register_fake
def _attn_fwd_dual_triton_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    mask_tensor: Tensor,
    sm_scale: float,
    dropout_p: float,
    philox_seed: int,
    causal: bool,
    warp_specialize: bool,
    mask_type: int,
    min_sequence_length: int = MIN_SEQUENCE_LENGTH,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fake implementation of the dual Triton forward launch for compilation purposes. This
    implementation does not perform any actual computation and returns empty tensors with the
    appropriate shapes.

    Args:
        q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
        v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
        q_t: Tangent query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        k_t: Tangent key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
        v_t: Tangent value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
        mask_tensor: Mask tensor for attention masking.
        sm_scale: Scaling factor for the softmax.
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG used in dropout.
        causal: Whether to apply causal masking.
        warp_specialize: Whether to enable warp specialization in the Triton kernel.
        mask_type: Type of masking (0: no mask, 1: boolean mask,
                        2: additive mask).
        min_sequence_length: Minimum block size for the kernels; see `JVPAttn.forward`.

    Returns:
        A tuple containing:
        - An empty output tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        - An empty tangent output tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
        - An empty memory tensor M of shape (Z, H, N_CTX) used for the backward pass.
    """
    Z, H, N_CTX, _ = q.shape
    return (
        torch.empty_like(q),
        torch.empty_like(q_t),
        torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32),
    )


class JVPAttn(Function):
    """JVP (Jacobian-Vector Product) for Attention Mechanism."""

    class Grid(NamedTuple):
        """Grid configuration for JVP Attention."""

        M_BLOCKS: int
        Z_H: int
        ONE: Literal[1]

    class FnCtx(FunctionCtx):
        """Function context for JVP Attention."""

        sm_scale: float
        HEAD_DIM_K: int
        causal: bool
        grid: JVPAttn.Grid
        mask_tensor: Tensor
        MASK_TYPE: int
        dropout_p: float
        philox_seed: int
        ENABLE_DROPOUT: bool
        min_sequence_length: int

    class FwdOutCtxContrib(NamedTuple):
        """Forward output context contributions for JVP Attention."""

        o_t: Tensor | None
        M: Tensor
        grid: JVPAttn.Grid
        HEAD_DIM_K: int
        sm_scale: float
        mask_tensor: Tensor
        MASK_TYPE: int
        dropout_p: float
        philox_seed: int
        ENABLE_DROPOUT: bool
        n_q: int
        n_kv: int
        pad_mode: str | None

    class FwdOut(NamedTuple):
        """Forward output for JVP Attention."""

        o: Tensor
        ctx: JVPAttn.FwdOutCtxContrib

    class JVPOut(NamedTuple):
        """JVP output for JVP Attention."""

        o: Tensor
        ctx: None

    class BwdOut(NamedTuple):
        """Backward output for JVP Attention."""

        q: Tensor
        k: Tensor
        v: Tensor
        q_t: None
        k_t: None
        v_t: None
        attn_mask: None
        dropout_p: None
        causal: None
        sm_scale: None
        warp_specialize: None
        USE_TMA: None
        verify_attn_mask: None
        USE_NAIVE: None
        min_sequence_length: None

    class Strides(NamedTuple):
        """Strides for JVP Attention."""

        z: int
        h: int
        n_ctx: int
        head_dim: int

    @staticmethod
    def forward(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_t: Tensor | None,
        k_t: Tensor | None,
        v_t: Tensor | None,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        sm_scale: float | None = None,
        warp_specialize: bool = True,
        USE_TMA: bool = True,
        verify_attn_mask: bool = True,
        USE_NAIVE: bool | None = None,
        min_sequence_length: int = MIN_SEQUENCE_LENGTH,
    ) -> JVPAttn.FwdOut:
        """Forward pass for JVP Attention.

        NOTE: The following warning(s) will be raised if `verify_attn_mask=True`
        and an attention mask with any all-null head is provided:
            `RuntimeWarning: overflow encountered in exp2.`

        Args:
            q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
            k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
            v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
            q_t: Optional tensor for query transpose.
            k_t: Optional tensor for key transpose.
            v_t: Optional tensor for value transpose.
            attn_mask: Optional attention mask of shape (Z, H, N_CTX, N_CTX).
                Two types of masks are supported. A boolean mask where a value
                of True indicates that the element should take part in attention,
                or a float mask of the same type as query, key, value that is added
                to the attention score. The constant `MASK_CONST` is used to
                indicate masked positions in the float mask. All other values
                denote unmasked positions.
            dropout_p: Dropout probability.
            causal: Whether the attention is causal.
            sm_scale: Optional scaling factor for softmax.
            warp_specialize: Whether to use warp specialization.
            USE_TMA: Whether to use TMA.
            verify_attn_mask: Whether to verify the correctness of the provided attention mask.
            USE_NAIVE: Whether to force the pure-PyTorch (True) or Triton (False)
                implementation, or to choose automatically (None). Rectangular
                queries/keys (n_q != n_kv) always use the pure-PyTorch fallback.
            min_sequence_length: The block size (and minimum sequence length) used
                by the Triton kernels; shorter or non-multiple sequence lengths
                are padded internally.

        Returns:
            Outputs of JVP Attention.
        """
        if dropout_p != 0.0:
            raise NotImplementedError("Dropout is not currently supported in JVP attention.")

        # Collect metadata
        Z, H, N_CTX, HEAD_DIM_Q = q.shape
        n_q = N_CTX
        n_kv = k.shape[2]
        HEAD_DIM_K = k.shape[-1]
        HEAD_DIM_V = v.shape[-1]  # NOTE: When v is in float8_e5m2 it is transposed.

        STAGE = 3 if causal else 1
        ENABLE_JVP = q_t is not None

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V, (
            "JVP attention requires HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V"
            f" but got HEAD_DIM_Q={HEAD_DIM_Q}, HEAD_DIM_K={HEAD_DIM_K}, HEAD_DIM_V={HEAD_DIM_V}"
        )
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}, (
            "JVP attention only supports HEAD_DIM_K in {16, 32, 64, 128, 256},"
            f" but got HEAD_DIM_K={HEAD_DIM_K}",
        )
        assert (
            n_kv >= n_q
        ), f"JVP attention requires key length >= query length, but got {n_kv} and {n_q}"
        if attn_mask is not None:
            assert n_kv == n_q, (
                "JVP attention requires q and k to have the same sequence length when an"
                f" attention mask is provided, but got {n_q} and {n_kv}"
            )

        if causal and attn_mask is not None:
            raise ValueError("Causal attention does not support an attention mask.")

        # Initialize arguments and tensors
        if sm_scale is None:
            sm_scale = HEAD_DIM_K**-0.5

        # pure pytorch fallback (no triton / cuda); handles any sequence length natively
        mask_tensor, MASK_TYPE = prepare_attn_mask(attn_mask, q, verify_attn_mask)
        # NOTE: Padding cached queries up to the KV length would make decoding
        # quadratic; the tiled path keeps rectangular Q/KV lengths instead.
        if n_q != n_kv:
            USE_NAIVE = True

        if use_naive_attention(q, USE_NAIVE):
            o, o_t, M = tiled_attention(
                q,
                k,
                v,
                mask_tensor,
                MASK_TYPE,
                causal,
                sm_scale,
                q_t,
                k_t,
                v_t,
            )

            return (  # was JVPAttn.FwdOut
                o,
                JVPAttn.FwdOutCtxContrib(  # was JVPAttn.FwdOutCtxContrib
                    o_t,
                    M,
                    None,
                    HEAD_DIM_K,
                    sm_scale,
                    mask_tensor,
                    MASK_TYPE,
                    dropout_p,
                    0,
                    False,
                    n_q,
                    n_kv,
                    None,
                ),
            )

        # NOTE: The Triton kernels require the sequence length to be at least
        # min_sequence_length and a multiple of it; short (or mismatched) sequences are
        # padded up and the padding is excised from the outputs afterwards.
        pad_mode = pad_mode_for(n_q, n_kv, attn_mask is not None, min_sequence_length)
        if pad_mode is not None:
            left = pad_mode == "left"
            N_CTX = min_padded_length(max(n_q, n_kv), min_sequence_length)
            q = pad_to_length(q, N_CTX, left=left)
            k = pad_to_length(k, N_CTX, left=left)
            v = pad_to_length(v, N_CTX, left=left)
            if ENABLE_JVP:
                q_t = pad_to_length(q_t, N_CTX, left=left)
                k_t = pad_to_length(k_t, N_CTX, left=left)
                v_t = pad_to_length(v_t, N_CTX, left=left)

            if left:
                # Encode the (optionally causal) query-as-tail alignment in a boolean mask.
                mask_tensor = build_internal_mask(n_q, n_kv, N_CTX, causal, q.device)
                mask_tensor = mask_tensor.expand(Z, H, N_CTX, N_CTX).contiguous()
                MASK_TYPE = 1
                causal = False
                STAGE = 1
            else:
                mask_tensor = pad_attn_mask(mask_tensor, MASK_TYPE, N_CTX)

        o = torch.empty_like(q)
        o_t: Tensor | None = torch.empty_like(q_t) if ENABLE_JVP else None
        M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

        # Tune kernel for custom (e.g., AMD) targets
        extra_kern_args = {}

        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        if is_cuda() and warp_specialize:
            # NOTE: We need more registers if we're doing JVP
            if (HEAD_DIM_K == 128 and q.dtype == torch.float16) or ENABLE_JVP:
                extra_kern_args["maxnreg"] = 168
            else:
                # NOTE: For backward pass with HEAD_DIM_K=128, this is probably too low for H100; register allocation fails.
                extra_kern_args["maxnreg"] = 80

        if hasattr(triton, "set_allocator") and is_cuda():

            def alloc_fn(size: int, align: int, _):
                """Custom allocator function for Triton."""
                return torch.empty(size, dtype=torch.int8, device="cuda")

            triton.set_allocator(alloc_fn)

        def strides_zhnd(t: Tensor) -> tuple[int, int, int, int]:
            """Get strides for a tensor with shape (Z, H, N_CTX, HEAD_DIM)."""
            return (t.stride(0), t.stride(1), t.stride(2), t.stride(3))  # was JVPAttn.Strides

        # Determine stride configuration for the mask tensor
        mask_strides = (0, 0, 0, 0) if MASK_TYPE == 0 else strides_zhnd(mask_tensor)

        # Prepare dropout arguments
        ENABLE_DROPOUT = dropout_p > 0.0
        if ENABLE_DROPOUT:
            philox_seed = torch.randint(0, 2**32, (1,), device=q.device, dtype=torch.int64).item()
        else:
            philox_seed = 0

        # Set up grid for kernel launch
        Z_H = Z * H

        def grid(META: dict[str, Any]) -> tuple[int, int, int]:
            """Determine grid configuration."""
            return (triton.cdiv(N_CTX, META["BLOCK_M"]), Z_H, 1)  # was JVPAttn.Grid

        if ENABLE_JVP and _is_compiling():
            if q_t is None or k_t is None or v_t is None:
                raise RuntimeError(
                    "Expected dual tangents q_t, k_t, v_t when compiling JVP attention."
                )
            with torch.no_grad():
                o, o_t, M = _attn_fwd_dual_triton(
                    q,
                    k,
                    v,
                    q_t,
                    k_t,
                    v_t,
                    mask_tensor,
                    sm_scale,
                    dropout_p,
                    philox_seed,
                    causal,
                    warp_specialize,
                    MASK_TYPE,
                    min_sequence_length,
                )
        elif USE_TMA and supports_tma():
            # NOTE: On Hopper, we cannot perform a FP8 dot with a non-transposed second tensor.
            y_dim = Z_H * N_CTX
            tma_block_shape = [min_sequence_length, HEAD_DIM_K]

            desc_q = TensorDescriptor(
                q,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=tma_block_shape,
            )
            desc_q_t = (
                desc_q
                if q_t is None
                else TensorDescriptor(
                    q_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=tma_block_shape,
                )
            )

            if q.dtype == torch.float8_e5m2:
                v_shape = [HEAD_DIM_K, y_dim]
                v_strides = [N_CTX, 1]
            else:
                v_shape = [y_dim, HEAD_DIM_K]
                v_strides = [HEAD_DIM_K, 1]
            desc_v = TensorDescriptor(
                v, shape=v_shape, strides=v_strides, block_shape=tma_block_shape
            )
            # NOTE: Probably we could share the shape and strides from above, but whatever
            if q_t is not None and q_t.dtype == torch.float8_e5m2:
                t_v_shape = [HEAD_DIM_K, y_dim]
                t_v_strides = [q_t.shape[2], 1]
            else:
                t_v_shape = [y_dim, HEAD_DIM_K]
                t_v_strides = [HEAD_DIM_K, 1]
            desc_v_t = (
                desc_v
                if v_t is None
                else TensorDescriptor(
                    v_t, shape=t_v_shape, strides=t_v_strides, block_shape=tma_block_shape
                )
            )

            desc_k = TensorDescriptor(
                k,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=tma_block_shape,
            )
            desc_k_t = (
                desc_k
                if k_t is None
                else TensorDescriptor(
                    k_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=tma_block_shape,
                )
            )

            desc_o = TensorDescriptor(
                o,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=tma_block_shape,
            )
            desc_o_t = (
                desc_o
                if o_t is None
                else TensorDescriptor(
                    o_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=tma_block_shape,
                )
            )

            _attn_fwd_tma[grid](
                sm_scale,
                M,  #
                Z,
                H,  #
                desc_q,
                desc_k,
                desc_v,  #
                desc_q_t,
                desc_k_t,
                desc_v_t,  #
                desc_o,
                desc_o_t,  #
                mask_tensor,  #
                dropout_p,  #
                philox_seed,  #
                *mask_strides,  #
                N_CTX=N_CTX,  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=STAGE,  #
                warp_specialize=warp_specialize,  #
                ENABLE_JVP=ENABLE_JVP,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                MASK_TYPE=MASK_TYPE,
                # NOTE: The following are safe (unit-tested) default values
                BLOCK_M=min_sequence_length,  #
                BLOCK_N=min_sequence_length,  #
                num_stages=NUM_STAGES_OPTIONS[0],  #
                num_warps=4,  #
                **extra_kern_args,
            )

        else:
            _attn_fwd[grid](
                q,
                k,
                v,
                q_t,
                k_t,
                v_t,  #
                sm_scale,
                M,
                o,
                o_t,  #
                mask_tensor,  #
                dropout_p,  #
                philox_seed,  #
                *strides_zhnd(q),  #
                *strides_zhnd(k),  #
                *strides_zhnd(v),  #
                *strides_zhnd(q if q_t is None else q_t),  #
                *strides_zhnd(k if k_t is None else k_t),  #
                *strides_zhnd(v if v_t is None else v_t),  #
                *strides_zhnd(o),  #
                *strides_zhnd(o if o_t is None else o_t),  #
                *mask_strides,  #
                Z,
                H,  #
                N_CTX=N_CTX,  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=STAGE,  #
                warp_specialize=warp_specialize,  #
                ENABLE_JVP=ENABLE_JVP,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                MASK_TYPE=MASK_TYPE,
                # NOTE: The following are safe (unit-tested) default values
                BLOCK_M=min_sequence_length,  #
                BLOCK_N=min_sequence_length,  #
                num_stages=NUM_STAGES_OPTIONS[0],  #
                num_warps=4,  #
                **extra_kern_args,
            )

        return (  # was JVPAttn.FwdOut
            o,
            JVPAttn.FwdOutCtxContrib(  # was JVPAttn.FwdOutCtxContrib
                o_t,
                M,
                grid,
                HEAD_DIM_K,
                sm_scale,
                mask_tensor,
                MASK_TYPE,
                dropout_p,
                philox_seed,
                ENABLE_DROPOUT,
                n_q,
                n_kv,
                pad_mode,
            ),
        )

    @staticmethod
    def setup_context(ctx: JVPAttn.FnCtx, inputs, outputs: JVPAttn.FwdOut) -> Tensor:
        """Set up the context for JVP Attention.

        Args:
            ctx: The context to set up
            inputs: The input tensors
            outputs: The output tensors
        """
        (
            q,
            k,
            v,
            q_t,
            k_t,
            v_t,
            attn_mask,
            dropout_p,
            causal,
            sm_scale,
            warp_specialize,
            USE_TMA,
            verify_attn_mask,
            USE_NAIVE,
            min_sequence_length,
        ) = inputs

        o, (
            o_t,
            M,
            grid,
            HEAD_DIM_K,
            sm_scale,
            mask_tensor,
            MASK_TYPE,
            dropout_p,
            philox_seed,
            ENABLE_DROPOUT,
            n_q,
            n_kv,
            pad_mode,
        ) = outputs

        # Pure-PyTorch tiled execution returns grid=None; Triton returns the launch grid.
        ctx.use_naive = grid is None
        ctx.grid = grid
        ctx.save_for_forward(o_t)
        ctx.save_for_backward(q, k, v, o, M)

        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM_K = HEAD_DIM_K
        # NOTE: For padded pathways, causality is encoded in the mask itself, so the
        # backward pass must run the (mask-driven) non-causal kernel.
        ctx.causal = causal if pad_mode is None else False
        ctx.mask_tensor = mask_tensor
        ctx.MASK_TYPE = MASK_TYPE
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.ENABLE_DROPOUT = ENABLE_DROPOUT
        ctx.min_sequence_length = min_sequence_length
        ctx.n_q = n_q
        ctx.n_kv = n_kv
        ctx.pad_mode = pad_mode

    @staticmethod
    def fwd(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        sm_scale: float | None = None,
        warp_specialize: bool = True,
        USE_TMA: bool = True,
        verify_attn_mask: bool = True,
        USE_NAIVE: bool | None = None,
        min_sequence_length: int = MIN_SEQUENCE_LENGTH,
    ) -> Tensor:
        """Forward pass for JVP Attention.

        NOTE: This is not an autograd convention. It's a workaround to get type-hinting and kwarg support.

        NOTE: Calls to `contiguous()` are necessary to ensure the inputs are contiguous in memory
        (e.g., due to an `unbind` call to create `q`, `k`, `v`) but nonetheless may incur a performance cost.

        Args:
            q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
            k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
            v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
            attn_mask: Optional attention mask of shape (Z, H, N_CTX, N_CTX). Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score.
            dropout_p: Dropout probability.
            causal: Whether to use causal attention.
            sm_scale: The softmax scale factor.
            warp_specialize: Whether to use warp specialization.
            USE_TMA: Whether to use TMA.
            verify_attn_mask: Whether to verify the correctness of the provided attention mask.
            USE_NAIVE: Whether to force the pure-PyTorch (True) or Triton (False)
                implementation, or to choose automatically (None). Rectangular queries/keys
                (n_q != n_kv) always use the pure-PyTorch fallback.
            min_sequence_length: The block size (and minimum sequence length) used
                by the Triton kernels; shorter or non-multiple sequence lengths
                are padded internally.

        Returns:
            The output tensor.
        """
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        out: JVPAttn.FwdOut = JVPAttn.apply(
            q,
            k,
            v,
            None,
            None,
            None,
            attn_mask,
            dropout_p,
            causal,
            sm_scale,
            warp_specialize,
            USE_TMA,
            verify_attn_mask,
            USE_NAIVE,
            min_sequence_length,
        )

        a, ctx_contrib = out
        return excise_padding(a, ctx_contrib.n_q, ctx_contrib.pad_mode)

    @staticmethod
    def fwd_dual(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = False,
        sm_scale: float | None = None,
        warp_specialize: bool = True,
        USE_TMA: bool = True,
        verify_attn_mask: bool = True,
        USE_NAIVE: bool | None = None,
        min_sequence_length: int = MIN_SEQUENCE_LENGTH,
    ) -> Tensor:
        """Forward pass for JVP Attention with dual tensor inputs.

        NOTE: This is not an autograd convention. It's a workaround to get type-hinting and kwarg support.

        NOTE: Calls to `contiguous()` are necessary to ensure the inputs are contiguous in memory
        (e.g., due to an `unbind` call to create `q`, `k`, `v`) but nonetheless may incur a performance cost.

        Args:
            q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
            k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
            v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
            attn_mask: Optional attention mask of shape (Z, H, N_CTX, N_CTX). Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score.
            dropout_p: Dropout probability.
            causal: Whether to use causal attention.
            sm_scale: The softmax scale factor.
            warp_specialize: Whether to use warp specialization.
            USE_TMA: Whether to use TMA.
            verify_attn_mask: Whether to verify the correctness of the provided attention mask.
            USE_NAIVE: Whether to force the pure-PyTorch (True) or Triton (False)
                implementation, or to choose automatically (None). Rectangular queries/keys
                (n_q != n_kv) always use the pure-PyTorch fallback.
            min_sequence_length: The block size (and minimum sequence length) used
                by the Triton kernels; shorter or non-multiple sequence lengths
                are padded internally.

        Returns:
            The output tensor.
        """
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        q_p, q_t = fwAD.unpack_dual(q)
        k_p, k_t = fwAD.unpack_dual(k)
        v_p, v_t = fwAD.unpack_dual(v)

        has_tangents = q_t is not None or k_t is not None or v_t is not None
        if has_tangents:
            q_t = torch.zeros_like(q_p) if q_t is None else q_t.contiguous()
            k_t = torch.zeros_like(k_p) if k_t is None else k_t.contiguous()
            v_t = torch.zeros_like(v_p) if v_t is None else v_t.contiguous()

        if _is_compiling() and has_tangents:
            o, ctx_contrib = JVPAttn.forward(
                q_p,
                k_p,
                v_p,
                q_t,
                k_t,
                v_t,
                attn_mask,
                dropout_p,
                causal,
                sm_scale,
                warp_specialize,
                USE_TMA,
                verify_attn_mask,
                USE_NAIVE,
                min_sequence_length,
            )
            o = excise_padding(o, ctx_contrib.n_q, ctx_contrib.pad_mode)
            o_t = (
                None
                if ctx_contrib.o_t is None
                else excise_padding(ctx_contrib.o_t, ctx_contrib.n_q, ctx_contrib.pad_mode)
            )
            return fwAD.make_dual(o, o_t)

        # NOTE: We pass some dualtensor args to ensure jvp() will be called,
        # but we also pass tangents separately, as forward() demotes dual
        # tensor args to primals for some reason.
        out: JVPAttn.FwdOut = JVPAttn.apply(
            q,
            k,
            v,
            q_t,
            k_t,
            v_t,
            attn_mask,
            dropout_p,
            causal,
            sm_scale,
            warp_specialize,
            USE_TMA,
            verify_attn_mask,
            USE_NAIVE,
            min_sequence_length,
        )

        a, ctx_contrib = out
        return excise_padding(a, ctx_contrib.n_q, ctx_contrib.pad_mode)

    @staticmethod
    def jvp(ctx: JVPAttn.FnCtx, gq: Tensor, gk: Tensor, gv: Tensor, *_) -> JVPAttn.JVPOut:
        """Compute the Jacobian-vector product (JVP) for JVP Attention.

        Args:
            ctx: The context
            gq: The gradient of the query tensor
            gk: The gradient of the key tensor
            gv: The gradient of the value tensor

        Returns:
            The JVP output.
        """
        return (ctx.saved_for_forward[0], None)  # was JVPAttn.JVPOut

    @staticmethod
    def backward(ctx, do, _) -> JVPAttn.BwdOut:
        """Backward pass for JVP Attention.

        NOTE: A call to `contiguous()` may be necessary to ensure the output derivatives are contiguous
        in memory (e.g., due to autograd weirdness) but nonetheless may incur a performance cost.

        Args:
            ctx: The context
            do: The gradient of the output tensor

        Returns:
            The backward output.
        """
        q, k, v, o, M = ctx.saved_tensors

        def unwrap(x):
            """Unwrap a functorch tensor if it is wrapped."""
            if (
                isinstance(x, torch.Tensor)
                and hasattr(torch, "_C")
                and hasattr(torch._C, "_functorch")
                and torch._C._functorch.is_functorch_wrapped_tensor(x)
            ):
                return torch._C._functorch.get_unwrapped(x)
            return x

        # pure pytorch backward (no triton / cuda)
        if ctx.use_naive:
            if torch.is_grad_enabled():
                # NOTE: Recompute the forward pass with a graph so that double
                # backward sees the derivatives of the output and normalizer.
                o, _, M = tiled_attention(
                    q, k, v, ctx.mask_tensor, ctx.MASK_TYPE, ctx.causal, ctx.sm_scale
                )

            dq, dk, dv = tiled_attention_backward(
                q,
                k,
                v,
                o,
                do,
                M,
                ctx.mask_tensor,
                ctx.MASK_TYPE,
                ctx.causal,
                ctx.sm_scale,
            )

            return (  # was JVPAttn.BwdOut
                dq,
                dk,
                dv,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        # NOTE: For padded pathways, restore the padded layout the kernels expect.
        if ctx.pad_mode is not None:
            left = ctx.pad_mode == "left"
            length = M.shape[-1]
            q = pad_to_length(unwrap(q), length, left=left)
            k = pad_to_length(unwrap(k), length, left=left)
            v = pad_to_length(unwrap(v), length, left=left)
            o = pad_to_length(unwrap(o), length, left=left)
            do = pad_to_length(unwrap(do), length, left=left)

        # Ensure inputs/outputs the kernel reads share the same (contiguous) layout
        if not (
            q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and o.is_contiguous()
        ):
            raise ValueError(
                "JVPAttn expected q, k, v, o to be contiguous; got "
                f"q.is_contiguous()={q.is_contiguous()}, k.is_contiguous()={k.is_contiguous()}, "
                f"v.is_contiguous()={v.is_contiguous()}, o.is_contiguous()={o.is_contiguous()}, "
                f"do.is_contiguous()={do.is_contiguous()}"
            )

        # NOTE: Autograd may deliver a non-contiguous output gradient; if so, normalize it.
        if not do.is_contiguous():
            do = do.contiguous()

        # Ensure all inputs/outputs the kernel reads share the same layout
        assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride(), (
            "JVPAttn expected q, k, v, o, do to have the same layout; got "
            f"q.stride()={q.stride()}, k.stride()={k.stride()}, v.stride()={v.stride()}, "
            f"o.stride()={o.stride()}, do.stride()={do.stride()}"
        )

        # Initialize tensors for gradients
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(M)

        # Collect metadata
        Z, H, N_CTX = q.shape[:3]

        BLK_SLICE_FACTOR = 2  # NOTE: This is a safe default value to reduce backward memory usage
        BLOCK_MIN = ctx.min_sequence_length
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = BLOCK_MIN, BLOCK_MIN, BLOCK_MIN, BLOCK_MIN

        assert N_CTX % BLOCK_MIN == 0, f"N_CTX must be divisible by BLOCK_MIN={BLOCK_MIN}"

        if not ctx.causal:
            assert (
                BLOCK_M1 == BLOCK_M2 == BLOCK_N1 == BLOCK_N2
            ), "For non-causal attention, all block sizes must be equal."

        # Scale k by sm_scale / ln(2) to account for softmax scaling and
        # change-of-base of exponentiation (exp2).
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)

        # Determine mask type
        if ctx.MASK_TYPE == 0:
            mask_strides = (0, 0, 0, 0)
        else:
            mask_strides = (
                ctx.mask_tensor.stride(0),
                ctx.mask_tensor.stride(1),
                ctx.mask_tensor.stride(2),
                ctx.mask_tensor.stride(3),
            )

        # Set up grid for kernel launch
        Z_H = Z * H

        # Preprocess output's deltas
        pre_grid = (N_CTX // BLOCK_MIN, Z_H)
        _attn_bwd_preprocess[pre_grid](
            unwrap(o),
            do,  #
            delta,  #
            N_CTX,  #
            BLOCK_M=BLOCK_MIN,
            HEAD_DIM=ctx.HEAD_DIM_K,  #
        )

        # Launch the backward kernel, enabling pipelining for backward pass on A100s
        grid = (N_CTX // BLOCK_MIN, Z_H)
        bwd_kernel = _attn_bwd_causal if ctx.causal else _attn_bwd
        num_stages = (
            5
            if is_cuda() and torch.cuda.get_device_capability()[0] == 9
            else NUM_STAGES_OPTIONS[0]
        )

        bwd_kernel[grid](
            unwrap(q),
            arg_k,
            unwrap(v),
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            unwrap(M),
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            mask_strides[0],
            mask_strides[1],
            mask_strides[2],
            mask_strides[3],  #
            H,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM_K,  #
            mask_ptr=unwrap(ctx.mask_tensor),
            MASK_TYPE=ctx.MASK_TYPE,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            ENABLE_DROPOUT=ctx.ENABLE_DROPOUT,
            # NOTE: The following are safe (unit-tested) default values
            num_stages=num_stages,  #
            num_warps=4,  #
        )

        return (  # was JVPAttn.BwdOut
            excise_padding(dq, ctx.n_q, ctx.pad_mode),
            excise_padding(dk, ctx.n_kv, ctx.pad_mode),
            excise_padding(dv, ctx.n_kv, ctx.pad_mode),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


attention = JVPAttn.fwd
