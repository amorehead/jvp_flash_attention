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
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import FunctionCtx

try:
    from triton.tools.tensor_descriptor import TensorDescriptor

    HAS_TENSOR_DESC = True
except ModuleNotFoundError:
    HAS_TENSOR_DESC = False

MIN_SEQUENCE_LENGTH = 32  # NOTE: All sequence lengths must be multiples of 2 >= 32


def is_hip():
    """Check if the current device is HIP."""
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    """Check if the current device is CUDA."""
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    """Check if the current device supports host tensor descriptors."""
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def supports_tma():
    """Check if the current device supports Tensor Memory Access (TMA)."""
    return HAS_TENSOR_DESC and is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    """Check if the current device is Blackwell architecture."""
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


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
    # Add mask and dropout parameters
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

    Returns:
        The output tensors as a tuple.
    """
    # Range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # NOTE: In fp8 mode, we may want to advance the V_block_ptr differently.
    # I did try advancing by (0, lo) instead for fp8, but I got an illegal memory access.
    # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    if ENABLE_JVP:
        T_K_block_ptr = tl.advance(T_K_block_ptr, (0, lo))
        T_V_block_ptr = tl.advance(T_V_block_ptr, (lo, 0))

    # Loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- Compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        if ENABLE_JVP:
            t_k = tl.load(T_K_block_ptr)
            t_qk = tl.dot(t_q, k) + tl.dot(q, t_k)

        # Load and apply attention mask if provided (before scaling for STAGE != 2)
        if STAGE != 2 and MASK_TYPE > 0:
            mask_offs = offs_m[:, None] * N_CTX + (start_n + offs_n[None, :])
            if MASK_TYPE == 1:  # Boolean mask
                mask = tl.load(mask_block_ptr + mask_offs, mask=mask_offs < N_CTX * N_CTX)
                # Convert boolean to additive mask: True (attend) -> 0, False (ignore) -> -inf
                mask_value = tl.where(mask, 0.0, -1.0e6)
                qk = qk * qk_scale + mask_value
            elif MASK_TYPE == 2:  # Additive mask
                mask = tl.load(
                    mask_block_ptr + mask_offs, mask=mask_offs < N_CTX * N_CTX, other=0.0
                )
                qk = qk * qk_scale + mask
            else:
                qk = qk * qk_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        elif STAGE == 2:
            # For causal attention (STAGE == 2)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)

            # Apply additional mask if provided
            if MASK_TYPE > 0:
                mask_offs = offs_m[:, None] * N_CTX + (start_n + offs_n[None, :])
                if MASK_TYPE == 1:  # Boolean mask
                    add_mask = tl.load(mask_block_ptr + mask_offs, mask=mask_offs < N_CTX * N_CTX)
                    # Only apply additional masking where causal mask allows attention
                    mask_value = tl.where(add_mask, 0.0, -1.0e6)
                    qk = tl.where(mask, qk + mask_value, qk)
                elif MASK_TYPE == 2:  # Additive mask
                    add_mask = tl.load(
                        mask_block_ptr + mask_offs, mask=mask_offs < N_CTX * N_CTX, other=0.0
                    )
                    qk = tl.where(mask, qk + add_mask, qk)

            if ENABLE_JVP:
                # Tangents should be masked with 0.0 since they represent derivatives
                t_qk = tl.where(mask, t_qk, 0.0)
            # TODO: Do we need a separate row maximum for qk_t?
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            # No masking case (STAGE != 2 and MASK_TYPE == 0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)

        # Apply dropout if enabled
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, N_CTX
            )
            p = p * dropout_mask.to(dtype) * dropout_scale

        l_ij = tl.sum(p, 1)
        # -- Update m_i and l_i
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
        # Update acc
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
            p_tv_acc = p_tv_acc * alpha[:, None] + tl.dot(p, t_v.to(dtype)).to(t_v.dtype)
            T_V_block_ptr = tl.advance(T_V_block_ptr, (BLOCK_N, 0))
            T_K_block_ptr = tl.advance(T_K_block_ptr, (0, BLOCK_N))
        acc = tl.dot(p, v.to(dtype), acc).to(acc.dtype)
        # Update m_i and l_i
        m_i = m_ij
        # The fp8 PR made a change to how K and V are advanced here but I believe we already have that.
        # https://github.com/triton-lang/triton/commit/75d27b0b425329bad8c13b9cd47177d93590ec31
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
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

    Returns:
        The output tensors as a tuple.
    """
    # Range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # Loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- Compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if ENABLE_JVP:
            t_k = desc_k_t.load([offsetk_y, 0]).T
            t_qk = tl.dot(t_q, k) + tl.dot(q, t_k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            if ENABLE_JVP:
                # Claude says "tangents should be masked with 0.0 since they represent derivatives".
                t_qk = tl.where(mask, t_qk, 0.0)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- Compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- Update output accumulator --
        if warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
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
            p_tv_acc = p_tv_acc * alpha[:, None] + tl.dot(p, t_v.to(dtype)).to(t_v.dtype)
        # NOTE: This non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v.to(dtype), acc).to(acc.dtype)
        # Update m_i and l_i
        # Place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
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


@triton.autotune(
    configs=list(filter(keep, configs)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
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
        BLOCK_M: Block M dimension.
        BLOCK_N: Block N dimension.
        FP8_OUTPUT: FP8 output flag.
        STAGE: Stage.
        warp_specialize: Warp specialization flag.
        ENABLE_JVP: Enable JVP flag.
        ENABLE_DROPOUT: Enable dropout flag.
        MASK_TYPE: Mask type (0: no mask, 1: boolean, 2: additive).
    """
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float32  # For dot products
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    # Prepare mask pointer if mask is provided
    if MASK_TYPE > 0:
        mask_offset = off_z * stride_mz + off_h * stride_mh
        mask_block_ptr = Mask + mask_offset
    else:
        mask_block_ptr = None

    # Generate philox offset for this block
    philox_offset_base = off_hz * N_CTX * N_CTX

    # Block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
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
        strides=(stride_kk, stride_kn),
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
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
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
        # Could probably just reuse v_order here
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
            strides=(stride_tkk, stride_tkn),
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
        # Load q_t: it will stay in SRAM throughout
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
    # Load scales
    qk_scale = sm_scale
    qk_scale = qk_scale * 1.44269504  # 1/log(2)
    # Load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # Stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
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
    # Stage 2: on-band
    if STAGE & 2:
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
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))
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


@triton.autotune(
    configs=list(filter(keep_tma, configs_tma)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
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
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    warp_specialize: tl.constexpr,  #
    ENABLE_JVP: tl.constexpr,  #
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
        N_CTX: Context length.
        HEAD_DIM: Dimension of each head.
        BLOCK_M: Block size for M dimension.
        BLOCK_N: Block size for N dimension.
        FP8_OUTPUT: Flag indicating if FP8 output is used.
        STAGE: Stage of the computation.
        warp_specialize: Flag indicating if warp specialization is used.
        ENABLE_JVP: Flag indicating if JVP (Jacobian-vector product) is enabled.
    """
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float32  # For dot products
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
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
    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
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
        # Load t_q: it will stay in SRAM throughout
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
    # Load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # Load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])
    # Stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
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
        )
    # Stage 2: on-band
    if STAGE & 2:
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
        )
    # Epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))
    if ENABLE_JVP:
        t_p_v = g_acc / l_i[:, None] - (mu_i / l_i)[:, None] * acc
        t_y_out = t_p_v + p_tv_acc / l_i[:, None]
        desc_o_t.store([qo_offset_y, 0], t_y_out.to(dtype))


@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, Z, H, N_CTX, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):  # noqa: E741
    """Preprocess the inputs for the backward attention pass.

    Args:
        O: Output tensor.
        DO: Gradient of the output tensor.
        Delta: Accumulated gradients.
        Z: Input tensor.
        H: Head dimension.
        N_CTX: Context length.
        BLOCK_M: Block size for M dimension.
        HEAD_DIM: Head dimension size.
    """
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # Load
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(
        tl.float32
    )
    delta = tl.sum(o * do, axis=1)
    # Write-back
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,  #
    Q,
    k,
    v,
    sm_scale,  #
    DO,  #
    M,
    D,  #
    # shared by Q/K/V/DO.
    stride_tok,
    stride_d,  #
    H,
    N_CTX,
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,  #
    # Filled in by the wrapper.
    start_n,
    start_m,
    num_steps,  #
    MASK: tl.constexpr,
    # Args for masking/dropout
    mask_ptr,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    ENABLE_DROPOUT: tl.constexpr,
):
    """The main inner-loop logic for computing dK and dV.

    Args:
        dk: Gradient of the key tensor.
        dv: Gradient of the value tensor.
        Q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        sm_scale: Scale factor for the softmax.
        DO: Gradient of the output tensor.
        M: Memory tensor.
        D: Delta tensor.
        stride_tok: Stride for the token dimension.
        stride_d: Stride for the head dimension.
        H: Head dimension.
        N_CTX: Context length.
        BLOCK_M1: Block size for M dimension.
        BLOCK_N1: Block size for N dimension.
        HEAD_DIM: Head dimension size.
        start_n: Starting index for N dimension.
        start_m: Starting index for M dimension.
        num_steps: Number of steps to unroll.
        MASK: Masking tensor.
        mask_ptr: Pointer to the mask tensor.
        MASK_TYPE: Type of masking (0: no mask, 1: boolean mask,
                     2: additive mask).
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        ENABLE_DROPOUT: Flag to enable dropout.

    Returns:
        dk: Gradient of the key tensor.
        dv: Gradient of the value tensor.
    """
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tok + offs_k[:, None] * stride_d
    do_ptrs = DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    # NOTE: BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    dtype = tl.float32  # For dot products
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        m = tl.load(M + offs_m)
        qkT = tl.dot(k, qT)
        # Causal masking before exponentiation.
        if MASK:
            causal_mask = offs_m[None, :] >= offs_n[:, None]
            qkT = tl.where(causal_mask, qkT, -1e6)
        # External masking before exponentiation.
        if MASK_TYPE > 0:
            mask_offs = offs_m[None, :] * N_CTX + offs_n[:, None]
            if MASK_TYPE == 1:
                mask = tl.load(mask_ptr + mask_offs)
                qkT = tl.where(mask, qkT, -1e6)
            elif MASK_TYPE == 2:
                add_mask = tl.load(mask_ptr + mask_offs)
                qkT += add_mask
        # Exponentiation.
        pT = tl.math.exp2(qkT - m[None, :])
        # Dropout after exponentiation.
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + curr_m * N_CTX + start_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M1, BLOCK_N1, N_CTX
            )
            pT = pT * dropout_mask.to(pT.dtype) * dropout_scale
        do = tl.load(do_ptrs)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(dtype)
        dv += tl.dot(ppT, do.to(dtype)).to(do.dtype)
        # NOTE: D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(dtype)
        dk += tl.dot(dsT, tl.trans(qT).to(dtype)).to(qT.dtype)
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_tok
        do_ptrs += step_m * stride_tok
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
    H,
    N_CTX,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_m,
    start_n,
    num_steps,  #
    MASK: tl.constexpr,
    # Args for masking/dropout
    mask_ptr,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    ENABLE_DROPOUT: tl.constexpr,
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
        H: Head dimension.
        N_CTX: Context length.
        BLOCK_M2: Block size for M dimension.
        BLOCK_N2: Block size for N dimension.
        HEAD_DIM: Head dimension size.
        start_m: Starting index for M dimension.
        start_n: Starting index for N dimension.
        num_steps: Number of steps to unroll.
        MASK: Masking tensor.
        mask_ptr: Pointer to the mask tensor.
        MASK_TYPE: Type of masking (0: no mask, 1: boolean mask,
                        2: additive mask).
        dropout_p: Dropout probability.
        philox_seed: Seed for Philox RNG.
        philox_offset_base: Base offset for Philox RNG.
        ENABLE_DROPOUT: Flag to enable dropout.

    Returns:
        dq: Gradient of the query tensor.
    """
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # NOTE: D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    dtype = tl.float32  # For dot products
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        # Causal masking before exponentiation.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, -1e6)
        # External masking before exponentiation.
        if MASK_TYPE > 0:
            mask_offs = offs_m[:, None] * N_CTX + offs_n[None, :]
            if MASK_TYPE == 1:
                mask = tl.load(mask_ptr + mask_offs)
                qk = tl.where(mask, qk, -1e6)
            elif MASK_TYPE == 2:
                add_mask = tl.load(mask_ptr + mask_offs)
                qk += add_mask
        # Exponentiation.
        p = tl.math.exp2(qk - m)
        # Dropout after exponentiation.
        if ENABLE_DROPOUT:
            philox_offset = philox_offset_base + start_m * N_CTX + curr_n
            dropout_mask, dropout_scale = create_dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M2, BLOCK_N2, N_CTX
            )
            p = p * dropout_mask.to(p.dtype) * dropout_scale
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(dtype)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT).to(dtype)).to(kT.dtype)
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok
    return dq


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
    # shared by Q/K/V/DO.
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  #
    H,
    N_CTX,  #
    BLOCK_M1: tl.constexpr,  #
    BLOCK_N1: tl.constexpr,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    BLK_SLICE_FACTOR: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    # Args for masking/dropout
    mask_ptr,
    MASK_TYPE: tl.constexpr,
    dropout_p,
    philox_seed,
    philox_offset_base,
    ENABLE_DROPOUT: tl.constexpr,
):
    """The main backward pass for the attention mechanism.

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
        philox_offset_base: Base offset for Philox RNG.
        ENABLE_DROPOUT: Flag to enable dropout.
    """
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # Offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz

    # Load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = start_n

    MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # Load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = BLOCK_N1 // MASK_BLOCK_M1

    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        MASK_BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=True,  #
        mask_ptr=mask_ptr,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    start_m += num_steps * MASK_BLOCK_M1
    num_steps = (N_CTX - start_m) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk,
        dv,  #
        Q,
        k,
        v,
        sm_scale,  #
        DO,  #
        M,
        D,  #
        stride_tok,
        stride_d,  #
        H,
        N_CTX,  #
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,  #
        start_n,
        start_m,
        num_steps,  #
        MASK=False,  #
        mask_ptr=mask_ptr,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)

    # NOTE: THIS BLOCK DOES DQ:
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2

    MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    q = tl.load(Q + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(DO + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d)

    m = tl.load(M + offs_m)
    m = m[:, None]

    # Compute dQ for masked (diagonal) blocks.
    # NOTE: This code scans each row of QK^T backward (from right to left,
    # but inside each call to _attn_bwd_dq, from left to right), but that's
    # not due to anything important.  I just wanted to reuse the loop
    # structure for dK & dV above as much as possible.
    num_steps = BLOCK_M2 // MASK_BLOCK_N2
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
        H,
        N_CTX,  #
        BLOCK_M2,
        MASK_BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * MASK_BLOCK_N2,
        num_steps,  #
        MASK=True,  #
        mask_ptr=mask_ptr,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )
    end_n -= num_steps * MASK_BLOCK_N2
    # Stage 2
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
        H,
        N_CTX,  #
        BLOCK_M2,
        BLOCK_N2,
        HEAD_DIM,  #
        start_m,
        end_n - num_steps * BLOCK_N2,
        num_steps,  #
        MASK=False,  #
        mask_ptr=mask_ptr,
        MASK_TYPE=MASK_TYPE,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset_base,
        ENABLE_DROPOUT=ENABLE_DROPOUT,
    )
    # Write back dQ.
    dq_ptrs = DQ + offs_m[:, None] * stride_tok + offs_k[None, :] * stride_d
    dq *= LN2
    tl.store(dq_ptrs, dq)


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
        philox_offset: int
        ENABLE_DROPOUT: bool

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
        philox_offset: int
        ENABLE_DROPOUT: bool

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
    ) -> JVPAttn.FwdOut:
        """Forward pass for JVP Attention.

        Args:
            q: Query tensor of shape (Z, H, N_CTX, HEAD_DIM_Q).
            k: Key tensor of shape (Z, H, N_CTX, HEAD_DIM_K).
            v: Value tensor of shape (Z, H, N_CTX, HEAD_DIM_V).
            q_t: Optional tensor for query transpose.
            k_t: Optional tensor for key transpose.
            v_t: Optional tensor for value transpose.
            attn_mask: Optional attention mask of shape (Z, H, N_CTX, N_CTX). Two types of masks are supported. A boolean mask where a value of True indicates that the element should take part in attention, or a float mask of the same type as query, key, value that is added to the attention score.
            dropout_p: Dropout probability.
            causal: Whether the attention is causal.
            sm_scale: Optional scaling factor for softmax.
            warp_specialize: Whether to use warp specialization.
            USE_TMA: Whether to use TMA.

        Returns:
            Outputs of JVP Attention.
        """
        if attn_mask is not None:
            raise NotImplementedError("JVP attention does not support attention masks yet.")
        if dropout_p != 0.0:
            raise NotImplementedError("JVP attention does not support attention dropout yet.")

        # Shape constraints
        Z, H, N_CTX, HEAD_DIM_Q = q.shape
        HEAD_DIM_K = k.shape[-1]
        # NOTE: When v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        if causal and attn_mask is not None:
            raise ValueError("Causal attention does not support an attention mask.")
        if attn_mask is not None:
            assert attn_mask.shape == (
                Z,
                H,
                N_CTX,
                N_CTX,
            ), "The provided attention mask must have 4 dimensions (Z, H, N_CTX, N_CTX)."
            assert attn_mask.dtype in {
                torch.bool,
                q.dtype,
            }, "The attention mask must be of the dtype bool or that of the query tensor."

        if sm_scale is None:
            sm_scale = HEAD_DIM_K**-0.5
        o = torch.empty_like(q)
        ENABLE_JVP = q_t is not None
        o_t: Tensor | None = torch.empty_like(q_t) if ENABLE_JVP else None
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        if is_cuda() and warp_specialize:
            # NOTE: We need more registers if we're doing JVP
            if (HEAD_DIM_K == 128 and q.dtype == torch.float16) or ENABLE_JVP:
                extra_kern_args["maxnreg"] = 168
            else:
                # TODO: I think for backwards pass of dim=128 this is too low for H100; register allocation fails
                extra_kern_args["maxnreg"] = 80

        M = torch.empty((Z, H, N_CTX), device=q.device, dtype=torch.float32)

        if hasattr(triton, "set_allocator") and is_cuda():

            def alloc_fn(size: int, align: int, _):
                """Custom allocator function for Triton."""
                return torch.empty(size, dtype=torch.int8, device="cuda")

            triton.set_allocator(alloc_fn)

        def strides_zhnd(t: Tensor) -> JVPAttn.Strides:
            """Get strides for a tensor with shape (Z, H, N_CTX, HEAD_DIM)."""
            return JVPAttn.Strides(t.stride(0), t.stride(1), t.stride(2), t.stride(3))

        # Determine mask type
        if attn_mask is None:
            MASK_TYPE = 0
            mask_tensor = torch.empty(0, device=q.device, dtype=q.dtype)
            mask_strides = (0, 0, 0, 0)
        elif attn_mask.dtype == torch.bool:
            MASK_TYPE = 1
            mask_tensor = attn_mask
            mask_strides = strides_zhnd(attn_mask)
        else:
            MASK_TYPE = 2
            mask_tensor = attn_mask.to(q.dtype)
            mask_strides = strides_zhnd(attn_mask)

        # Setup dropout
        ENABLE_DROPOUT = dropout_p > 0.0
        if ENABLE_DROPOUT:
            philox_seed = torch.randint(0, 2**32, (1,), device=q.device, dtype=torch.int64).item()
        else:
            philox_seed = 0

        # Grid setup
        Z_H = Z * H

        def grid(META: dict[str, Any]) -> JVPAttn.Grid:
            """Determine grid configuration."""
            return JVPAttn.Grid(triton.cdiv(N_CTX, META["BLOCK_M"]), Z_H, 1)

        # if USE_TMA and supports_tma() and not (torch.cuda.get_device_capability()[0] == 9
        #                                        and q.dtype == torch.float8_e5m2):
        if USE_TMA and supports_tma():
            if attn_mask or ENABLE_DROPOUT:
                raise NotImplementedError(
                    "TMA kernel currently does not support attention masking or dropout."
                )

            # NOTE: On Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim = Z_H * N_CTX

            dummy_block = [1, 1]
            desc_q = TensorDescriptor(
                q,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_q_t = (
                desc_q
                if q_t is None
                else TensorDescriptor(
                    q_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=dummy_block,
                )
            )
            if q.dtype == torch.float8_e5m2:
                v_shape = [HEAD_DIM_K, y_dim]
                v_strides = [N_CTX, 1]
            else:
                v_shape = [y_dim, HEAD_DIM_K]
                v_strides = [HEAD_DIM_K, 1]
            desc_v = TensorDescriptor(v, shape=v_shape, strides=v_strides, block_shape=dummy_block)
            # NOTE: Probably we could share the shape and strides from above, but whatever
            if q_t.dtype == torch.float8_e5m2:
                t_v_shape = [HEAD_DIM_K, y_dim]
                t_v_strides = [q_t.shape[2], 1]
            else:
                t_v_shape = [y_dim, HEAD_DIM_K]
                t_v_strides = [HEAD_DIM_K, 1]
            desc_v_t = (
                desc_v
                if v_t is None
                else TensorDescriptor(
                    v_t, shape=t_v_shape, strides=t_v_strides, block_shape=dummy_block
                )
            )
            desc_k = TensorDescriptor(
                k,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_k_t = (
                desc_k
                if k_t is None
                else TensorDescriptor(
                    k_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=dummy_block,
                )
            )
            desc_o = TensorDescriptor(
                o,
                shape=[y_dim, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_o_t = (
                desc_o
                if o_t is None
                else TensorDescriptor(
                    o_t,
                    shape=[y_dim, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=dummy_block,
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
                N_CTX=N_CTX,  #
                HEAD_DIM=HEAD_DIM_K,  #
                FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                STAGE=stage,  #
                warp_specialize=warp_specialize,  #
                ENABLE_JVP=ENABLE_JVP,  #
                # # NOTE: The following are safe (unit-tested) default values
                # BLOCK_M=MIN_SEQUENCE_LENGTH,  #
                # BLOCK_N=MIN_SEQUENCE_LENGTH,  #
                # num_stages=NUM_STAGES_OPTIONS[0],  #
                # num_warps=4,  #
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
                STAGE=stage,  #
                warp_specialize=warp_specialize,  #
                ENABLE_JVP=ENABLE_JVP,  #
                ENABLE_DROPOUT=ENABLE_DROPOUT,
                MASK_TYPE=MASK_TYPE,
                # # NOTE: The following are safe (unit-tested) default values
                # BLOCK_M=MIN_SEQUENCE_LENGTH,  #
                # BLOCK_N=MIN_SEQUENCE_LENGTH,  #
                # num_stages=NUM_STAGES_OPTIONS[0],  #
                # num_warps=4,  #
                **extra_kern_args,
            )

        # TODO: Decide whether `0` is the right philox_offset here
        philox_offset = 0
        return JVPAttn.FwdOut(
            o,
            JVPAttn.FwdOutCtxContrib(
                o_t,
                M,
                grid,
                HEAD_DIM_K,
                sm_scale,
                mask_tensor,
                MASK_TYPE,
                dropout_p,
                philox_seed,
                philox_offset,
                ENABLE_DROPOUT,
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
            philox_offset,
            ENABLE_DROPOUT,
        ) = outputs
        ctx.grid = grid
        ctx.save_for_forward(o_t)
        ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM_K = HEAD_DIM_K
        ctx.causal = causal
        ctx.mask_tensor = mask_tensor
        ctx.MASK_TYPE = MASK_TYPE
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.ENABLE_DROPOUT = ENABLE_DROPOUT

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
        )
        a, _ = out
        return a

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
    ) -> Tensor:
        """Forward pass for JVP Attention with dual tensor inputs.

        This is not an autograd convention. It's a workaround for invoking
        JVPAttn::forward with the right arguments when you have a dual tensor input.

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

        Returns:
            The output tensor.
        """
        q_p, q_t = fwAD.unpack_dual(q)
        k_p, k_t = fwAD.unpack_dual(k)
        v_p, v_t = fwAD.unpack_dual(v)
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
        )
        a, _ = out
        return a

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
        return JVPAttn.JVPOut(ctx.saved_for_forward[0], None)

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

        # Ensure inputs/outputs the kernel reads share the same (contiguous) layout
        if (
            not q.is_contiguous()
            or not k.is_contiguous()
            or not v.is_contiguous()
            or not o.is_contiguous()
        ):
            raise RuntimeError(
                "JVPAttn expects q, k, v, o to be contiguous; call .contiguous() in forward."
            )

        # Autograd may deliver a non-contiguous grad_output; normalize it.
        if (not do.is_contiguous()) or (do.stride() != o.stride()):
            do = do.contiguous()

        assert do.is_contiguous()

        assert q.stride() == k.stride() == v.stride() == o.stride()
        assert do.stride() == o.stride()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        BATCH, N_HEAD, N_CTX = q.shape[:3]
        PRE_BLOCK = MIN_SEQUENCE_LENGTH  # NOTE: Adjust according to minimum input sequence length
        NUM_WARPS, NUM_STAGES = 4, NUM_STAGES_OPTIONS[0]
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, PRE_BLOCK, PRE_BLOCK, 32
        BLK_SLICE_FACTOR = 2
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o,
            do,  #
            delta,  #
            BATCH,
            N_HEAD,
            N_CTX,  #
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM_K,  #
        )
        grid = (N_CTX // BLOCK_N1, 1, BATCH * N_HEAD)
        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dq,
            dk,
            dv,  #
            M,
            delta,  #
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            N_HEAD,
            N_CTX,  #
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,  #
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,  #
            HEAD_DIM=ctx.HEAD_DIM_K,  #
            mask_ptr=ctx.mask_tensor,
            MASK_TYPE=ctx.MASK_TYPE,
            dropout_p=ctx.dropout_p,
            philox_seed=ctx.philox_seed,
            philox_offset_base=ctx.philox_offset,  # Same as forward
            ENABLE_DROPOUT=ctx.ENABLE_DROPOUT,
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES,  #
        )

        return JVPAttn.BwdOut(dq, dk, dv, None, None, None, None, None, None, None, None, None)


attention = JVPAttn.fwd
