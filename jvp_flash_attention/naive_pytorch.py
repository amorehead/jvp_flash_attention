"""Tiled Flash Attention with explicit JVP and backward in pure PyTorch.

Computes online base-2 softmax, forward-mode JVP tangents, and recomputed backward
gradients over (q, k) tiles so no full (N_Q, N_KV) matrix is ever materialized.
"""

from __future__ import annotations

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor, arange, cat, einsum
from torch.autograd import Function

RCP_LN2 = 1.4426950408889634  # 1 / ln(2)
MASK_CONST = -1.0e2  # large negative additive-mask constant


def exists(val) -> bool:
    return val is not None


def default_dtype(t: Tensor) -> torch.dtype:
    return t.dtype if t.dtype in (torch.float32, torch.float64) else torch.float32


def _is_compiling() -> bool:
    try:
        return torch.compiler.is_compiling()
    except Exception:
        return False


def validate_attn_mask(mask: Tensor, q: Tensor, k: Tensor | None = None) -> None:
    """Assert that `mask` matches `(q, k)` and no head is completely masked out."""
    z, h, n_q, _ = q.shape
    n_kv = k.shape[-2] if exists(k) else n_q

    assert mask.shape == (
        z,
        h,
        n_q,
        n_kv,
    ), "The attention mask must have shape (Z, H, N_CTX, N_CTX)."
    assert mask.dtype in (
        torch.bool,
        q.dtype,
    ), "The attention mask must be bool or of the query dtype."

    if mask.dtype == torch.bool:
        assert mask.any(dim=(-1, -2)).all(), "The attention mask cannot be all False for any head."
        return

    assert not torch.isinf(mask).any(), "The attention mask cannot contain -inf or inf."
    assert not torch.isnan(mask).any(), "The attention mask cannot contain NaNs."
    assert (
        (mask != MASK_CONST).any(dim=(-1, -2)).all()
    ), f"The attention mask cannot be all {MASK_CONST} for any head."
    if not (mask == MASK_CONST).any():
        raise UserWarning(
            f"The attention mask does not mask out any elements with {MASK_CONST}; "
            "use that constant for correct masking behavior."
        )


def prepare_attn_mask(
    attn_mask: Tensor | None,
    q: Tensor,
    verify_attn_mask: bool = True,
    k: Tensor | None = None,
) -> tuple[Tensor, int]:
    """Return `(contiguous_mask, MASK_TYPE)`, with `MASK_TYPE` in `{0: none, 1: bool, 2: additive}`."""
    if not exists(attn_mask):
        return torch.empty(0, device=q.device, dtype=q.dtype), 0
    if verify_attn_mask:
        validate_attn_mask(attn_mask, q, k)
    is_bool = attn_mask.dtype == torch.bool
    return (attn_mask if is_bool else attn_mask.to(q.dtype)).contiguous(), 1 if is_bool else 2


def causal_keep_mask(
    q_start: int,
    q_len: int,
    k_start: int,
    k_len: int,
    kv_offset: int,
    device: torch.device,
) -> Tensor | None:
    """Return the causal keep-mask on diagonal tiles, or `None` when the full tile is valid."""
    q_min = kv_offset + q_start
    if k_start + k_len - 1 <= q_min:
        return None
    q_idx = arange(q_min, q_min + q_len, device=device)
    k_idx = arange(k_start, k_start + k_len, device=device)
    return q_idx[:, None] >= k_idx[None, :]


def score_tile(
    q_chunk: Tensor,
    k_chunk: Tensor,
    mask: Tensor,
    MASK_TYPE: int,
    qk_scale: float,
    causal_keep: Tensor | None,
    q_slice: slice,
    k_slice: slice,
) -> tuple[Tensor, Tensor | None]:
    """Compute scaled base-2 attention logits and optional boolean keep-mask for one tile."""
    scores = einsum("z h i d, z h j d -> z h i j", q_chunk, k_chunk)

    if MASK_TYPE == 2:
        return (scores + mask[..., q_slice, k_slice]) * qk_scale, None

    scores = scores * qk_scale
    keep = mask[..., q_slice, k_slice] if MASK_TYPE == 1 else causal_keep
    if exists(keep):
        scores = scores.masked_fill(~keep, float("-inf"))

    return scores, keep


def tiled_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask_tensor: Tensor,
    MASK_TYPE: int,
    causal: bool,
    sm_scale: float,
    q_t: Tensor | None = None,
    k_t: Tensor | None = None,
    v_t: Tensor | None = None,
    q_bucket_size: int = 128,
    k_bucket_size: int = 128,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Tiled forward + optional JVP -> `(o, o_t, lse)`; `o_t` is `None` when no tangents are given."""
    z, h, n_q, _ = q.shape
    n_kv = k.shape[-2]
    kv_offset = n_kv - n_q  # queries are aligned to the tail of kv
    device, out_dtype = q.device, q.dtype
    compute_dtype = default_dtype(q)

    q, k, v = (t.to(compute_dtype) for t in (q, k, v))
    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks, v_chunks = k.split(k_bucket_size, dim=-2), v.split(k_bucket_size, dim=-2)

    has_qt, has_kt, has_vt = exists(q_t), exists(k_t), exists(v_t)
    enable_jvp = has_qt or has_kt or has_vt
    has_qk_tangent = has_qt or has_kt

    if has_qt:
        q_t = q_t.to(compute_dtype)
    kt_chunks = k_t.to(compute_dtype).split(k_bucket_size, dim=-2) if has_kt else None
    vt_chunks = v_t.to(compute_dtype).split(k_bucket_size, dim=-2) if has_vt else None

    qk_scale = sm_scale * RCP_LN2
    o_chunks, ot_chunks, lse_chunks = [], [], []

    for i, q_chunk in enumerate(q_chunks):
        q_len = q_chunk.shape[-2]
        q_start = i * q_bucket_size
        q_end = q_start + q_len
        q_slice = slice(q_start, q_end)
        qt_chunk = q_t[..., q_slice, :] if has_qt else None

        m = q_chunk.new_full((z, h, q_len, 1), float("-inf"))
        l_i = torch.zeros_like(m)
        acc = torch.zeros_like(q_chunk)
        if enable_jvp:
            g_acc = torch.zeros_like(acc)
            mu = torch.zeros_like(m)
            p_tv_acc = torch.zeros_like(acc)

        for j, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
            k_len = k_chunk.shape[-2]
            k_start = j * k_bucket_size
            if causal and k_start > kv_offset + q_end - 1:
                break  # all remaining key blocks lie strictly in the future

            k_slice = slice(k_start, k_start + k_len)
            causal_keep = (
                causal_keep_mask(q_start, q_len, k_start, k_len, kv_offset, device)
                if causal
                else None
            )
            scores, keep = score_tile(
                q_chunk, k_chunk, mask_tensor, MASK_TYPE, qk_scale, causal_keep, q_slice, k_slice
            )

            # Softmax and lse are shift-invariant to the running max, so detach it from AD.
            # Guard all-(-inf) tiles with 0.0 so exp2(-inf - m_ij_safe) evaluates to 0, not NaN.
            m_ij = fwAD.unpack_dual(torch.maximum(m, scores.amax(dim=-1, keepdim=True)))[0].detach()
            m_ij_safe = torch.where(m_ij == float("-inf"), 0.0, m_ij)

            p = torch.exp2(scores - m_ij_safe)
            alpha = torch.exp2(m - m_ij_safe)

            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + einsum("z h i j, z h j d -> z h i d", p, v_chunk)

            if enable_jvp:
                if has_qk_tangent:
                    t_scores = (
                        einsum("z h i d, z h j d -> z h i j", qt_chunk, k_chunk)
                        if has_qt
                        else 0.0
                    )
                    if has_kt:
                        t_scores = t_scores + einsum(
                            "z h i d, z h j d -> z h i j", q_chunk, kt_chunks[j]
                        )
                    if exists(keep):
                        t_scores = t_scores.masked_fill(~keep, 0.0)

                    t_p = p * (t_scores * sm_scale)
                    g_acc = g_acc * alpha + einsum("z h i j, z h j d -> z h i d", t_p, v_chunk)
                    mu = mu * alpha + t_p.sum(dim=-1, keepdim=True)

                if has_vt:
                    p_tv_acc = p_tv_acc * alpha + einsum(
                        "z h i j, z h j d -> z h i d", p, vt_chunks[j]
                    )

            m = m_ij

        empty = l_i == 0.0
        l_i_safe = torch.where(empty, 1.0, l_i)
        lse = m + torch.where(empty, 0.0, torch.log2(l_i_safe))
        o_chunk = acc / l_i_safe

        o_chunks.append(o_chunk.to(out_dtype))
        lse_chunks.append(lse.squeeze(-1))
        if enable_jvp:
            ot_chunk = g_acc / l_i_safe - (mu / l_i_safe) * o_chunk + p_tv_acc / l_i_safe
            ot_chunks.append(ot_chunk.to(out_dtype))

    o = cat(o_chunks, dim=-2)
    o_t = cat(ot_chunks, dim=-2) if enable_jvp else None
    return o, o_t, cat(lse_chunks, dim=-1)


def tiled_attention_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    do: Tensor,
    lse: Tensor,
    mask_tensor: Tensor,
    MASK_TYPE: int,
    causal: bool,
    sm_scale: float,
    q_bucket_size: int = 128,
    k_bucket_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """Tiled backward -> `(dq, dk, dv)`; recomputes `p` per tile from `lse`."""
    n_q = q.shape[-2]
    n_kv = k.shape[-2]
    kv_offset = n_kv - n_q
    device, out_dtype = q.device, q.dtype
    compute_dtype = default_dtype(q)

    q, k, v, o, do = (t.to(compute_dtype) for t in (q, k, v, o, do))
    lse = lse.to(compute_dtype).unsqueeze(-1)
    lse_safe = torch.where(lse == float("-inf"), 0.0, lse)

    qk_scale = sm_scale * RCP_LN2
    delta = (o * do).sum(dim=-1, keepdim=True)

    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks, v_chunks = k.split(k_bucket_size, dim=-2), v.split(k_bucket_size, dim=-2)
    dq_chunks = []
    dk_chunks = [torch.zeros_like(kc) for kc in k_chunks]
    dv_chunks = [torch.zeros_like(vc) for vc in v_chunks]

    for i, q_chunk in enumerate(q_chunks):
        q_len = q_chunk.shape[-2]
        q_start = i * q_bucket_size
        q_end = q_start + q_len
        q_slice = slice(q_start, q_end)

        do_chunk = do[..., q_slice, :]
        delta_chunk = delta[..., q_slice, :]
        lse_chunk = lse_safe[..., q_slice, :]
        dq_chunk = torch.zeros_like(q_chunk)

        for j, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
            k_len = k_chunk.shape[-2]
            k_start = j * k_bucket_size
            if causal and k_start > kv_offset + q_end - 1:
                break

            k_slice = slice(k_start, k_start + k_len)
            causal_keep = (
                causal_keep_mask(q_start, q_len, k_start, k_len, kv_offset, device)
                if causal
                else None
            )
            scores, _ = score_tile(
                q_chunk, k_chunk, mask_tensor, MASK_TYPE, qk_scale, causal_keep, q_slice, k_slice
            )

            p = torch.exp2(scores - lse_chunk)
            dp = einsum("z h i d, z h j d -> z h i j", do_chunk, v_chunk)
            ds = p * (dp - delta_chunk) * sm_scale

            dq_chunk = dq_chunk + einsum("z h i j, z h j d -> z h i d", ds, k_chunk)
            dk_chunks[j] = dk_chunks[j] + einsum("z h i j, z h i d -> z h j d", ds, q_chunk)
            dv_chunks[j] = dv_chunks[j] + einsum("z h i j, z h i d -> z h j d", p, do_chunk)

        dq_chunks.append(dq_chunk)

    return (
        cat(dq_chunks, dim=-2).to(out_dtype),
        cat(dk_chunks, dim=-2).to(out_dtype),
        cat(dv_chunks, dim=-2).to(out_dtype),
    )


class _NaiveJVPAttn(Function):
    """Autograd + forward-AD wrapper around `tiled_attention` with O(N) saved state."""

    @staticmethod
    def forward(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        q_t: Tensor | None,
        k_t: Tensor | None,
        v_t: Tensor | None,
        mask_tensor: Tensor,
        MASK_TYPE: int,
        causal: bool,
        sm_scale: float,
        q_bucket_size: int,
        k_bucket_size: int,
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor]]:
        o, o_t, lse = tiled_attention(
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
            q_bucket_size=q_bucket_size,
            k_bucket_size=k_bucket_size,
        )
        return o, (o_t, lse)

    @staticmethod
    def setup_context(ctx, inputs, outputs) -> None:
        q, k, v, _, _, _, mask_tensor, MASK_TYPE, causal, sm_scale, q_bucket_size, k_bucket_size = (
            inputs
        )
        o, (o_t, lse) = outputs
        ctx.save_for_forward(o_t)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.mask_tensor = mask_tensor
        ctx.MASK_TYPE = MASK_TYPE
        ctx.causal = causal
        ctx.sm_scale = sm_scale
        ctx.q_bucket_size = q_bucket_size
        ctx.k_bucket_size = k_bucket_size

    @staticmethod
    def jvp(ctx, gq, gk, gv, *_) -> tuple[Tensor | None, None]:
        return ctx.saved_for_forward[0], None

    @staticmethod
    def backward(ctx, do, _):
        q, k, v, o, lse = ctx.saved_tensors
        if torch.is_grad_enabled():
            o, _, lse = tiled_attention(
                q,
                k,
                v,
                ctx.mask_tensor,
                ctx.MASK_TYPE,
                ctx.causal,
                ctx.sm_scale,
                q_bucket_size=ctx.q_bucket_size,
                k_bucket_size=ctx.k_bucket_size,
            )

        dq, dk, dv = tiled_attention_backward(
            q,
            k,
            v,
            o,
            do,
            lse,
            ctx.mask_tensor,
            ctx.MASK_TYPE,
            ctx.causal,
            ctx.sm_scale,
            q_bucket_size=ctx.q_bucket_size,
            k_bucket_size=ctx.k_bucket_size,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def naive_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    sm_scale: float | None = None,
    verify_attn_mask: bool = True,
    q_bucket_size: int = 128,
    k_bucket_size: int = 128,
) -> Tensor:
    """Drop-in pure-PyTorch tiled JVP attention supporting forward-AD, backward, and HVPs."""
    if dropout_p != 0.0:
        raise NotImplementedError("Dropout is not currently supported in JVP attention.")

    _, _, n_q, dim = q.shape
    n_kv, dim_k = k.shape[-2], k.shape[-1]
    dim_v = v.shape[-1]
    assert dim == dim_k == dim_v, (
        "JVP attention requires equal query, key, and value head dimensions"
        f" but got {dim}, {dim_k}, and {dim_v}"
    )
    assert dim in {
        16,
        32,
        64,
        128,
        256,
    }, f"JVP attention only supports head dims in {{16, 32, 64, 128, 256}}, but got {dim}"
    assert (
        n_q >= 1 and n_kv >= n_q
    ), f"JVP attention requires sequence length >= 1 and key length >= query length, but got {n_q} and {n_kv}"
    if causal and exists(attn_mask):
        raise ValueError("Causal attention does not support an attention mask.")

    q_p, q_t = fwAD.unpack_dual(q)
    k_p, k_t = fwAD.unpack_dual(k)
    v_p, v_t = fwAD.unpack_dual(v)

    mask_tensor, MASK_TYPE = prepare_attn_mask(attn_mask, q_p, verify_attn_mask, k_p)
    sm_scale = dim**-0.5 if sm_scale is None else sm_scale

    if _is_compiling() and (exists(q_t) or exists(k_t) or exists(v_t)):
        o, o_t, _ = tiled_attention(
            q_p,
            k_p,
            v_p,
            mask_tensor,
            MASK_TYPE,
            causal,
            sm_scale,
            q_t,
            k_t,
            v_t,
            q_bucket_size=q_bucket_size,
            k_bucket_size=k_bucket_size,
        )
        return fwAD.make_dual(o, o_t)

    out, _ = _NaiveJVPAttn.apply(
        q,
        k,
        v,
        q_t,
        k_t,
        v_t,
        mask_tensor,
        MASK_TYPE,
        causal,
        sm_scale,
        q_bucket_size,
        k_bucket_size,
    )
    return out
