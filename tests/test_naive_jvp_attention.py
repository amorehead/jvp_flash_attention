"""Tests for the pure-PyTorch tiled attention fallback (`naive_pytorch`)."""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor, enable_grad
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from jvp_flash_attention.jvp_attention import (
    HAS_TRITON,
    MASK_CONST,
    JVPAttn,
    use_naive_attention,
)
from jvp_flash_attention.naive_pytorch import (
    naive_attention,
    prepare_attn_mask,
    tiled_attention,
    tiled_attention_backward,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ATOL = {torch.float32: 1e-4, torch.float16: 4e-3, torch.bfloat16: 4e-3}
RTOL = 1e-3
Z, H, N, D = 2, 3, 64, 32
MASK_CASES = [
    pytest.param(None, False, id="none"),
    pytest.param("boolean", False, id="boolean"),
    pytest.param("additive", False, id="additive"),
    pytest.param(None, True, id="causal"),
]


def rand(*shape: int, dtype: torch.dtype = torch.float32, seed: int = 0) -> Tensor:
    """Create a reproducible random tensor on the test device."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=generator).to(DEVICE)


def make_qkv_tangents_target(dtype: torch.dtype, n_ctx: int = N) -> tuple[Tensor, ...]:
    """Create reproducible (q, k, v, q_t, k_t, v_t, target) tensors."""

    return tuple(rand(Z, H, n_ctx, D, dtype=dtype, seed=i) for i in range(7))


def make_duals(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Make dual tensors that retain input gradients, as `torch.autograd` expects."""

    duals = []
    for primal, tangent in zip(tensors[::2], tensors[1::2]):
        dual = fwAD.make_dual(primal, tangent)
        dual.requires_grad = True
        dual.retain_grad()
        duals.append(dual)
    return tuple(duals)


def make_attn_mask(mask_kind: str | None, dtype: torch.dtype, n_ctx: int = N) -> Tensor | None:
    """Create a mask that passes JVP attention's mask validation."""

    if mask_kind is None:
        return None

    generator = torch.Generator(device="cpu").manual_seed(42)
    if mask_kind == "boolean":
        mask = torch.rand(Z, H, n_ctx, n_ctx, generator=generator) > 0.3
        mask[..., 0, 0] = True  # Ensure no head is fully masked
        return mask.to(DEVICE)

    mask = torch.where(
        torch.rand(Z, H, n_ctx, n_ctx, generator=generator) > 0.3,
        torch.zeros(()),
        torch.full((), MASK_CONST),
    )
    mask[..., 0, 0] = 0.0  # Ensure no head is fully masked
    return mask.to(dtype).to(DEVICE)


def sdpa_dual_reference(
    q_p: Tensor,
    k_p: Tensor,
    v_p: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    target: Tensor,
    mask: Tensor | None,
    causal: bool,
) -> tuple[Tensor, ...]:
    """Compute SDPA's primal, tangent, and input gradients in float32 MATH mode."""

    with sdpa_kernel(SDPBackend.MATH), fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()

    return o_p.detach(), o_t.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def assert_close_metrics(
    got: tuple[Tensor, ...],
    expected: tuple[Tensor, ...],
    dtype: torch.dtype,
    atol: float | None = None,
) -> None:
    """Compare primal/tangent/gradient tensors with dtype-appropriate tolerances."""

    tol = ATOL[dtype] if atol is None else atol
    for actual, desired in zip(got, expected):
        torch.testing.assert_close(actual, desired, atol=tol, rtol=RTOL)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_fwd_matches_sdpa(dtype: torch.dtype, mask_kind: str | None, causal: bool) -> None:
    """The naive forward primal and gradients match SDPA's."""

    q_p, k_p, v_p, *_, target = make_qkv_tangents_target(dtype)
    mask = make_attn_mask(mask_kind, dtype)

    with sdpa_kernel(SDPBackend.MATH), enable_grad():
        q_ref, k_ref, v_ref = (t.clone().requires_grad_() for t in (q_p, k_p, v_p))
        ref = scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=mask, is_causal=causal)
        ((ref - target) ** 2).mean().backward()

    q, k, v = (t.clone().requires_grad_() for t in (q_p, k_p, v_p))
    out = JVPAttn.fwd(q, k, v, attn_mask=mask, causal=causal, USE_NAIVE=True)
    ((out - target) ** 2).mean().backward()

    assert_close_metrics(
        (out, q.grad, k.grad, v.grad),
        (ref, q_ref.grad, k_ref.grad, v_ref.grad),
        dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_jvp_interfaces_match_sdpa(
    dtype: torch.dtype, mask_kind: str | None, causal: bool
) -> None:
    """`fwd_dual` and `torch.func.jvp` both match SDPA's primal, tangent, and gradients."""

    q_p, k_p, v_p, q_t, k_t, v_t, target = make_qkv_tangents_target(dtype)
    mask = make_attn_mask(mask_kind, dtype)
    expected = sdpa_dual_reference(q_p, k_p, v_p, q_t, k_t, v_t, target, mask, causal)

    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = JVPAttn.fwd_dual(q, k, v, attn_mask=mask, causal=causal, USE_NAIVE=True)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()
    assert_close_metrics((o_p, o_t, q.grad, k.grad, v.grad), expected, dtype)

    primals = tuple(t.clone().requires_grad_() for t in (q_p, k_p, v_p))
    with enable_grad():
        o_p, o_t = torch.func.jvp(
            partial(JVPAttn.fwd_dual, attn_mask=mask, causal=causal, USE_NAIVE=True),
            primals,
            (q_t, k_t, v_t),
        )
        ((o_p - target) ** 2).mean().backward()
    assert_close_metrics((o_p, o_t, *[t.grad for t in primals]), expected, dtype)


def test_fwd_dual_with_plain_inputs() -> None:
    """`fwd_dual` accepts plain tensors and returns a plain tensor."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    out = JVPAttn.fwd_dual(q_p, k_p, v_p, USE_NAIVE=True)
    with fwAD.dual_level():
        assert fwAD.unpack_dual(out)[1] is None
    torch.testing.assert_close(
        out, JVPAttn.fwd(q_p, k_p, v_p, USE_NAIVE=True), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("use_naive", [True, False])
@pytest.mark.parametrize("active_tangent", ["q", "k", "v"])
def test_partial_dual_inputs(use_naive: bool, active_tangent: str) -> None:
    """`fwd_dual` supports dual tensors where only a subset of (q, k, v) has tangents."""
    if not use_naive and not (HAS_TRITON and torch.cuda.is_available()):
        pytest.skip("requires Triton on a CUDA device")

    q_p, k_p, v_p, q_t, k_t, v_t, _ = make_qkv_tangents_target(torch.float32)
    with fwAD.dual_level():
        q = fwAD.make_dual(q_p, q_t) if active_tangent == "q" else q_p
        k = fwAD.make_dual(k_p, k_t) if active_tangent == "k" else k_p
        v = fwAD.make_dual(v_p, v_t) if active_tangent == "v" else v_p

        out = JVPAttn.fwd_dual(q, k, v, USE_NAIVE=use_naive)
        o_p, o_t = fwAD.unpack_dual(out)
        assert o_t is not None
        assert not torch.isnan(o_p).any()
        assert not torch.isnan(o_t).any()


def test_apply_interface() -> None:
    """`JVPAttn.apply` returns the kernel's (output, context) structure."""

    q_p, k_p, v_p, q_t, k_t, v_t, _ = make_qkv_tangents_target(torch.float32)
    expected = JVPAttn.fwd(q_p, k_p, v_p)
    args = (None, 0.0, False, None, True, True, True, True)

    out, ctx = JVPAttn.apply(q_p, k_p, v_p, None, None, None, *args)
    assert ctx[0] is None
    assert ctx[1].shape == (Z, H, N)  # Base-2 log-sum-exp
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    out, ctx = JVPAttn.apply(q_p, k_p, v_p, q_t, k_t, v_t, *args)
    _, expected_tangent = torch.func.jvp(JVPAttn.fwd_dual, (q_p, k_p, v_p), (q_t, k_t, v_t))
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ctx[0], expected_tangent, atol=1e-6, rtol=1e-6)


def test_lse_matches_logsumexp() -> None:
    """The returned base-2 log-sum-exp matches `logsumexp` of the natural scores."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    sm_scale = D**-0.5
    _, _, lse = tiled_attention(q_p, k_p, v_p, torch.empty(0), 0, False, sm_scale)
    natural_scores = (q_p @ k_p.transpose(-1, -2)) * sm_scale
    expected = torch.logsumexp(natural_scores, dim=-1) / math.log(2)
    torch.testing.assert_close(lse, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n_ctx", [32, 96, 130, 258])
def test_bucket_size_invariance(n_ctx: int) -> None:
    """Bucket sizes only affect memory usage, not the result."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32, n_ctx)
    empty = torch.empty(0, device=DEVICE)
    small = tiled_attention(
        q_p, k_p, v_p, empty, 0, False, D**-0.5, q_bucket_size=32, k_bucket_size=32
    )
    large = tiled_attention(
        q_p, k_p, v_p, empty, 0, False, D**-0.5, q_bucket_size=512, k_bucket_size=512
    )
    torch.testing.assert_close(small[0], large[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(small[2], large[2], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "n_ctx,q_bucket_size,k_bucket_size",
    [
        pytest.param(32, 128, 128, id="shorter-than-one-tile"),
        pytest.param(96, 64, 64, id="partial-trailing-tiles"),
        pytest.param(130, 64, 32, id="uneven-partial-tiles"),
        pytest.param(258, 128, 128, id="partial-tile-after-full-tiles"),
    ],
)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_arbitrary_seq_lens(
    n_ctx: int, q_bucket_size: int, k_bucket_size: int, mask_kind: str | None, causal: bool
) -> None:
    """Any even length >= 32 works, whether or not the tile size divides it."""

    dtype = torch.float32
    q_p, k_p, v_p, q_t, k_t, v_t, target = make_qkv_tangents_target(dtype, n_ctx)
    mask = make_attn_mask(mask_kind, dtype, n_ctx)
    expected = sdpa_dual_reference(q_p, k_p, v_p, q_t, k_t, v_t, target, mask, causal)

    # default buckets, through the autograd.Function
    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = JVPAttn.fwd_dual(q, k, v, attn_mask=mask, causal=causal, USE_NAIVE=True)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()
    assert_close_metrics((o_p, o_t, q.grad, k.grad, v.grad), expected, dtype)

    # explicit (possibly non-dividing) buckets, through the pure-PyTorch pass
    mask_tensor, MASK_TYPE = prepare_attn_mask(mask, q_p)
    o, o_t, lse = tiled_attention(
        q_p,
        k_p,
        v_p,
        mask_tensor,
        MASK_TYPE,
        causal,
        D**-0.5,
        q_t,
        k_t,
        v_t,
        q_bucket_size=q_bucket_size,
        k_bucket_size=k_bucket_size,
    )
    torch.testing.assert_close(o, expected[0], atol=ATOL[dtype], rtol=RTOL)
    torch.testing.assert_close(o_t, expected[1], atol=ATOL[dtype], rtol=RTOL)

    do = 2 * (o - target) / o.numel()
    dq, dk, dv = tiled_attention_backward(
        q_p,
        k_p,
        v_p,
        o,
        do,
        lse,
        mask_tensor,
        MASK_TYPE,
        causal,
        D**-0.5,
        q_bucket_size=q_bucket_size,
        k_bucket_size=k_bucket_size,
    )
    assert_close_metrics((dq, dk, dv), expected[2:5], dtype)


def test_cpu_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU (and Triton-less) execution dispatches to the naive implementation, with overrides."""

    assert use_naive_attention(torch.device("cpu"))
    assert use_naive_attention(torch.device("cpu"), USE_NAIVE=True)
    if not (HAS_TRITON and torch.cuda.is_available()):
        assert use_naive_attention(torch.device("cuda"))
        assert use_naive_attention(torch.device("cuda"), USE_NAIVE=True)

    with pytest.raises(AssertionError, match="Cannot force the Triton backend"):
        use_naive_attention(torch.device("cpu"), USE_NAIVE=False)
    with pytest.raises(AssertionError, match="USE_NAIVE must be"):
        use_naive_attention(torch.device("cpu"), USE_NAIVE="yes")

    if DEVICE.type == "cpu":
        q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
        torch.testing.assert_close(
            JVPAttn.fwd(q_p, k_p, v_p, USE_NAIVE=True),
            naive_attention(q_p, k_p, v_p),
            atol=1e-6,
            rtol=1e-6,
        )
        with pytest.raises(AssertionError, match="Cannot force the Triton backend"):
            JVPAttn.fwd(q_p, k_p, v_p, USE_NAIVE=False)

        q_t, k_t, v_t = (rand(Z, H, N, D, seed=i + 3) for i in range(3))
        with fwAD.dual_level():
            out = JVPAttn.fwd_dual(q_p, k_p, v_p, USE_NAIVE=True)
            out_t = JVPAttn.fwd_dual(
                fwAD.make_dual(q_p, q_t),
                fwAD.make_dual(k_p, k_t),
                fwAD.make_dual(v_p, v_t),
                USE_NAIVE=True,
            )
            o_p, _ = fwAD.unpack_dual(out_t)
        torch.testing.assert_close(out, o_p, atol=1e-6, rtol=1e-6)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert use_naive_attention(torch.device("cuda"))


def test_unsupported_options() -> None:
    """Dropout is unsupported, and causal attention rejects a mask."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    with pytest.raises(NotImplementedError, match="Dropout is not currently supported"):
        JVPAttn.fwd(q_p, k_p, v_p, dropout_p=0.1)
    with pytest.raises(ValueError, match="Causal attention does not support an attention mask"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=torch.ones(Z, H, N, N, dtype=torch.bool), causal=True)


def test_invalid_masks_rejected() -> None:
    """Invalid masks are rejected, and additive masks without the constant warn."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)

    mask = make_attn_mask("boolean", torch.float32)
    mask[0, 0] = False
    with pytest.raises(AssertionError, match="cannot be all False"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=mask)

    for message, value in (
        ("cannot contain NaNs", float("nan")),
        ("cannot contain -inf or inf", float("inf")),
    ):
        mask = make_attn_mask("additive", torch.float32)
        mask[0, 0, 0, 0] = value
        with pytest.raises(AssertionError, match=message):
            JVPAttn.fwd(q_p, k_p, v_p, attn_mask=mask)

    mask = make_attn_mask("additive", torch.float32)
    mask[0, 0] = MASK_CONST
    with pytest.raises(AssertionError, match="cannot be all"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=mask)

    zeros = torch.zeros(Z, H, N, N, dtype=torch.float32, device=DEVICE)
    with pytest.raises(UserWarning, match="does not mask out any elements"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=zeros)

    # NOTE: The warning can be skipped entirely by disabling mask verification
    out = JVPAttn.fwd(q_p, k_p, v_p, attn_mask=zeros, verify_attn_mask=False)
    assert out.shape == (Z, H, N, D)


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="requires Triton on a CUDA device"
)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_naive_matches_triton_kernel(
    dtype: torch.dtype, mask_kind: str | None, causal: bool
) -> None:
    """The naive implementation matches the Triton kernel's outputs and gradients."""

    q_p, k_p, v_p, q_t, k_t, v_t, target = make_qkv_tangents_target(dtype)
    mask = make_attn_mask(mask_kind, dtype)

    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = JVPAttn.fwd_dual(q, k, v, attn_mask=mask, causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()
        triton = (o_p.detach(), o_t.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach())

    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = naive_attention(q, k, v, attn_mask=mask, causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()
        naive = (o_p.detach(), o_t.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach())

    # NOTE: The naive path computes in fp32 while the kernel uses bf16 tensor cores,
    # so the two implementations agree only to within a few bf16 ulps.
    assert_close_metrics(naive, triton, dtype, atol=5e-2 if dtype is torch.bfloat16 else None)


@pytest.mark.parametrize("use_naive", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_sequential_matches_parallel(dtype: torch.dtype, use_naive: bool) -> None:
    """One token at a time (with a growing context) matches parallel causal attention."""

    if not use_naive and not (HAS_TRITON and torch.cuda.is_available()):
        pytest.skip("requires Triton on a CUDA device")

    q_p, k_p, v_p, q_t, k_t, v_t, _ = make_qkv_tangents_target(dtype, 64)

    with fwAD.dual_level():
        parallel = JVPAttn.fwd_dual(
            *make_duals(
                q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
            ),
            causal=True,
            USE_NAIVE=use_naive,
        )
        parallel_p, parallel_t = fwAD.unpack_dual(parallel)

        seq_p, seq_t = [], []
        for i in range(64):
            step = JVPAttn.fwd_dual(
                *make_duals(
                    q_p[..., i : i + 1, :].clone(),
                    q_t[..., i : i + 1, :].clone(),
                    k_p[..., : i + 1, :].clone(),
                    k_t[..., : i + 1, :].clone(),
                    v_p[..., : i + 1, :].clone(),
                    v_t[..., : i + 1, :].clone(),
                ),
                causal=True,
                USE_NAIVE=use_naive,
            )
            step_p, step_t = fwAD.unpack_dual(step)
            seq_p.append(step_p)
            seq_t.append(step_t)

    seq_p = torch.cat(seq_p, dim=-2)
    seq_t = torch.cat(seq_t, dim=-2)
    # NOTE: Rectangular Q/KV routes sequential steps to the tiled fallback, so
    # use_naive=False compares it against the Triton parallel path; bf16 then
    # differs by a few ulps. The same applies to the accumulated JVP tangent.
    atol = 5e-2 if dtype is torch.bfloat16 and not use_naive else ATOL[dtype]
    torch.testing.assert_close(seq_p, parallel_p, atol=atol, rtol=RTOL)
    torch.testing.assert_close(
        seq_t, parallel_t, atol=5e-2 if dtype is torch.bfloat16 else ATOL[dtype], rtol=RTOL
    )


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="requires Triton on a CUDA device"
)
def test_additive_mask_padding_is_exact() -> None:
    """Padded keys contribute nothing to additive-mask attention or its gradients."""

    n, d = 2, 128
    q = torch.zeros(1, 1, n, d, device=DEVICE)
    k = torch.zeros_like(q)
    v = torch.ones_like(q).requires_grad_()
    mask = torch.zeros(1, 1, n, n, device=DEVICE)
    mask[..., -1] = MASK_CONST

    out = JVPAttn.fwd(q, k, v, attn_mask=mask, USE_NAIVE=False)
    out.sum().backward()

    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        v.grad[..., 0, :], torch.full_like(v.grad[..., 0, :], n), atol=1e-3, rtol=0
    )
    torch.testing.assert_close(
        v.grad[..., 1, :], torch.zeros_like(v.grad[..., 1, :]), atol=1e-6, rtol=0
    )


def test_causal_fallback_matches_sdpa_with_second_order() -> None:
    """Causal masking is exact and the fallback's second derivatives match SDPA."""

    n, d = 32, 16
    q = rand(1, 1, n, d, seed=0) * 0.5
    k = rand(1, 1, n, d, seed=1) * 0.5
    v = rand(1, 1, n, d, seed=2)
    target = rand(1, 1, n, d, seed=3)
    q[0, 0, 0, 0] = 25.0
    k[0, 0, 1, 0] = 25.0  # a future logit that exceeds MASK_CONST

    def run(attn):
        q_, k_, v_ = (t.clone().requires_grad_() for t in (q, k, v))
        out = attn(q_, k_, v_)
        grads = torch.autograd.grad((out * target).sum(), (q_, k_, v_), create_graph=True)
        hess = torch.autograd.grad(sum(g.sum() for g in grads), (q_, k_, v_))
        return tuple(t.detach() for t in (out, *grads, *hess))

    with sdpa_kernel(SDPBackend.MATH):
        expected = run(lambda q, k, v: scaled_dot_product_attention(q, k, v, is_causal=True))

    actual = run(lambda q, k, v: JVPAttn.fwd(q, k, v, causal=True, USE_NAIVE=True))
    assert_close_metrics(actual, expected, torch.float32)


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="requires Triton on a CUDA device"
)
def test_cached_decoding_uses_rectangular_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cached decoding keeps rectangular Q/KV instead of building a square mask."""

    def fail(*args, **kwargs):
        raise AssertionError("padded square mask should not be built")

    monkeypatch.setattr("jvp_flash_attention.jvp_attention.build_internal_mask", fail)

    q = rand(1, 1, 1, 16)
    k = rand(1, 1, 129, 16)
    v = rand(1, 1, 129, 16)
    out = JVPAttn.fwd(q, k, v, USE_NAIVE=False)

    with sdpa_kernel(SDPBackend.MATH):
        expected = scaled_dot_product_attention(q, k, v)

    torch.testing.assert_close(out, expected, atol=ATOL[torch.float32], rtol=RTOL)


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="requires Triton on a CUDA device"
)
@pytest.mark.parametrize("n_ctx", [1, 17, 33])
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_short_sequences_match_sdpa(n_ctx: int, mask_kind: str | None, causal: bool) -> None:
    """Short (padded-and-excised) sequences match SDPA's outputs and gradients."""

    if n_ctx == 1 and mask_kind == "additive":
        pytest.skip("a 1x1 additive mask cannot contain MASK_CONST and still be valid")

    dtype = torch.float32
    q_p, k_p, v_p, *_ = make_qkv_tangents_target(dtype, n_ctx)
    mask = make_attn_mask(mask_kind, dtype, n_ctx)

    with sdpa_kernel(SDPBackend.MATH), enable_grad():
        q_ref, k_ref, v_ref = (t.clone().requires_grad_() for t in (q_p, k_p, v_p))
        ref = scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=mask, is_causal=causal)
        ref.sum().backward()

    q, k, v = (t.clone().requires_grad_() for t in (q_p, k_p, v_p))
    out = JVPAttn.fwd(q, k, v, attn_mask=mask, causal=causal, USE_NAIVE=False)
    out.sum().backward()

    assert_close_metrics(
        (out, q.grad, k.grad, v.grad), (ref, q_ref.grad, k_ref.grad, v_ref.grad), dtype
    )
