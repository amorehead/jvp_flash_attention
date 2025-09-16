from __future__ import annotations

import gc
import os
import random
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, NamedTuple

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor, enable_grad
from torch.nn import MSELoss
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

try:
    import matplotlib.pyplot as plt
    import numpy as np

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from jvp_flash_attention.jvp_attention import MASK_CONST, JVPAttn


def mpi_to_flops(ms_per_iter: float, flop_count: int) -> float:
    """Convert milliseconds per iteration to FLOPS.

    Args:
        ms_per_iter: Milliseconds per iteration.
        flop_count: Number of floating point operations.

    Returns:
        The number of FLOPS.
    """
    iters_per_second = 1e3 / ms_per_iter
    return iters_per_second * flop_count


def fmt_flops(flops: int) -> str:
    """Return a string representation of FLOPS in TFLOP/s."""
    return f"{flops / 1e12:5.1f} TFLOP/s"


def get_attention_flop_count(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool,
    is_jvp: bool = False,
) -> int:
    """Calculate FLOPs for attention operations.

    Args:
        batch_size: Batch size.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        head_dim: Dimension of each attention head.
        is_causal: Whether the attention is causal.
        is_jvp: Whether to include JVP (Jacobian-vector product) FLOPs.

    Returns:
        The total FLOPs for the attention operation.
    """
    # Base attention FLOPs
    qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
    av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim

    total_flops = qk_flops + softmax_flops + av_flops

    if is_causal:
        total_flops = total_flops // 2

    if is_jvp:
        total_flops = total_flops * 2

    return total_flops


def measure_memory_usage(f: Callable[[], Any]) -> tuple[float, float]:
    """Measure GPU memory usage of a function.

    Args:
        f: The function to measure.

    Returns:
        Tuple of (allocated_mb, reserved_mb) memory in megabytes.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()

    f()

    torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    allocated_mb = (peak_allocated - initial_allocated) / (1024 * 1024)
    reserved_mb = (peak_reserved - initial_reserved) / (1024 * 1024)

    return allocated_mb, reserved_mb


def benchmark_function(
    f: Callable[[], Any], warmup_iters: int = 10, benchmark_iters: int = 100
) -> float:
    """Benchmark a function's execution time.

    Args:
        f: The function to benchmark.
        warmup_iters: Number of warmup iterations.
        benchmark_iters: Number of benchmark iterations.

    Returns:
        Average time per iteration in milliseconds.
    """
    # Warmup
    for _ in range(warmup_iters):
        f()

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for _ in range(benchmark_iters):
        f()

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) * 1000 / benchmark_iters
    return avg_time_ms


class QKV(NamedTuple):
    """Query, Key, Value tensors."""

    q: Tensor
    k: Tensor
    v: Tensor


class UnpackedDualQKV(NamedTuple):
    """Unpacked dual Query, Key, Value tensors."""

    primal: QKV
    tangent: QKV


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for numerical validation."""

    primal_error: float
    tangent_error: float
    loss_error: float
    q_grad_error: float
    k_grad_error: float
    v_grad_error: float
    tolerance: float = 5e-3

    @property
    def max_error(self) -> float:
        """Return the maximum error across all metrics."""
        return max(
            self.primal_error,
            self.tangent_error,
            self.loss_error,
            self.q_grad_error,
            self.k_grad_error,
            self.v_grad_error,
        )

    def is_accurate(self) -> bool:
        """Check if all errors are within tolerance."""
        return self.max_error < self.tolerance


class BenchmarkResult(NamedTuple):
    """Results from a single benchmark run."""

    seq_len: int
    is_causal: bool
    method: str  # 'sdpa' or 'jvp_attn'
    time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    mask_type: str  # 'none', 'boolean', or 'additive'
    flops: int | None = None
    accuracy: AccuracyMetrics | None = None


@dataclass
class Args:
    """Training arguments."""

    bsz: int
    model_dim: int
    head_dim: int
    seq_lengths: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512, 1024, 2048])
    warmup_iters: int = 10
    benchmark_iters: int = 100
    dtype: str = "float16"
    seed: int = 42
    test_masks: bool = True
    validate_gradients: bool = True
    benchmark_performance: bool = True
    mask_prob: float = 0.9  # Probability of masking out an attention weight

    @staticmethod
    def get_parser() -> ArgumentParser:
        """Get the argument parser for training."""
        parser = ArgumentParser()
        parser.add_argument("--bsz", default=2, type=int)
        parser.add_argument("--model-dim", default=768, type=int)
        parser.add_argument("--head-dim", default=64, type=int)
        parser.add_argument(
            "--seq-lengths", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024, 2048]
        )
        parser.add_argument("--warmup-iters", default=10, type=int)
        parser.add_argument("--benchmark-iters", default=100, type=int)
        parser.add_argument(
            "--dtype", default="float16", choices=["float16", "float32", "bfloat16"]
        )
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument(
            "--no-test-masks",
            action="store_true",
            help="Skip testing with attention masks",
        )
        parser.add_argument(
            "--no-validate-gradients",
            action="store_true",
            help="Skip gradient validation",
        )
        parser.add_argument(
            "--no-benchmark-performance",
            action="store_true",
            help="Skip performance benchmarking",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.9,
            type=float,
            help="Probability of masking out attention weights",
        )
        return parser

    @staticmethod
    def from_namespace(namespace: Namespace) -> Args:
        """Create Args from a namespace."""
        kwargs = vars(namespace)

        test_masks = not kwargs.pop("no_test_masks", False)
        validate_gradients = not kwargs.pop("no_validate_gradients", False)
        benchmark_performance = not kwargs.pop("no_benchmark_performance", False)

        kwargs["test_masks"] = test_masks
        kwargs["validate_gradients"] = validate_gradients
        kwargs["benchmark_performance"] = benchmark_performance

        return Args(**kwargs)


def create_test_tensors(
    args: Args, seq_len: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, ...]:
    """Create test tensors for benchmarking.

    Args:
        args: The training arguments.
        seq_len: The sequence length.
        device: The device to create the tensors on.
        dtype: The data type of the tensors.

    Returns:
        Tuple of (q_p, q_t, k_p, k_t, v_p, v_t, target) tensors.
    """
    gen = torch.Generator(device=device).manual_seed(args.seed)
    heads = args.model_dim // args.head_dim

    tensors = tuple(
        torch.randn(
            args.bsz,
            heads,
            seq_len,
            args.head_dim,
            device=device,
            dtype=dtype,
            generator=gen,
        )
        for _ in range(7)
    )

    return tensors


def create_attention_mask(
    args: Args,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_type: str,
) -> Tensor | None:
    """Create an attention mask for testing.

    Args:
        args: The training arguments.
        seq_len: The sequence length.
        device: The device to create the mask on.
        dtype: The data type of the mask.
        mask_type: Type of mask ('none', 'boolean', or 'additive').

    Returns:
        The attention mask tensor or None if mask_type is 'none'.
    """
    if mask_type == "none":
        return None

    gen = torch.Generator(device=device).manual_seed(args.seed + 1000)  # Different seed for masks
    heads = args.model_dim // args.head_dim

    if mask_type == "boolean":
        # Create a boolean mask where True means "attend" and False means "ignore"
        # We'll create a random mask with some positions masked out
        mask = (
            torch.rand(args.bsz, heads, seq_len, seq_len, device=device, generator=gen)
            > args.mask_prob
        )
        # mask[0, :-1, :, :2] = (
        #     True  # Ensure first two columns of the first batch element (except for its last head) are True
        # )
        # mask[1, :-1, :, -2:] = (
        #     True  # Ensure last two columns of the second batch element (except for its last head) are True
        # )

        # Find completely masked heads
        fully_masked = ~mask.view(args.bsz, heads, -1).any(dim=2)

        # For each fully masked head, unmask some random positions
        if fully_masked.any():
            print("  ⚠️  Some heads were fully masked; unmasking some positions to avoid this.")
            for b in range(args.bsz):
                for h in range(heads):
                    if fully_masked[b, h]:
                        num_to_unmask = max(1, seq_len * seq_len // 10)
                        indices = torch.randperm(seq_len * seq_len, device=device, generator=gen)[
                            :num_to_unmask
                        ]
                        mask[b, h].view(-1)[indices] = True

        return mask

    elif mask_type == "additive":
        # Create an additive mask with values to be added to attention scores
        # Use -inf (MASK_CONST) for positions to ignore, 0 for positions to attend
        rand_mask = torch.rand(args.bsz, heads, seq_len, seq_len, device=device, generator=gen)
        mask = torch.where(rand_mask > args.mask_prob, 0.0, MASK_CONST)
        # Convert to the target dtype
        mask = mask.to(dtype)
        # mask[0, :-1, :, :2] = (
        #     0.0  # Ensure first two columns of the first batch element (except for its last head) are zeros
        # )
        # mask[1, :-1, :, -2:] = (
        #     0.0  # Ensure last two columns of the second batch element (except for its last head) are zeros
        # )

        # Find completely masked heads
        fully_masked = (mask.view(args.bsz, heads, -1) == MASK_CONST).all(dim=2)

        # For each fully masked head, unmask some random positions
        if fully_masked.any():
            print("  ⚠️  Some heads were fully masked; unmasking some positions to avoid this.")
            for b in range(args.bsz):
                for h in range(heads):
                    if fully_masked[b, h]:
                        num_to_unmask = max(1, seq_len * seq_len // 10)
                        indices = torch.randperm(seq_len * seq_len, device=device, generator=gen)[
                            :num_to_unmask
                        ]
                        mask[b, h].view(-1)[indices] = 0.0

        return mask

    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def loss_fn(out: Tensor, target: Tensor) -> Tensor:
    """Compute the mean squared error loss.

    Args:
        out: The output tensor.
        target: The target tensor.

    Returns:
        The mean squared error loss.
    """
    return (out - target).square().mean()


def make_qkv_with_grad(
    q_p: Tensor, k_p: Tensor, v_p: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor
) -> QKV:
    """Make a QKV tuple with gradients enabled.

    Args:
        q_p: The query projection tensor.
        k_p: The key projection tensor.
        v_p: The value projection tensor.
        q_t: The query tangent tensor.
        k_t: The key tangent tensor.
        v_t: The value tangent tensor.

    Returns:
        A QKV tuple containing the primal and tangent QKV tensors.
    """
    # Create dual tensors
    q = fwAD.make_dual(q_p, q_t)
    k = fwAD.make_dual(k_p, k_t)
    v = fwAD.make_dual(v_p, v_t)

    for t in (q, k, v):
        t.requires_grad = True
        t.retain_grad()

    return QKV(q, k, v)


def make_qkv(q_p: Tensor, k_p: Tensor, v_p: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor) -> QKV:
    """Make a QKV tuple from the given tensors with dual numbers.

    Args:
        q_p: The query projection tensor.
        k_p: The key projection tensor.
        v_p: The value projection tensor.
        q_t: The query tangent tensor.
        k_t: The key tangent tensor.
        v_t: The value tangent tensor.

    Returns:
        A QKV tuple containing the primal and tangent QKV tensors.
    """
    q = fwAD.make_dual(q_p, q_t)
    k = fwAD.make_dual(k_p, k_t)
    v = fwAD.make_dual(v_p, v_t)
    return QKV(q, k, v)


def make_qkv_unpacked(
    q_p: Tensor, k_p: Tensor, v_p: Tensor, q_t: Tensor, k_t: Tensor, v_t: Tensor
) -> UnpackedDualQKV:
    """Make an unpacked dual QKV from the given tensors.

    Args:
        q_p: The query projection tensor.
        k_p: The key projection tensor.
        v_p: The value projection tensor.
        q_t: The query tangent tensor.
        k_t: The key tangent tensor.
        v_t: The value tangent tensor.

    Returns:
        An unpacked dual QKV containing the primal and tangent QKV tensors.
    """
    for t in (q_p, k_p, v_p):
        t.requires_grad = True
        t.retain_grad()

    return UnpackedDualQKV(
        primal=QKV(
            q=q_p,
            k=k_p,
            v=v_p,
        ),
        tangent=QKV(
            q=q_t,
            k=k_t,
            v=v_t,
        ),
    )


def compute_absolute_error(*tensors: Tensor) -> float:
    """Compute the maximum absolute pairwise error between all tensors.

    Args:
        tensors: The input tensors to compare.

    Returns:
        The maximum absolute pairwise error.
    """
    if len(tensors) < 2:
        raise ValueError("At least two tensors are required to compute absolute error.")
    max_error = 0.0
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            diff = (tensors[i] - tensors[j]).abs().max().item()
            if diff > max_error:
                max_error = diff
    return max_error


def validate_accuracy_and_gradients(
    q_p: Tensor,
    k_p: Tensor,
    v_p: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    target: Tensor,
    is_causal: bool,
    attn_mask: Tensor | None = None,
    tolerance: float = 4e-3,
    loss_tolerance: float = 5e-4,
    grad_tolerance: float = 5e-4,
) -> AccuracyMetrics:
    """Validate numerical accuracy and gradient matching between SDPA and JVP attention.

    Args:
        q_p: The query projection tensor.
        k_p: The key projection tensor.
        v_p: The value projection tensor.
        q_t: The query tangent tensor.
        k_t: The key tangent tensor.
        v_t: The value tangent tensor.
        target: The target tensor.
        is_causal: Whether the attention is causal.
        attn_mask: Optional attention mask tensor.
        tolerance: The tolerance for primal errors.
        loss_tolerance: The tolerance for loss errors.
        grad_tolerance: The tolerance for gradient errors.

    Returns:
        AccuracyMetrics containing all error measurements.
    """
    with sdpa_kernel(SDPBackend.MATH), fwAD.dual_level(), enable_grad():
        # Run SDPA
        q0, k0, v0 = make_qkv_with_grad(
            q_p.clone(), k_p.clone(), v_p.clone(), q_t.clone(), k_t.clone(), v_t.clone()
        )

        sdpa_out = scaled_dot_product_attention(
            q0, k0, v0, attn_mask=attn_mask, is_causal=is_causal
        )
        sdpa_out.retain_grad()
        sdpa_op, sdpa_ot = fwAD.unpack_dual(sdpa_out)

        loss0 = loss_fn(sdpa_out, target)
        loss0.backward()

        assert not any(
            t.grad.isnan().any() or t.grad.isinf().any() for t in (q0, k0, v0)
        ), "NaN/Inf in SDPA input gradients."

        # Run JVP Attention
        q1, k1, v1 = make_qkv_with_grad(
            q_p.clone(), k_p.clone(), v_p.clone(), q_t.clone(), k_t.clone(), v_t.clone()
        )

        jvp_out = JVPAttn.fwd_dual(q1, k1, v1, attn_mask=attn_mask, causal=is_causal)
        jvp_out.retain_grad()
        jvp_op, jvp_ot = fwAD.unpack_dual(jvp_out)

        loss1 = loss_fn(jvp_out, target)
        loss1.backward()

        assert not any(
            t.grad.isnan().any() or t.grad.isinf().any() for t in (q1, k1, v1)
        ), "NaN/Inf in JVP input gradients."

    mse_fn = MSELoss()
    with enable_grad():
        # Run JVP Attention with torch.func.jvp
        qkv_p, qkv_t = make_qkv_unpacked(
            q_p.clone(),
            k_p.clone(),
            v_p.clone(),
            q_t.clone(),
            k_t.clone(),
            v_t.clone(),
        )

        jvp_func_op, jvp_func_ot = torch.func.jvp(
            partial(JVPAttn.fwd_dual, attn_mask=attn_mask, causal=is_causal), qkv_p, qkv_t
        )
        jvp_func_op.retain_grad()

        loss2: Tensor = mse_fn(jvp_func_op, target)
        loss2.backward()

        q2, k2, v2 = qkv_p

        assert not any(
            t.grad.isnan().any() or t.grad.isinf().any() for t in (q2, k2, v2)
        ), "NaN/Inf in JVP (func) input gradients."

    # Compute errors
    primal_error = compute_absolute_error(jvp_func_op, jvp_op, sdpa_op)
    tangent_error = compute_absolute_error(jvp_func_ot, jvp_ot, sdpa_ot)
    loss_error = compute_absolute_error(loss2, loss1, loss0)

    # Compute gradient errors
    q_grad_error = compute_absolute_error(q2.grad, q1.grad, q0.grad)
    k_grad_error = compute_absolute_error(k2.grad, k1.grad, k0.grad)
    v_grad_error = compute_absolute_error(v2.grad, v1.grad, v0.grad)

    metrics = AccuracyMetrics(
        primal_error=primal_error,
        tangent_error=tangent_error,
        loss_error=loss_error,
        q_grad_error=q_grad_error,
        k_grad_error=k_grad_error,
        v_grad_error=v_grad_error,
    )

    # Validate using torch.testing.assert_close
    try:
        torch.testing.assert_close(jvp_op, sdpa_op, atol=tolerance, rtol=1e-5)
        torch.testing.assert_close(
            # TODO: Improve this (causal) accuracy for longer sequence lengths
            jvp_func_op,
            sdpa_op,
            atol=tolerance,
            rtol=1e-5,
        )

        # TODO: Improve these tangent accuracies
        torch.testing.assert_close(
            jvp_ot,
            sdpa_ot,
            atol=tolerance,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            jvp_func_ot,
            sdpa_ot,
            atol=tolerance,
            rtol=1e-5,
        )

        torch.testing.assert_close(loss1, loss0, atol=loss_tolerance, rtol=1e-5)
        torch.testing.assert_close(loss2, loss0, atol=loss_tolerance, rtol=1e-5)

        torch.testing.assert_close(q1.grad, q0.grad, atol=grad_tolerance, rtol=1e-5)
        torch.testing.assert_close(k1.grad, k0.grad, atol=grad_tolerance, rtol=1e-5)
        torch.testing.assert_close(v1.grad, v0.grad, atol=grad_tolerance, rtol=1e-5)

        torch.testing.assert_close(q2.grad, q0.grad, atol=grad_tolerance, rtol=1e-5)
        torch.testing.assert_close(k2.grad, k0.grad, atol=grad_tolerance, rtol=1e-5)
        torch.testing.assert_close(v2.grad, v0.grad, atol=grad_tolerance, rtol=1e-5)

    except AssertionError as e:
        print(f"  ⚠️  Accuracy validation failed (causal={is_causal}): {e}")

    return metrics


def run_benchmark_suite(args: Args) -> list[BenchmarkResult]:
    """Run comprehensive benchmarks across different configurations.

    Args:
        args: The command-line arguments for the benchmark.

    Returns:
        A list of benchmark results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    tolerance_map = {
        "float16": 4e-3,
        "float32": 2.35e-2,
        "bfloat16": 3.2e-2,
    }
    tolerance = tolerance_map[args.dtype]

    results = []

    # Define mask types to test
    mask_types = ["none"]
    if args.test_masks:
        mask_types.extend(["boolean", "additive"])

    for seq_len in args.seq_lengths:
        print(f"\n{'='*60}")
        print(f"Benchmarking sequence length: {seq_len}")
        print(f"{'='*60}")

        # Create test tensors
        q_p, q_t, k_p, k_t, v_p, v_t, target = create_test_tensors(args, seq_len, device, dtype)

        for mask_type in mask_types:
            # Create attention mask if needed
            attn_mask = create_attention_mask(args, seq_len, device, dtype, mask_type)

            for is_causal in [False, True]:
                print(f"\nCausal: {is_causal}, Mask: {mask_type}")
                print("-" * 40)

                if is_causal and mask_type != "none":
                    print("  Skipping invalid combination of causal + mask")
                    continue

                # Validate accuracy and gradients first
                if args.validate_gradients:
                    print("Validating accuracy and gradients...")
                    accuracy_metrics = validate_accuracy_and_gradients(
                        q_p,
                        k_p,
                        v_p,
                        q_t,
                        k_t,
                        v_t,
                        target,
                        is_causal,
                        attn_mask=attn_mask,
                        tolerance=tolerance,
                    )
                    accuracy_metrics.tolerance = tolerance

                    print(f"  Primal error: {accuracy_metrics.primal_error:.2e}")
                    print(f"  Tangent error: {accuracy_metrics.tangent_error:.2e}")
                    print(f"  Loss error: {accuracy_metrics.loss_error:.2e}")
                    print(f"  Q gradient error: {accuracy_metrics.q_grad_error:.2e}")
                    print(f"  K gradient error: {accuracy_metrics.k_grad_error:.2e}")
                    print(f"  V gradient error: {accuracy_metrics.v_grad_error:.2e}")

                    if accuracy_metrics.is_accurate():
                        print("  ✓ All accuracy checks passed!")
                    else:
                        print(f"  ⚠️  Max error {accuracy_metrics.max_error:.2e} exceeds tolerance")
                else:
                    accuracy_metrics = None

                # Benchmark performance
                with sdpa_kernel(SDPBackend.MATH), fwAD.dual_level(), enable_grad():
                    # Create functions for benchmarking
                    def run_sdpa():
                        """Run SDPA attention."""
                        q, k, v = make_qkv(
                            q_p.clone(),
                            k_p.clone(),
                            v_p.clone(),
                            q_t.clone(),
                            k_t.clone(),
                            v_t.clone(),
                        )
                        out = scaled_dot_product_attention(
                            q, k, v, attn_mask=attn_mask, is_causal=is_causal
                        )

                    def run_jvp_attn():
                        """Run JVP attention."""
                        q, k, v = make_qkv(
                            q_p.clone(),
                            k_p.clone(),
                            v_p.clone(),
                            q_t.clone(),
                            k_t.clone(),
                            v_t.clone(),
                        )
                        out = JVPAttn.fwd_dual(q, k, v, attn_mask=attn_mask, causal=is_causal)

                    if not args.benchmark_performance:
                        print("  Skipping performance benchmarking.")
                        results.append(
                            BenchmarkResult(
                                seq_len=seq_len,
                                is_causal=is_causal,
                                method="sdpa",
                                time_ms=np.nan,
                                memory_allocated_mb=np.nan,
                                memory_reserved_mb=np.nan,
                                mask_type=mask_type,
                                flops=np.nan,
                                accuracy=None,
                            )
                        )
                        results.append(
                            BenchmarkResult(
                                seq_len=seq_len,
                                is_causal=is_causal,
                                method="jvp_attn",
                                time_ms=np.nan,
                                memory_allocated_mb=np.nan,
                                memory_reserved_mb=np.nan,
                                mask_type=mask_type,
                                flops=np.nan,
                                accuracy=accuracy_metrics,
                            )
                        )
                        continue

                    print("\nBenchmarking performance...")
                    heads = args.model_dim // args.head_dim

                    # Measure SDPA performance
                    sdpa_time = benchmark_function(
                        run_sdpa, args.warmup_iters, args.benchmark_iters
                    )
                    sdpa_mem_alloc, sdpa_mem_reserved = measure_memory_usage(run_sdpa)
                    sdpa_flops = get_attention_flop_count(
                        args.bsz, heads, seq_len, args.head_dim, is_causal, is_jvp=False
                    )

                    # Measure JVP Attention performance
                    jvp_time = benchmark_function(
                        run_jvp_attn, args.warmup_iters, args.benchmark_iters
                    )
                    jvp_mem_alloc, jvp_mem_reserved = measure_memory_usage(run_jvp_attn)
                    jvp_flops = get_attention_flop_count(
                        args.bsz, heads, seq_len, args.head_dim, is_causal, is_jvp=True
                    )

                    # Store results
                    results.append(
                        BenchmarkResult(
                            seq_len=seq_len,
                            is_causal=is_causal,
                            method="sdpa",
                            time_ms=sdpa_time,
                            memory_allocated_mb=sdpa_mem_alloc,
                            memory_reserved_mb=sdpa_mem_reserved,
                            mask_type=mask_type,
                            flops=sdpa_flops,
                            accuracy=None,
                        )
                    )

                    results.append(
                        BenchmarkResult(
                            seq_len=seq_len,
                            is_causal=is_causal,
                            method="jvp_attn",
                            time_ms=jvp_time,
                            memory_allocated_mb=jvp_mem_alloc,
                            memory_reserved_mb=jvp_mem_reserved,
                            mask_type=mask_type,
                            flops=jvp_flops,
                            accuracy=accuracy_metrics,
                        )
                    )

                    # Print results
                    print("PyTorch SDPA:")
                    print(f"  Time: {sdpa_time:.3f} ms")
                    print(
                        f"  Memory (alloc/reserved): {sdpa_mem_alloc:.2f}/{sdpa_mem_reserved:.2f} MB"
                    )
                    print(f"  FLOPS: {fmt_flops(mpi_to_flops(sdpa_time, sdpa_flops))}")

                    print("\nJVP Attention:")
                    print(f"  Time: {jvp_time:.3f} ms")
                    print(
                        f"  Memory (alloc/reserved): {jvp_mem_alloc:.2f}/{jvp_mem_reserved:.2f} MB"
                    )
                    print(f"  FLOPS: {fmt_flops(mpi_to_flops(jvp_time, jvp_flops))}")

                    print(f"\nSpeedup: {sdpa_time/jvp_time:.2f}x")
                    print(f"Memory ratio: {jvp_mem_alloc/sdpa_mem_alloc:.2f}x")

    return results


def print_summary_table(results: list[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results.

    Args:
        results: The list of benchmark results to summarize.
    """
    print("\n" + "=" * 110)
    print("BENCHMARK SUMMARY")
    print("=" * 110)

    # Group results by seq_len, causal, and mask_type
    from collections import defaultdict

    grouped = defaultdict(dict)

    for r in results:
        key = (r.seq_len, r.is_causal, r.mask_type)
        grouped[key][r.method] = r

    # Print header
    print(
        f"{'Seq Len':<10} {'Causal':<8} {'Mask':<10} {'Method':<10} "
        f"{'Time (ms)':<12} {'Mem (MB)':<12} {'TFLOP/s':<12} "
        f"{'Max Error':<12} {'Grad Check':<10}"
    )
    print("-" * 110)

    for (seq_len, is_causal, mask_type), methods in sorted(grouped.items()):
        for method in ["sdpa", "jvp_attn"]:
            if method in methods:
                r = methods[method]
                flops_str = fmt_flops(mpi_to_flops(r.time_ms, r.flops)) if r.flops else "N/A"

                if r.accuracy:
                    error_str = f"{r.accuracy.max_error:.2e}"
                    grad_check = "✓" if r.accuracy.is_accurate() else "✗"
                else:
                    error_str = "baseline"
                    grad_check = "N/A"

                print(
                    f"{seq_len:<10} {str(is_causal):<8} {mask_type:<10} {method:<10} "
                    f"{r.time_ms:<12.3f} {r.memory_allocated_mb:<12.2f} "
                    f"{flops_str:<12} {error_str:<12} {grad_check:<10}"
                )
        print()


def print_mask_comparison_table(results: list[BenchmarkResult]) -> None:
    """Print a comparison table showing the impact of different mask types.

    Args:
        results: The list of benchmark results to analyze.
    """
    print("\n" + "=" * 80)
    print("MASK TYPE PERFORMANCE COMPARISON")
    print("=" * 80)

    # Group by seq_len, is_causal, and method
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(dict))

    for r in results:
        grouped[(r.seq_len, r.is_causal, r.method)][r.mask_type] = r

    print(
        f"{'Seq Len':<10} {'Causal':<8} {'Method':<10} "
        f"{'No Mask':<15} {'Boolean Mask':<15} {'Additive Mask':<15}"
    )
    print("-" * 80)

    for (seq_len, is_causal, method), mask_results in sorted(grouped.items()):
        if method == "jvp_attn":  # Only show JVP results for clarity
            none_time = mask_results.get("none", None)
            bool_time = mask_results.get("boolean", None)
            add_time = mask_results.get("additive", None)

            none_str = f"{none_time.time_ms:.2f} ms" if none_time else "N/A"
            bool_str = f"{bool_time.time_ms:.2f} ms" if bool_time else "N/A"
            add_str = f"{add_time.time_ms:.2f} ms" if add_time else "N/A"

            # Add relative performance
            if none_time and bool_time:
                bool_str += f" ({bool_time.time_ms/none_time.time_ms:.2f}x)"
            if none_time and add_time:
                add_str += f" ({add_time.time_ms/none_time.time_ms:.2f}x)"

            print(
                f"{seq_len:<10} {str(is_causal):<8} {method:<10} "
                f"{none_str:<15} {bool_str:<15} {add_str:<15}"
            )


def plot_benchmark_results(
    results: list[BenchmarkResult], args: Args, verbose: bool = False
) -> None:
    """Generate, save, and display plots summarizing benchmark results.

    Args:
        results: The list of benchmark results to plot.
        args: The command-line arguments, used for filename generation.
        verbose: Whether to print verbose output.
    """
    if not PLOTTING_AVAILABLE:
        print("\nmatplotlib and/or numpy not found. Skipping plotting.")
        return

    from collections import defaultdict

    # Group results by (is_causal, mask_type) for creating subplots
    grouped = defaultdict(list)
    for r in results:
        key = (r.is_causal, r.mask_type)
        grouped[key].append(r)

    num_configs = len(grouped)
    if num_configs == 0:
        return

    # --- 1. Create and Populate the Performance Figure ---
    fig_perf, axes_perf = plt.subplots(1, num_configs, figsize=(6 * num_configs, 5), squeeze=False)
    fig_perf.suptitle("Performance Speedup (JVP Attention vs. SDPA)", fontsize=16)

    # --- 2. Create and Populate the Memory Figure ---
    fig_mem, axes_mem = plt.subplots(1, num_configs, figsize=(6 * num_configs, 5), squeeze=False)
    fig_mem.suptitle("Peak Allocated Memory Comparison", fontsize=16)

    # Iterate through data to draw on the axes of BOTH figures
    for i, ((is_causal, mask_type), config_results) in enumerate(grouped.items()):
        config_results.sort(key=lambda r: r.seq_len)

        # Fixed: Use the correct method names "jvp_attn" and "sdpa"
        jvp_results = [r for r in config_results if r.method == "jvp_attn"]
        sdpa_results = [r for r in config_results if r.method == "sdpa"]

        if not jvp_results or not sdpa_results:
            # Debug print to help identify issues
            if verbose:
                print(
                    f"Warning: Missing results for config (causal={is_causal}, mask={mask_type})"
                )
                print(f"  Available methods: {set(r.method for r in config_results)}")
            continue

        seq_lens = [r.seq_len for r in jvp_results]
        jvp_times = np.array([r.time_ms for r in jvp_results])
        sdpa_times = np.array([r.time_ms for r in sdpa_results])
        jvp_mems = np.array([r.memory_allocated_mb for r in jvp_results])
        sdpa_mems = np.array([r.memory_allocated_mb for r in sdpa_results])
        speedup = sdpa_times / jvp_times
        x = np.arange(len(seq_lens))

        # Draw on the performance subplot
        ax_perf = axes_perf[0, i]
        bar_perf = ax_perf.bar(x, speedup, width=0.5, color="g")
        ax_perf.bar_label(bar_perf, fmt=lambda val: f"{val:.2f}x")
        ax_perf.axhline(1.0, color="grey", linestyle="--")
        ax_perf.set(
            ylabel="Speedup (SDPA Time / JVP Time)",
            xlabel="Sequence Length",
            title=f"Causal={is_causal}, Mask={mask_type}",
            xticks=x,
            xticklabels=seq_lens,
            ylim=(0, max(1.1, np.max(speedup) * 1.15)),
        )

        # Draw on the memory subplot
        ax_mem = axes_mem[0, i]
        width = 0.35
        rects1 = ax_mem.bar(x - width / 2, sdpa_mems, width, label="PyTorch SDPA")
        rects2 = ax_mem.bar(x + width / 2, jvp_mems, width, label="JVP Attention")
        ax_mem.bar_label(rects1, padding=3, fmt=lambda val: f"{val:.1f}")
        ax_mem.bar_label(rects2, padding=3, fmt=lambda val: f"{val:.1f}")
        ax_mem.set(
            ylabel="Peak Allocated Memory (MB)",
            xlabel="Sequence Length",
            title=f"Causal={is_causal}, Mask={mask_type}",
            xticks=x,
            xticklabels=seq_lens,
            ylim=(0, max(np.max(sdpa_mems), np.max(jvp_mems)) * 1.25),
        )
        ax_mem.legend()

    # --- 3. Finalize and Save Each Figure Individually ---
    plot_dir = "tests"
    os.makedirs(plot_dir, exist_ok=True)
    mask_suffix = "_with_masks" if args.test_masks else ""

    # Finalize and save the performance plot
    perf_plot_path = os.path.join(plot_dir, f"{args.dtype}_jvp_attention_perf{mask_suffix}.png")
    fig_perf.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_perf.savefig(perf_plot_path, dpi=150)
    if verbose:
        print(f"Saved performance plot to {perf_plot_path}")

    # Finalize and save the memory plot
    mem_plot_path = os.path.join(plot_dir, f"{args.dtype}_jvp_attention_mem{mask_suffix}.png")
    fig_mem.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig_mem.savefig(mem_plot_path, dpi=150)
    if verbose:
        print(f"Saved memory plot to {mem_plot_path}")

    # --- 4. Show and Close ---
    plt.show()

    # Explicitly close figures to free memory
    plt.close(fig_perf)
    plt.close(fig_mem)


def main(args: Args) -> None:
    """Main benchmarking loop."""
    print("Flash Attention JVP Kernel Benchmark with Mask Testing")
    print(
        f"Configuration: bsz={args.bsz}, model_dim={args.model_dim}, "
        f"head_dim={args.head_dim}, dtype={args.dtype}"
    )
    print(f"Mask testing: {'Enabled' if args.test_masks else 'Disabled'}")
    print(f"Gradient validation: {'Enabled' if args.validate_gradients else 'Disabled'}")
    print(f"Performance benchmarking: {'Enabled' if args.benchmark_performance else 'Disabled'}")
    if args.test_masks:
        print(f"Mask probability: {args.mask_prob}")

    # Seed everything
    random.seed(args.seed)
    if PLOTTING_AVAILABLE:
        np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    results = run_benchmark_suite(args)

    # Print summary tables
    print_summary_table(results)
    plot_benchmark_results(results, args)

    # If masks were tested, print comparison table
    if args.test_masks:
        print_mask_comparison_table(results)

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    # Calculate average speedup
    speedups = []
    for i in range(0, len(results), 2):  # Assuming pairs of sdpa/jvp_attn
        if i + 1 < len(results):
            sdpa_result = results[i]
            jvp_result = results[i + 1]
            if sdpa_result.method == "sdpa" and jvp_result.method == "jvp_attn":
                speedup = sdpa_result.time_ms / jvp_result.time_ms
                speedups.append(speedup)

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        min_speedup = min(speedups)
        max_speedup = max(speedups)
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Min speedup: {min_speedup:.2f}x")
        print(f"Max speedup: {max_speedup:.2f}x")

    # Calculate accuracy statistics
    accuracy_results = [r for r in results if r.accuracy is not None]
    if accuracy_results:
        all_accurate = all(r.accuracy.is_accurate() for r in accuracy_results)
        num_accurate = sum(1 for r in accuracy_results if r.accuracy.is_accurate())
        print(f"\nAccuracy: {num_accurate}/{len(accuracy_results)} tests passed")
        if all_accurate:
            print("✓ All accuracy checks passed!")
        else:
            print("⚠️  Some accuracy checks failed")

            # Show which configurations failed
            failed_configs = [
                (r.seq_len, r.is_causal, r.mask_type)
                for r in accuracy_results
                if not r.accuracy.is_accurate()
            ]
            if failed_configs:
                print("\nFailed configurations:")
                for seq_len, is_causal, mask_type in failed_configs:
                    print(f"  - Seq={seq_len}, Causal={is_causal}, Mask={mask_type}")

    # Save results to file
    import json

    # Convert results to JSON-serializable format
    results_data = []
    for r in results:
        result_dict = r._asdict()
        if r.accuracy:
            result_dict["accuracy"] = {
                "primal_error": r.accuracy.primal_error,
                "tangent_error": r.accuracy.tangent_error,
                "loss_error": r.accuracy.loss_error,
                "q_grad_error": r.accuracy.q_grad_error,
                "k_grad_error": r.accuracy.k_grad_error,
                "v_grad_error": r.accuracy.v_grad_error,
                "max_error": r.accuracy.max_error,
                "is_accurate": r.accuracy.is_accurate(),
            }
        results_data.append(result_dict)

    # Include configuration in filename
    mask_suffix = "_with_masks" if args.test_masks else ""
    output_filepath = os.path.join(
        "tests", f"{args.dtype}_test_jvp_attention_results{mask_suffix}.json"
    )
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Save both results and configuration
    output_data = {
        "configuration": {
            "bsz": args.bsz,
            "model_dim": args.model_dim,
            "head_dim": args.head_dim,
            "seq_lengths": args.seq_lengths,
            "dtype": args.dtype,
            "seed": args.seed,
            "test_masks": args.test_masks,
            "validate_gradients": args.validate_gradients,
            "benchmark_performance": args.benchmark_performance,
            "mask_prob": args.mask_prob if args.test_masks else None,
        },
        "results": results_data,
        "summary": {
            "avg_speedup": avg_speedup if speedups else None,
            "min_speedup": min_speedup if speedups else None,
            "max_speedup": max_speedup if speedups else None,
            "accuracy_rate": (
                f"{num_accurate}/{len(accuracy_results)}" if accuracy_results else "N/A"
            ),
        },
    }

    with open(output_filepath, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_filepath}")


if __name__ == "__main__":
    parser = Args.get_parser()
    namespace = parser.parse_args()
    args = Args.from_namespace(namespace)
    main(args)
