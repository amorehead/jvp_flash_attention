<div align="center">

# JVP Flash Attention

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17050188.svg)](https://doi.org/10.5281/zenodo.17050188)
[![PyPI version](https://badge.fury.io/py/jvp_flash_attention.svg)](https://badge.fury.io/py/jvp_flash_attention)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="main.png" width="600">

</div>

## Description

Flash Attention Triton kernel with support for second-order derivatives, such as Jacobian-Vector Products (JVPs) and Hessian-Vector Products (HVPs)

## Installation

Using `pip`, one can install `jvp_flash_attention` as follows.

```bash
# Install package
pip install jvp_flash_attention

# [OPTIONAL, for development] Install package and pre-commit hooks
pip install -e .
pre-commit install
```

## Usage

Once installed, one can use `jvp_flash_attention` in place of PyTorch's `scaled_dot_product_attention` as follows.

```python
import torch.nn.functional as F

from torch.nn.attention import SDPBackend, sdpa_kernel
from jvp_flash_attention.jvp_attention import JVPAttn, attention as jvp_attention

with sdpa_kernel(SDPBackend.MATH):
  # Regular (quadratic) attention
  x = F.scaled_dot_product_attention(
      q,
      k,
      v,
      attn_mask=attn_mask,
      dropout_p=attn_dropout_p if self.training else 0.0,
  )

# JVP flash attention
x = jvp_attention(
    q,
    k,
    v,
    attn_mask=attn_mask,
    # dropout_p=attn_dropout_p if self.training else 0.0,  # NOTE: Attention dropout is currently unsupported
)
```

Anecdotally, one can also swap out `F.scaled_dot_product_attention` with `jvp_attention` **even for pretrained models** with minimal impact on numerical accuracy.

> Note: If calling `torch.func.jvp` manually in your model's forward pass like
> `pred, df = torch.func.jvp(*(lambda x_jvp: model(x_jvp), (x,), (gt,)))`,
> make sure to use JVP Flash Attention in your model as `model = lambda q, k, v: JVPAttn.fwd_dual(q, k, v)` instead of as `model = lambda q, k, v: jvp_attention(q, k, v)` to ensure each input's tangent vectors are computed [prior](https://github.com/amorehead/jvp_flash_attention/issues/10) to running PyTorch's `autograd` engine. Models that rely on `torch.autograd.grad` to compute higher-order derivatives in their forward pass (e.g., energy-based models) should not require this change.

Contributions or enhancements are welcome!

## Results

### Loss matching

Model training with either `F.scaled_dot_product_attention` or `JVPAttn.fwd_dual` produces the same loss trajectory.

<img width="1369" height="704" alt="image" src="https://github.com/user-attachments/assets/70df8ddc-e558-4eb7-a4ba-0464dabb1b40" />

### Speed matching

Model training with either `F.scaled_dot_product_attention` or `JVPAttn.fwd_dual` achieves the same iteration speed.

<img width="1369" height="704" alt="image" src="https://github.com/user-attachments/assets/6b52cb5d-ed0f-447f-9488-6f3057a99cc7" />

> Note: The following results can be reproduced (for `float32` precision) by running `python tests/test_jvp_attention.py --dtype float32`.

### Time scaling

`jvp_attention` outscales the speed of (`SDPBackend.MATH`-based) `F.scaled_dot_product_attention` when calculating second-order derivatives.

<div align="center">

<img src="./float32_time_scaling.png" width="800">

</div>

### Memory scaling

`jvp_attention` improves the memory usage of (`SDPBackend.MATH`-based) `F.scaled_dot_product_attention` when calculating second-order derivatives.

<div align="center">

<img src="./float32_mem_scaling.png" width="800">

</div>

## Tests

If you want to run all the unit tests verifying the correctness of the JVP Flash Attention Triton kernel, run the following command(s).

```bash
python tests/test_jvp_attention.py --dtype {float16,bfloat16,float32}
```

In principle, the kernel should support ROCm systems as well, though it has not yet been tested on them. macOS is currently unsupported except using a CPU-only backend.

Full results for `float16`:

```
==============================================================================================================
BENCHMARK SUMMARY
==============================================================================================================
Seq Len    Causal   Mask       Method     Time (ms)    Mem (MB)     TFLOP/s      Max Error    Grad Check
--------------------------------------------------------------------------------------------------------------
32         False    additive   sdpa       0.821        3.09           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.723        1.08           0.0 TFLOP/s 1.83e+01     ✗

32         False    boolean    sdpa       0.961        3.14           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.504        1.03           0.0 TFLOP/s 3.91e-03     ✓

32         False    none       sdpa       0.576        3.09           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.447        1.03           0.0 TFLOP/s 1.95e-03     ✓

32         True     none       sdpa       0.934        3.10           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.458        1.03           0.0 TFLOP/s 3.91e-03     ✓

64         False    additive   sdpa       0.860        6.75           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.847        2.26           0.1 TFLOP/s 2.23e+00     ✗

64         False    boolean    sdpa       0.908        6.94           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.521        2.07           0.1 TFLOP/s 3.91e-03     ✓

64         False    none       sdpa       0.542        6.75           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.414        2.07           0.1 TFLOP/s 1.95e-03     ✓

64         True     none       sdpa       0.888        6.77           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.437        2.07           0.1 TFLOP/s 2.20e-03     ✓

128        False    additive   sdpa       0.834        16.51          0.1 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.750        4.89           0.3 TFLOP/s 3.91e-03     ✓

128        False    boolean    sdpa       0.840        17.26          0.1 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.520        4.14           0.4 TFLOP/s 3.91e-03     ✓

128        False    none       sdpa       0.610        16.51          0.2 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.459        4.14           0.4 TFLOP/s 9.77e-04     ✓

128        True     none       sdpa       1.053        16.57          0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.438        4.14           0.2 TFLOP/s 2.44e-03     ✓

256        False    additive   sdpa       0.829        47.77          0.5 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.738        12.02          1.1 TFLOP/s 3.91e-03     ✓

256        False    boolean    sdpa       0.872        50.77          0.5 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.482        8.27           1.7 TFLOP/s 3.91e-03     ✓

256        False    none       sdpa       0.812        47.27          0.5 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.460        8.27           1.8 TFLOP/s 9.77e-04     ✓

256        True     none       sdpa       0.964        47.52          0.2 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.436        8.27           0.9 TFLOP/s 3.91e-03     ✓

512        False    additive   sdpa       1.416        153.55         1.2 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.715        30.55          4.6 TFLOP/s 1.95e-03     ✓

512        False    boolean    sdpa       1.441        165.05         1.1 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.500        16.55          6.6 TFLOP/s 1.95e-03     ✓

512        False    none       sdpa       1.374        153.05         1.2 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.407        16.55          8.1 TFLOP/s 4.88e-04     ✓

512        True     none       sdpa       1.402        154.05         0.6 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.460        16.55          3.6 TFLOP/s 2.93e-03     ✓

1024       False    additive   sdpa       4.963        546.84         1.3 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   1.183        96.84         11.1 TFLOP/s 1.95e-03     ✓

1024       False    boolean    sdpa       4.991        594.84         1.3 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.622        33.84         21.1 TFLOP/s 1.95e-03     ✓

1024       False    none       sdpa       4.227        546.84         1.6 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.420        33.84         31.3 TFLOP/s 4.88e-04     ✓

1024       True     none       sdpa       4.861        550.84         0.7 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.469        33.84         14.0 TFLOP/s 3.91e-03     ✓

2048       False    additive   sdpa       18.773       2052.19        1.4 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   3.379        336.19        15.6 TFLOP/s 1.95e-03     ✓

2048       False    boolean    sdpa       18.815       2244.19        1.4 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   1.674        66.19         31.4 TFLOP/s 1.95e-03     ✓

2048       False    none       sdpa       16.156       2052.19        1.6 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   1.186        66.19         44.3 TFLOP/s 4.88e-04     ✓

2048       True     none       sdpa       18.587       2068.19        0.7 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.720        66.19         36.5 TFLOP/s 1.95e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.45 ms         0.50 ms (1.13x) 0.72 ms (1.62x)
32         True     jvp_attn   0.46 ms         N/A             N/A
64         False    jvp_attn   0.41 ms         0.52 ms (1.26x) 0.85 ms (2.05x)
64         True     jvp_attn   0.44 ms         N/A             N/A
128        False    jvp_attn   0.46 ms         0.52 ms (1.13x) 0.75 ms (1.63x)
128        True     jvp_attn   0.44 ms         N/A             N/A
256        False    jvp_attn   0.46 ms         0.48 ms (1.05x) 0.74 ms (1.60x)
256        True     jvp_attn   0.44 ms         N/A             N/A
512        False    jvp_attn   0.41 ms         0.50 ms (1.23x) 0.72 ms (1.76x)
512        True     jvp_attn   0.46 ms         N/A             N/A
1024       False    jvp_attn   0.42 ms         0.62 ms (1.48x) 1.18 ms (2.82x)
1024       True     jvp_attn   0.47 ms         N/A             N/A
2048       False    jvp_attn   1.19 ms         1.67 ms (1.41x) 3.38 ms (2.85x)
2048       True     jvp_attn   0.72 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 4.50x
Min speedup: 1.02x
Max speedup: 25.82x

Accuracy: 26/28 tests passed
⚠️  Some accuracy checks failed

Failed configurations:
  - Seq=32, Causal=False, Mask=additive
  - Seq=64, Causal=False, Mask=additive
```

Full results for `bfloat16`:

```
==============================================================================================================
BENCHMARK SUMMARY
==============================================================================================================
Seq Len    Causal   Mask       Method     Time (ms)    Mem (MB)     TFLOP/s      Max Error    Grad Check
--------------------------------------------------------------------------------------------------------------
32         False    additive   sdpa       0.864        3.09           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.773        1.08           0.0 TFLOP/s 1.84e+01     ✗

32         False    boolean    sdpa       0.949        3.14           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.569        1.03           0.0 TFLOP/s 3.12e-02     ✓

32         False    none       sdpa       0.662        3.09           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.447        1.03           0.0 TFLOP/s 1.56e-02     ✓

32         True     none       sdpa       0.945        3.10           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.469        1.03           0.0 TFLOP/s 3.12e-02     ✓

64         False    additive   sdpa       0.923        6.75           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   1.149        2.26           0.0 TFLOP/s 2.23e+00     ✗

64         False    boolean    sdpa       0.910        6.94           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.518        2.07           0.1 TFLOP/s 3.12e-02     ✓

64         False    none       sdpa       0.554        6.75           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.427        2.07           0.1 TFLOP/s 1.56e-02     ✓

64         True     none       sdpa       0.886        6.77           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.458        2.07           0.1 TFLOP/s 3.12e-02     ✓

128        False    additive   sdpa       0.860        16.51          0.1 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.896        4.89           0.2 TFLOP/s 1.56e-02     ✓

128        False    boolean    sdpa       0.891        17.26          0.1 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.771        4.14           0.3 TFLOP/s 1.56e-02     ✓

128        False    none       sdpa       0.578        16.51          0.2 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.467        4.14           0.4 TFLOP/s 7.81e-03     ✓

128        True     none       sdpa       0.917        16.57          0.1 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.447        4.14           0.2 TFLOP/s 3.12e-02     ✓

256        False    additive   sdpa       0.822        47.77          0.5 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.734        12.02          1.1 TFLOP/s 1.56e-02     ✓

256        False    boolean    sdpa       0.880        50.77          0.5 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.532        8.27           1.5 TFLOP/s 1.56e-02     ✓

256        False    none       sdpa       0.597        47.27          0.7 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.441        8.27           1.9 TFLOP/s 7.81e-03     ✓

256        True     none       sdpa       0.869        47.52          0.2 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.469        8.27           0.9 TFLOP/s 1.56e-02     ✓

512        False    additive   sdpa       1.429        153.55         1.1 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.710        30.55          4.6 TFLOP/s 1.56e-02     ✓

512        False    boolean    sdpa       1.714        165.05         1.0 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.552        16.55          5.9 TFLOP/s 1.56e-02     ✓

512        False    none       sdpa       1.314        153.05         1.2 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.403        16.55          8.2 TFLOP/s 7.81e-03     ✓

512        True     none       sdpa       1.788        154.05         0.5 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.432        16.55          3.8 TFLOP/s 3.12e-02     ✓

1024       False    additive   sdpa       5.720        546.84         1.1 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   1.133        96.84         11.6 TFLOP/s 1.56e-02     ✓

1024       False    boolean    sdpa       5.376        594.84         1.2 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.634        33.84         20.7 TFLOP/s 1.56e-02     ✓

1024       False    none       sdpa       4.646        546.84         1.4 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.423        33.84         31.1 TFLOP/s 3.91e-03     ✓

1024       True     none       sdpa       5.566        550.84         0.6 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.466        33.84         14.1 TFLOP/s 1.56e-02     ✓

2048       False    additive   sdpa       21.231       2052.19        1.2 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   3.735        336.19        14.1 TFLOP/s 1.56e-02     ✓

2048       False    boolean    sdpa       21.626       2244.19        1.2 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   1.926        66.19         27.3 TFLOP/s 1.56e-02     ✓

2048       False    none       sdpa       18.311       2052.19        1.4 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   1.139        66.19         46.1 TFLOP/s 3.91e-03     ✓

2048       True     none       sdpa       20.748       2068.19        0.6 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.750        66.19         35.0 TFLOP/s 3.12e-02     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.45 ms         0.57 ms (1.27x) 0.77 ms (1.73x)
32         True     jvp_attn   0.47 ms         N/A             N/A
64         False    jvp_attn   0.43 ms         0.52 ms (1.21x) 1.15 ms (2.69x)
64         True     jvp_attn   0.46 ms         N/A             N/A
128        False    jvp_attn   0.47 ms         0.77 ms (1.65x) 0.90 ms (1.92x)
128        True     jvp_attn   0.45 ms         N/A             N/A
256        False    jvp_attn   0.44 ms         0.53 ms (1.21x) 0.73 ms (1.66x)
256        True     jvp_attn   0.47 ms         N/A             N/A
512        False    jvp_attn   0.40 ms         0.55 ms (1.37x) 0.71 ms (1.76x)
512        True     jvp_attn   0.43 ms         N/A             N/A
1024       False    jvp_attn   0.42 ms         0.63 ms (1.50x) 1.13 ms (2.68x)
1024       True     jvp_attn   0.47 ms         N/A             N/A
2048       False    jvp_attn   1.14 ms         1.93 ms (1.69x) 3.74 ms (3.28x)
2048       True     jvp_attn   0.75 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 4.75x
Min speedup: 0.80x
Max speedup: 27.65x

Accuracy: 26/28 tests passed
⚠️  Some accuracy checks failed

Failed configurations:
  - Seq=32, Causal=False, Mask=additive
  - Seq=64, Causal=False, Mask=additive
```

Full results for `float32`:

```
==============================================================================================================
BENCHMARK SUMMARY
==============================================================================================================
Seq Len    Causal   Mask       Method     Time (ms)    Mem (MB)     TFLOP/s      Max Error    Grad Check
--------------------------------------------------------------------------------------------------------------
32         False    additive   sdpa       0.770        2.44           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.812        2.16           0.0 TFLOP/s 2.31e-02     ✓

32         False    boolean    sdpa       0.830        2.53           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.575        2.07           0.0 TFLOP/s 9.16e-03     ✓

32         False    none       sdpa       0.491        2.44           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.528        2.07           0.0 TFLOP/s 7.83e-03     ✓

32         True     none       sdpa       0.831        2.44           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.457        2.07           0.0 TFLOP/s 8.60e-03     ✓

64         False    additive   sdpa       0.859        5.25           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.793        4.51           0.1 TFLOP/s 1.24e-02     ✓

64         False    boolean    sdpa       0.778        5.62           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.522        4.13           0.1 TFLOP/s 1.23e-02     ✓

64         False    none       sdpa       0.501        5.25           0.1 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.437        4.13           0.1 TFLOP/s 7.03e-03     ✓

64         True     none       sdpa       0.810        5.27           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.450        4.13           0.1 TFLOP/s 1.05e-02     ✓

128        False    additive   sdpa       0.869        13.51          0.1 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.697        9.76           0.3 TFLOP/s 9.14e-03     ✓

128        False    boolean    sdpa       0.832        15.76          0.1 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.527        8.26           0.4 TFLOP/s 8.91e-03     ✓

128        False    none       sdpa       0.458        14.26          0.2 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.610        8.26           0.3 TFLOP/s 5.07e-03     ✓

128        True     none       sdpa       0.817        14.32          0.1 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.478        8.26           0.2 TFLOP/s 1.05e-02     ✓

256        False    additive   sdpa       0.786        43.27          0.5 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.689        23.77          1.2 TFLOP/s 9.98e-03     ✓

256        False    boolean    sdpa       0.754        48.52          0.5 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.514        17.02          1.6 TFLOP/s 9.93e-03     ✓

256        False    none       sdpa       0.596        43.27          0.7 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.461        17.77          1.8 TFLOP/s 4.03e-03     ✓

256        True     none       sdpa       0.837        43.52          0.2 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.424        17.77          1.0 TFLOP/s 9.61e-03     ✓

512        False    additive   sdpa       1.383        144.80         1.2 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.793        57.80          4.1 TFLOP/s 7.29e-03     ✓

512        False    boolean    sdpa       1.342        168.80         1.2 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.792        33.80          4.1 TFLOP/s 7.22e-03     ✓

512        False    none       sdpa       1.479        144.80         1.1 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.453        33.80          7.2 TFLOP/s 3.94e-03     ✓

512        True     none       sdpa       1.582        145.80         0.5 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.501        33.80          3.3 TFLOP/s 6.56e-03     ✓

1024       False    additive   sdpa       5.448        528.09         1.2 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   2.148        168.09         6.1 TFLOP/s 6.90e-03     ✓

1024       False    boolean    sdpa       5.131        624.09         1.3 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   1.139        66.09         11.5 TFLOP/s 6.85e-03     ✓

1024       False    none       sdpa       4.339        528.09         1.5 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.538        66.09         24.4 TFLOP/s 2.92e-03     ✓

1024       True     none       sdpa       5.262        532.09         0.6 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.437        66.09         15.0 TFLOP/s 8.65e-03     ✓

2048       False    additive   sdpa       19.849       2016.19        1.3 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   6.468        576.19         8.1 TFLOP/s 7.22e-03     ✓

2048       False    boolean    sdpa       20.017       2400.19        1.3 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   3.247        132.19        16.2 TFLOP/s 7.16e-03     ✓

2048       False    none       sdpa       16.573       2016.19        1.6 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   1.883        132.19        27.9 TFLOP/s 2.61e-03     ✓

2048       True     none       sdpa       19.577       2032.19        0.7 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   1.114        132.19        23.6 TFLOP/s 7.71e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.53 ms         0.57 ms (1.09x) 0.81 ms (1.54x)
32         True     jvp_attn   0.46 ms         N/A             N/A
64         False    jvp_attn   0.44 ms         0.52 ms (1.20x) 0.79 ms (1.81x)
64         True     jvp_attn   0.45 ms         N/A             N/A
128        False    jvp_attn   0.61 ms         0.53 ms (0.87x) 0.70 ms (1.14x)
128        True     jvp_attn   0.48 ms         N/A             N/A
256        False    jvp_attn   0.46 ms         0.51 ms (1.11x) 0.69 ms (1.49x)
256        True     jvp_attn   0.42 ms         N/A             N/A
512        False    jvp_attn   0.45 ms         0.79 ms (1.75x) 0.79 ms (1.75x)
512        True     jvp_attn   0.50 ms         N/A             N/A
1024       False    jvp_attn   0.54 ms         1.14 ms (2.12x) 2.15 ms (3.99x)
1024       True     jvp_attn   0.44 ms         N/A             N/A
2048       False    jvp_attn   1.88 ms         3.25 ms (1.72x) 6.47 ms (3.43x)
2048       True     jvp_attn   1.11 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 3.37x
Min speedup: 0.75x
Max speedup: 17.57x

Accuracy: 28/28 tests passed
✓ All accuracy checks passed!
```

Note: Based on these results, for all precision types, it is recommended to provide a boolean `attn_mask` to `jvp_attention()` where possible.

## License

This project is covered under the **MIT License**.

## Copyright

JVP Flash Attention (jvp_flash_attention) Copyright (c) 2025, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

**NOTICE.** This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights. As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

## Citing this work

If you use the code associated with this package or otherwise find this work useful, please use GitHub's `Cite this repository` feature or the BibTeX below.

```bibtex
@software{Morehead_JVP_Flash_Attention_2025,
  author = {Morehead, Alex},
  doi = {10.5281/zenodo.17050188},
  license = {MIT},
  month = sep,
  title = {{JVP Flash Attention}},
  url = {https://github.com/amorehead/jvp_flash_attention},
  version = {0.11.0},
  year = {2025}
}
```

## Acknowledgements

`jvp_flash_attention` builds upon the contributions and insights from the following sources:

- [flash-attention](https://github.com/Dao-AILab/flash-attention)
  - [JVP Triton kernel thread](https://github.com/Dao-AILab/flash-attention/issues/1672)
    - [benjamin-dinkelmann](https://gist.github.com/benjamin-dinkelmann)
    - *[Birch-san](https://github.com/Birch-san)*
    - [dabeschte](https://github.com/dabeschte)
    - [IsaacYQH](https://gist.github.com/IsaacYQH)
    - [KohakuBlueleaf](https://github.com/KohakuBlueleaf)
    - [leon](https://github.com/leon532)
    - [limsanky](https://github.com/limsanky)
    - [lucidrains](https://github.com/lucidrains)
    - [Peterande](https://gist.github.com/Peterande)
    - *[Ryu1845](https://github.com/Ryu1845)*
    - [tridao](https://github.com/tridao)

Thank you to each and every contributor!
