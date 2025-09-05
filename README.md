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
from jvp_flash_attention.jvp_attention import attention as jvp_attention

with sdpa_kernel(SDPBackend.MATH):
  # Regular (quadratic) attention
  # x = F.scaled_dot_product_attention(
  #     q,
  #     k,
  #     v,
  #     attn_mask=attn_mask,
  #     dropout_p=attn_dropout_p if self.training else 0.0,
  # )

  # JVP flash attention
  x = jvp_attention(
      q,
      k,
      v,
      attn_mask=attn_mask,
      dropout_p=attn_dropout_p if self.training else 0.0,
  )
```

Contributions or enhancements are welcome!

## Tests

If you want to run the unit tests verifying the correctness of the JVP Flash Attention Triton kernel, run the following command(s).

```bash
python tests/test_jvp_attention.py --dtype {float16,bfloat16,float32}
```

In principle, the kernel should support ROCm systems as well, though it has not yet been tested on them. macOS is currently unsupported.

Results for `float16`:

```
==============================================================================================================
BENCHMARK SUMMARY
==============================================================================================================
Seq Len    Causal   Mask       Method     Time (ms)    Mem (MB)     TFLOP/s      Max Error    Grad Check
--------------------------------------------------------------------------------------------------------------
32         False    additive   sdpa       0.788        0.64           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.496        0.23           0.0 TFLOP/s 1.95e-03     ✓

32         False    boolean    sdpa       0.809        0.65           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.488        0.22           0.0 TFLOP/s 1.95e-03     ✓

32         False    none       sdpa       0.511        0.64           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.493        0.22           0.0 TFLOP/s 1.95e-03     ✓

32         True     none       sdpa       0.858        0.65           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.501        0.22           0.0 TFLOP/s 1.95e-03     ✓

64         False    additive   sdpa       0.809        1.41           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.502        0.47           0.0 TFLOP/s 9.77e-04     ✓

64         False    boolean    sdpa       0.806        1.45           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.488        0.43           0.0 TFLOP/s 9.77e-04     ✓

64         False    none       sdpa       0.539        1.41           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.494        0.43           0.0 TFLOP/s 9.77e-04     ✓

64         True     none       sdpa       0.849        1.42           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.479        0.43           0.0 TFLOP/s 1.95e-03     ✓

128        False    additive   sdpa       0.803        3.28           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.488        1.02           0.1 TFLOP/s 9.77e-04     ✓

128        False    boolean    sdpa       0.826        3.44           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.472        0.86           0.1 TFLOP/s 9.77e-04     ✓

128        False    none       sdpa       0.534        3.28           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.519        0.86           0.1 TFLOP/s 9.77e-04     ✓

128        True     none       sdpa       0.869        3.35           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.497        0.86           0.0 TFLOP/s 1.95e-03     ✓

256        False    additive   sdpa       0.861        9.69           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.511        2.35           0.3 TFLOP/s 9.77e-04     ✓

256        False    boolean    sdpa       0.856        10.32          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.494        1.72           0.3 TFLOP/s 9.77e-04     ✓

256        False    none       sdpa       0.589        9.69           0.1 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.476        1.72           0.4 TFLOP/s 9.77e-04     ✓

256        True     none       sdpa       0.885        9.94           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.480        1.72           0.2 TFLOP/s 1.95e-03     ✓

512        False    additive   sdpa       1.003        31.88          0.3 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.508        5.95           1.3 TFLOP/s 4.88e-04     ✓

512        False    boolean    sdpa       1.122        34.38          0.3 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.515        3.45           1.3 TFLOP/s 4.88e-04     ✓

512        False    none       sdpa       0.750        31.88          0.5 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.496        3.45           1.4 TFLOP/s 4.88e-04     ✓

512        True     none       sdpa       1.185        32.88          0.1 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.496        3.45           0.7 TFLOP/s 1.95e-03     ✓

1024       False    additive   sdpa       2.137        113.77         0.6 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.503        16.89          5.4 TFLOP/s 4.88e-04     ✓

1024       False    boolean    sdpa       2.166        123.77         0.6 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.516        6.89           5.3 TFLOP/s 4.88e-04     ✓

1024       False    none       sdpa       1.934        113.77         0.7 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.520        6.89           5.3 TFLOP/s 4.88e-04     ✓

1024       True     none       sdpa       2.169        117.77         0.3 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.493        6.89           2.8 TFLOP/s 1.95e-03     ✓

2048       False    additive   sdpa       6.884        427.54         0.8 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   0.702        53.79         15.6 TFLOP/s 2.44e-04     ✓

2048       False    boolean    sdpa       6.992        467.54         0.8 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   0.566        13.79         19.3 TFLOP/s 2.44e-04     ✓

2048       False    none       sdpa       5.820        427.54         0.9 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   0.507        13.79         21.6 TFLOP/s 2.44e-04     ✓

2048       True     none       sdpa       6.906        443.54         0.4 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.522        13.79         10.5 TFLOP/s 1.95e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.49 ms         0.49 ms (0.99x) 0.50 ms (1.01x)
32         True     jvp_attn   0.50 ms         N/A             N/A
64         False    jvp_attn   0.49 ms         0.49 ms (0.99x) 0.50 ms (1.02x)
64         True     jvp_attn   0.48 ms         N/A             N/A
128        False    jvp_attn   0.52 ms         0.47 ms (0.91x) 0.49 ms (0.94x)
128        True     jvp_attn   0.50 ms         N/A             N/A
256        False    jvp_attn   0.48 ms         0.49 ms (1.04x) 0.51 ms (1.07x)
256        True     jvp_attn   0.48 ms         N/A             N/A
512        False    jvp_attn   0.50 ms         0.52 ms (1.04x) 0.51 ms (1.02x)
512        True     jvp_attn   0.50 ms         N/A             N/A
1024       False    jvp_attn   0.52 ms         0.52 ms (0.99x) 0.50 ms (0.97x)
1024       True     jvp_attn   0.49 ms         N/A             N/A
2048       False    jvp_attn   0.51 ms         0.57 ms (1.12x) 0.70 ms (1.39x)
2048       True     jvp_attn   0.52 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 3.44x
Min speedup: 1.03x
Max speedup: 13.23x

Accuracy: 28/28 tests passed
✓ All accuracy checks passed!
```

Results for `bfloat16`:

```
==============================================================================================================
BENCHMARK SUMMARY
==============================================================================================================
Seq Len    Causal   Mask       Method     Time (ms)    Mem (MB)     TFLOP/s      Max Error    Grad Check
--------------------------------------------------------------------------------------------------------------
32         False    additive   sdpa       0.764        0.64           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.497        0.23           0.0 TFLOP/s 1.56e-02     ✓

32         False    boolean    sdpa       0.786        0.65           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.468        0.22           0.0 TFLOP/s 1.56e-02     ✓

32         False    none       sdpa       0.494        0.64           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.470        0.22           0.0 TFLOP/s 1.56e-02     ✓

32         True     none       sdpa       0.850        0.65           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.469        0.22           0.0 TFLOP/s 1.56e-02     ✓

64         False    additive   sdpa       0.814        1.41           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.488        0.47           0.0 TFLOP/s 7.81e-03     ✓

64         False    boolean    sdpa       0.916        1.45           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.540        0.43           0.0 TFLOP/s 7.81e-03     ✓

64         False    none       sdpa       0.515        1.41           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.471        0.43           0.0 TFLOP/s 7.81e-03     ✓

64         True     none       sdpa       0.821        1.42           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.471        0.43           0.0 TFLOP/s 1.56e-02     ✓

128        False    additive   sdpa       0.767        3.28           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.487        1.02           0.1 TFLOP/s 7.81e-03     ✓

128        False    boolean    sdpa       0.821        3.44           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.482        0.86           0.1 TFLOP/s 7.81e-03     ✓

128        False    none       sdpa       0.506        3.28           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.486        0.86           0.1 TFLOP/s 7.81e-03     ✓

128        True     none       sdpa       0.843        3.35           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.473        0.86           0.0 TFLOP/s 1.56e-02     ✓

256        False    additive   sdpa       0.822        9.69           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.495        2.35           0.3 TFLOP/s 7.81e-03     ✓

256        False    boolean    sdpa       0.849        10.32          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.470        1.72           0.4 TFLOP/s 7.81e-03     ✓

256        False    none       sdpa       0.602        9.69           0.1 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.465        1.72           0.4 TFLOP/s 3.91e-03     ✓

256        True     none       sdpa       1.028        9.94           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.462        1.72           0.2 TFLOP/s 1.56e-02     ✓

512        False    additive   sdpa       1.104        31.88          0.3 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.617        5.95           1.1 TFLOP/s 3.91e-03     ✓

512        False    boolean    sdpa       1.158        34.38          0.3 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.502        3.45           1.4 TFLOP/s 3.91e-03     ✓

512        False    none       sdpa       0.924        31.88          0.4 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.478        3.45           1.4 TFLOP/s 3.91e-03     ✓

512        True     none       sdpa       0.935        32.88          0.2 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.462        3.45           0.7 TFLOP/s 1.56e-02     ✓

1024       False    additive   sdpa       2.151        113.77         0.6 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.497        16.89          5.5 TFLOP/s 3.91e-03     ✓

1024       False    boolean    sdpa       2.224        123.77         0.6 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.487        6.89           5.6 TFLOP/s 3.91e-03     ✓

1024       False    none       sdpa       1.936        113.77         0.7 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.487        6.89           5.6 TFLOP/s 3.91e-03     ✓

1024       True     none       sdpa       2.149        117.77         0.3 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.484        6.89           2.8 TFLOP/s 1.56e-02     ✓

2048       False    additive   sdpa       6.891        427.54         0.8 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   0.550        53.79         19.9 TFLOP/s 1.95e-03     ✓

2048       False    boolean    sdpa       7.020        467.54         0.8 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   0.550        13.79         19.9 TFLOP/s 1.95e-03     ✓

2048       False    none       sdpa       5.827        427.54         0.9 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   0.532        13.79         20.6 TFLOP/s 1.95e-03     ✓

2048       True     none       sdpa       6.914        443.54         0.4 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.549        13.79         10.0 TFLOP/s 3.12e-02     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.47 ms         0.47 ms (1.00x) 0.50 ms (1.06x)
32         True     jvp_attn   0.47 ms         N/A             N/A
64         False    jvp_attn   0.47 ms         0.54 ms (1.15x) 0.49 ms (1.04x)
64         True     jvp_attn   0.47 ms         N/A             N/A
128        False    jvp_attn   0.49 ms         0.48 ms (0.99x) 0.49 ms (1.00x)
128        True     jvp_attn   0.47 ms         N/A             N/A
256        False    jvp_attn   0.47 ms         0.47 ms (1.01x) 0.49 ms (1.06x)
256        True     jvp_attn   0.46 ms         N/A             N/A
512        False    jvp_attn   0.48 ms         0.50 ms (1.05x) 0.62 ms (1.29x)
512        True     jvp_attn   0.46 ms         N/A             N/A
1024       False    jvp_attn   0.49 ms         0.49 ms (1.00x) 0.50 ms (1.02x)
1024       True     jvp_attn   0.48 ms         N/A             N/A
2048       False    jvp_attn   0.53 ms         0.55 ms (1.04x) 0.55 ms (1.03x)
2048       True     jvp_attn   0.55 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 3.56x
Min speedup: 1.04x
Max speedup: 12.76x

Accuracy: 28/28 tests passed
✓ All accuracy checks passed!
```

Results for `float32`:

```
==============================================================================================================
BENCHMARK SUMMARY
==============================================================================================================
Seq Len    Causal   Mask       Method     Time (ms)    Mem (MB)     TFLOP/s      Max Error    Grad Check
--------------------------------------------------------------------------------------------------------------
32         False    additive   sdpa       0.724        0.51           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.495        0.45           0.0 TFLOP/s 7.21e-03     ✓

32         False    boolean    sdpa       0.718        0.53           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.487        0.43           0.0 TFLOP/s 7.21e-03     ✓

32         False    none       sdpa       0.438        0.51           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.484        0.43           0.0 TFLOP/s 7.22e-03     ✓

32         True     none       sdpa       0.750        0.51           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.475        0.43           0.0 TFLOP/s 6.18e-03     ✓

64         False    additive   sdpa       0.697        1.09           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.475        0.94           0.0 TFLOP/s 7.17e-03     ✓

64         False    boolean    sdpa       0.710        1.17           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.474        0.86           0.0 TFLOP/s 7.17e-03     ✓

64         False    none       sdpa       0.432        1.09           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.478        0.86           0.0 TFLOP/s 7.03e-03     ✓

64         True     none       sdpa       0.783        1.11           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.471        0.86           0.0 TFLOP/s 6.18e-03     ✓

128        False    additive   sdpa       0.755        2.81           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.523        2.03           0.1 TFLOP/s 5.41e-03     ✓

128        False    boolean    sdpa       0.767        3.13           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.508        1.72           0.1 TFLOP/s 5.41e-03     ✓

128        False    none       sdpa       0.467        2.81           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.497        1.72           0.1 TFLOP/s 5.07e-03     ✓

128        True     none       sdpa       0.743        2.88           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.467        1.72           0.0 TFLOP/s 6.18e-03     ✓

256        False    additive   sdpa       0.745        8.75           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.495        4.69           0.3 TFLOP/s 3.41e-03     ✓

256        False    boolean    sdpa       0.790        10.00          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.522        3.44           0.3 TFLOP/s 3.41e-03     ✓

256        False    none       sdpa       0.502        8.75           0.2 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.487        3.44           0.4 TFLOP/s 3.67e-03     ✓

256        True     none       sdpa       0.780        9.00           0.1 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.487        3.44           0.2 TFLOP/s 5.78e-03     ✓

512        False    additive   sdpa       1.013        30.01          0.3 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.503        11.88          1.4 TFLOP/s 3.09e-03     ✓

512        False    boolean    sdpa       0.833        35.01          0.4 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.485        6.88           1.4 TFLOP/s 3.09e-03     ✓

512        False    none       sdpa       0.698        30.01          0.5 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.489        6.88           1.4 TFLOP/s 2.88e-03     ✓

512        True     none       sdpa       1.073        31.01          0.2 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.496        6.88           0.7 TFLOP/s 5.13e-03     ✓

1024       False    additive   sdpa       2.139        110.02         0.6 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.526        33.77          5.2 TFLOP/s 2.84e-03     ✓

1024       False    boolean    sdpa       2.092        130.02         0.7 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.529        13.77          5.2 TFLOP/s 2.84e-03     ✓

1024       False    none       sdpa       1.845        110.02         0.7 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.503        13.77          5.4 TFLOP/s 2.61e-03     ✓

1024       True     none       sdpa       2.107        115.02         0.3 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.523        13.77          2.6 TFLOP/s 5.61e-03     ✓

2048       False    additive   sdpa       6.535        420.04         0.8 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   0.893        107.54        12.3 TFLOP/s 1.57e-03     ✓

2048       False    boolean    sdpa       6.656        500.04         0.8 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   1.167        27.54          9.4 TFLOP/s 1.57e-03     ✓

2048       False    none       sdpa       4.616        420.04         1.2 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   0.759        27.54         14.4 TFLOP/s 1.56e-03     ✓

2048       True     none       sdpa       6.348        436.04         0.4 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.553        27.54          9.9 TFLOP/s 6.47e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.48 ms         0.49 ms (1.00x) 0.50 ms (1.02x)
32         True     jvp_attn   0.48 ms         N/A             N/A
64         False    jvp_attn   0.48 ms         0.47 ms (0.99x) 0.47 ms (0.99x)
64         True     jvp_attn   0.47 ms         N/A             N/A
128        False    jvp_attn   0.50 ms         0.51 ms (1.02x) 0.52 ms (1.05x)
128        True     jvp_attn   0.47 ms         N/A             N/A
256        False    jvp_attn   0.49 ms         0.52 ms (1.07x) 0.49 ms (1.01x)
256        True     jvp_attn   0.49 ms         N/A             N/A
512        False    jvp_attn   0.49 ms         0.49 ms (0.99x) 0.50 ms (1.03x)
512        True     jvp_attn   0.50 ms         N/A             N/A
1024       False    jvp_attn   0.50 ms         0.53 ms (1.05x) 0.53 ms (1.04x)
1024       True     jvp_attn   0.52 ms         N/A             N/A
2048       False    jvp_attn   0.76 ms         1.17 ms (1.54x) 0.89 ms (1.18x)
2048       True     jvp_attn   0.55 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 2.70x
Min speedup: 0.90x
Max speedup: 11.49x

Accuracy: 28/28 tests passed
✓ All accuracy checks passed!
```

## License

This project is covered under the **MIT License**.

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
  version = {0.0.1},
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
