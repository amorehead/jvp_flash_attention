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
32         False    additive   sdpa       0.847        0.64           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.501        0.23           0.0 TFLOP/s 1.95e-03     ✓

32         False    boolean    sdpa       0.888        0.65           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.482        0.22           0.0 TFLOP/s 1.95e-03     ✓

32         False    none       sdpa       0.551        0.64           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.474        0.22           0.0 TFLOP/s 1.95e-03     ✓

32         True     none       sdpa       0.852        0.65           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.471        0.22           0.0 TFLOP/s 1.95e-03     ✓

64         False    additive   sdpa       0.788        1.41           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.518        0.47           0.0 TFLOP/s 9.77e-04     ✓

64         False    boolean    sdpa       0.804        1.45           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.473        0.43           0.0 TFLOP/s 9.77e-04     ✓

64         False    none       sdpa       0.555        1.41           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.472        0.43           0.0 TFLOP/s 9.77e-04     ✓

64         True     none       sdpa       0.836        1.42           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.479        0.43           0.0 TFLOP/s 1.95e-03     ✓

128        False    additive   sdpa       0.881        3.28           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.512        1.02           0.1 TFLOP/s 9.77e-04     ✓

128        False    boolean    sdpa       1.174        3.44           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.510        0.86           0.1 TFLOP/s 9.77e-04     ✓

128        False    none       sdpa       0.812        3.28           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.500        0.86           0.1 TFLOP/s 9.77e-04     ✓

128        True     none       sdpa       0.954        3.35           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.473        0.86           0.0 TFLOP/s 1.95e-03     ✓

256        False    additive   sdpa       1.189        9.69           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.503        2.35           0.3 TFLOP/s 9.77e-04     ✓

256        False    boolean    sdpa       1.115        10.32          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.482        1.72           0.4 TFLOP/s 9.77e-04     ✓

256        False    none       sdpa       1.036        9.69           0.1 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.475        1.72           0.4 TFLOP/s 9.77e-04     ✓

256        True     none       sdpa       1.125        9.94           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.496        1.72           0.2 TFLOP/s 1.95e-03     ✓

512        False    additive   sdpa       1.530        31.88          0.2 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.485        5.95           1.4 TFLOP/s 4.88e-04     ✓

512        False    boolean    sdpa       1.644        34.38          0.2 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.481        3.45           1.4 TFLOP/s 4.88e-04     ✓

512        False    none       sdpa       1.585        31.88          0.2 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.511        3.45           1.3 TFLOP/s 4.88e-04     ✓

512        True     none       sdpa       1.980        32.88          0.1 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.528        3.45           0.6 TFLOP/s 1.95e-03     ✓

1024       False    additive   sdpa       4.025        113.77         0.3 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.523        16.89          5.2 TFLOP/s 4.88e-04     ✓

1024       False    boolean    sdpa       4.368        123.77         0.3 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.522        6.89           5.2 TFLOP/s 4.88e-04     ✓

1024       False    none       sdpa       3.512        113.77         0.4 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.497        6.89           5.5 TFLOP/s 4.88e-04     ✓

1024       True     none       sdpa       4.061        117.77         0.2 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.546        6.89           2.5 TFLOP/s 1.95e-03     ✓

2048       False    additive   sdpa       12.266       427.54         0.4 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   0.931        53.79         11.8 TFLOP/s 2.44e-04     ✓

2048       False    boolean    sdpa       12.501       467.54         0.4 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   1.063        13.79         10.3 TFLOP/s 2.44e-04     ✓

2048       False    none       sdpa       11.218       427.54         0.5 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   0.699        13.79         15.7 TFLOP/s 2.44e-04     ✓

2048       True     none       sdpa       12.849       443.54         0.2 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.772        13.79          7.1 TFLOP/s 1.95e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.47 ms         0.48 ms (1.02x) 0.50 ms (1.06x)
32         True     jvp_attn   0.47 ms         N/A             N/A
64         False    jvp_attn   0.47 ms         0.47 ms (1.00x) 0.52 ms (1.10x)
64         True     jvp_attn   0.48 ms         N/A             N/A
128        False    jvp_attn   0.50 ms         0.51 ms (1.02x) 0.51 ms (1.02x)
128        True     jvp_attn   0.47 ms         N/A             N/A
256        False    jvp_attn   0.48 ms         0.48 ms (1.02x) 0.50 ms (1.06x)
256        True     jvp_attn   0.50 ms         N/A             N/A
512        False    jvp_attn   0.51 ms         0.48 ms (0.94x) 0.48 ms (0.95x)
512        True     jvp_attn   0.53 ms         N/A             N/A
1024       False    jvp_attn   0.50 ms         0.52 ms (1.05x) 0.52 ms (1.05x)
1024       True     jvp_attn   0.55 ms         N/A             N/A
2048       False    jvp_attn   0.70 ms         1.06 ms (1.52x) 0.93 ms (1.33x)
2048       True     jvp_attn   0.77 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 4.68x
Min speedup: 1.16x
Max speedup: 16.64x

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
32         False    additive   sdpa       0.787        0.64           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.488        0.23           0.0 TFLOP/s 1.56e-02     ✓

32         False    boolean    sdpa       0.849        0.65           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.489        0.22           0.0 TFLOP/s 1.56e-02     ✓

32         False    none       sdpa       0.513        0.64           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.476        0.22           0.0 TFLOP/s 1.56e-02     ✓

32         True     none       sdpa       0.860        0.65           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.492        0.22           0.0 TFLOP/s 1.56e-02     ✓

64         False    additive   sdpa       0.795        1.41           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.501        0.47           0.0 TFLOP/s 7.81e-03     ✓

64         False    boolean    sdpa       0.846        1.45           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.466        0.43           0.0 TFLOP/s 7.81e-03     ✓

64         False    none       sdpa       0.558        1.41           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.510        0.43           0.0 TFLOP/s 7.81e-03     ✓

64         True     none       sdpa       0.892        1.42           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.475        0.43           0.0 TFLOP/s 1.56e-02     ✓

128        False    additive   sdpa       0.830        3.28           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.521        1.02           0.1 TFLOP/s 7.81e-03     ✓

128        False    boolean    sdpa       0.841        3.44           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.479        0.86           0.1 TFLOP/s 7.81e-03     ✓

128        False    none       sdpa       0.788        3.28           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.495        0.86           0.1 TFLOP/s 7.81e-03     ✓

128        True     none       sdpa       0.863        3.35           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.513        0.86           0.0 TFLOP/s 1.56e-02     ✓

256        False    additive   sdpa       1.357        9.69           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.483        2.35           0.4 TFLOP/s 7.81e-03     ✓

256        False    boolean    sdpa       1.111        10.32          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.495        1.72           0.3 TFLOP/s 7.81e-03     ✓

256        False    none       sdpa       1.012        9.69           0.1 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.481        1.72           0.4 TFLOP/s 3.91e-03     ✓

256        True     none       sdpa       1.202        9.94           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.481        1.72           0.2 TFLOP/s 1.56e-02     ✓

512        False    additive   sdpa       1.586        31.88          0.2 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.483        5.95           1.4 TFLOP/s 3.91e-03     ✓

512        False    boolean    sdpa       1.446        34.38          0.2 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.518        3.45           1.3 TFLOP/s 3.91e-03     ✓

512        False    none       sdpa       1.531        31.88          0.2 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.484        3.45           1.4 TFLOP/s 3.91e-03     ✓

512        True     none       sdpa       1.624        32.88          0.1 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.495        3.45           0.7 TFLOP/s 1.56e-02     ✓

1024       False    additive   sdpa       3.934        113.77         0.3 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.486        16.89          5.6 TFLOP/s 3.91e-03     ✓

1024       False    boolean    sdpa       4.111        123.77         0.3 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.498        6.89           5.5 TFLOP/s 3.91e-03     ✓

1024       False    none       sdpa       3.992        113.77         0.3 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.474        6.89           5.8 TFLOP/s 3.91e-03     ✓

1024       True     none       sdpa       4.127        117.77         0.2 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.478        6.89           2.9 TFLOP/s 1.56e-02     ✓

2048       False    additive   sdpa       12.082       427.54         0.5 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   0.993        53.79         11.0 TFLOP/s 1.95e-03     ✓

2048       False    boolean    sdpa       12.604       467.54         0.4 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   1.069        13.79         10.2 TFLOP/s 1.95e-03     ✓

2048       False    none       sdpa       10.963       427.54         0.5 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   0.801        13.79         13.7 TFLOP/s 1.95e-03     ✓

2048       True     none       sdpa       13.057       443.54         0.2 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.605        13.79          9.1 TFLOP/s 3.12e-02     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.48 ms         0.49 ms (1.03x) 0.49 ms (1.02x)
32         True     jvp_attn   0.49 ms         N/A             N/A
64         False    jvp_attn   0.51 ms         0.47 ms (0.91x) 0.50 ms (0.98x)
64         True     jvp_attn   0.47 ms         N/A             N/A
128        False    jvp_attn   0.50 ms         0.48 ms (0.97x) 0.52 ms (1.05x)
128        True     jvp_attn   0.51 ms         N/A             N/A
256        False    jvp_attn   0.48 ms         0.49 ms (1.03x) 0.48 ms (1.00x)
256        True     jvp_attn   0.48 ms         N/A             N/A
512        False    jvp_attn   0.48 ms         0.52 ms (1.07x) 0.48 ms (1.00x)
512        True     jvp_attn   0.50 ms         N/A             N/A
1024       False    jvp_attn   0.47 ms         0.50 ms (1.05x) 0.49 ms (1.03x)
1024       True     jvp_attn   0.48 ms         N/A             N/A
2048       False    jvp_attn   0.80 ms         1.07 ms (1.34x) 0.99 ms (1.24x)
2048       True     jvp_attn   0.60 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 4.79x
Min speedup: 1.08x
Max speedup: 21.60x

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
32         False    additive   jvp_attn   0.509        0.45           0.0 TFLOP/s 7.21e-03     ✓

32         False    boolean    sdpa       0.743        0.53           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.510        0.43           0.0 TFLOP/s 7.21e-03     ✓

32         False    none       sdpa       0.519        0.51           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.549        0.43           0.0 TFLOP/s 7.22e-03     ✓

32         True     none       sdpa       0.782        0.51           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.551        0.43           0.0 TFLOP/s 6.18e-03     ✓

64         False    additive   sdpa       0.765        1.09           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.507        0.94           0.0 TFLOP/s 7.17e-03     ✓

64         False    boolean    sdpa       0.748        1.17           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.498        0.86           0.0 TFLOP/s 7.17e-03     ✓

64         False    none       sdpa       0.555        1.09           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.537        0.86           0.0 TFLOP/s 7.03e-03     ✓

64         True     none       sdpa       0.876        1.11           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.525        0.86           0.0 TFLOP/s 6.18e-03     ✓

128        False    additive   sdpa       0.806        2.81           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.499        2.03           0.1 TFLOP/s 5.41e-03     ✓

128        False    boolean    sdpa       0.775        3.13           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.481        1.72           0.1 TFLOP/s 5.41e-03     ✓

128        False    none       sdpa       0.858        2.81           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.592        1.72           0.1 TFLOP/s 5.07e-03     ✓

128        True     none       sdpa       1.040        2.88           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.479        1.72           0.0 TFLOP/s 6.18e-03     ✓

256        False    additive   sdpa       0.951        8.75           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.526        4.69           0.3 TFLOP/s 3.41e-03     ✓

256        False    boolean    sdpa       0.994        10.00          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.481        3.44           0.4 TFLOP/s 3.41e-03     ✓

256        False    none       sdpa       1.020        8.75           0.1 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.495        3.44           0.3 TFLOP/s 3.67e-03     ✓

256        True     none       sdpa       0.991        9.00           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.505        3.44           0.2 TFLOP/s 5.78e-03     ✓

512        False    additive   sdpa       1.284        30.01          0.3 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.521        11.88          1.3 TFLOP/s 3.09e-03     ✓

512        False    boolean    sdpa       1.450        35.01          0.2 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.549        6.88           1.2 TFLOP/s 3.09e-03     ✓

512        False    none       sdpa       1.423        30.01          0.2 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.475        6.88           1.4 TFLOP/s 2.88e-03     ✓

512        True     none       sdpa       1.558        31.01          0.1 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.496        6.88           0.7 TFLOP/s 5.13e-03     ✓

1024       False    additive   sdpa       3.836        110.02         0.4 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.727        33.77          3.8 TFLOP/s 2.84e-03     ✓

1024       False    boolean    sdpa       4.327        130.02         0.3 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.657        13.77          4.2 TFLOP/s 2.84e-03     ✓

1024       False    none       sdpa       3.384        110.02         0.4 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.578        13.77          4.7 TFLOP/s 2.61e-03     ✓

1024       True     none       sdpa       3.875        115.02         0.2 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.805        13.77          1.7 TFLOP/s 5.61e-03     ✓

2048       False    additive   sdpa       11.817       420.04         0.5 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   2.141        107.54         5.1 TFLOP/s 1.57e-03     ✓

2048       False    boolean    sdpa       11.919       500.04         0.5 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   1.607        27.54          6.8 TFLOP/s 1.57e-03     ✓

2048       False    none       sdpa       9.251        420.04         0.6 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   1.932        27.54          5.7 TFLOP/s 1.56e-03     ✓

2048       True     none       sdpa       11.846       436.04         0.2 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   1.312        27.54          4.2 TFLOP/s 6.47e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.55 ms         0.51 ms (0.93x) 0.51 ms (0.93x)
32         True     jvp_attn   0.55 ms         N/A             N/A
64         False    jvp_attn   0.54 ms         0.50 ms (0.93x) 0.51 ms (0.94x)
64         True     jvp_attn   0.53 ms         N/A             N/A
128        False    jvp_attn   0.59 ms         0.48 ms (0.81x) 0.50 ms (0.84x)
128        True     jvp_attn   0.48 ms         N/A             N/A
256        False    jvp_attn   0.50 ms         0.48 ms (0.97x) 0.53 ms (1.06x)
256        True     jvp_attn   0.50 ms         N/A             N/A
512        False    jvp_attn   0.48 ms         0.55 ms (1.16x) 0.52 ms (1.10x)
512        True     jvp_attn   0.50 ms         N/A             N/A
1024       False    jvp_attn   0.58 ms         0.66 ms (1.14x) 0.73 ms (1.26x)
1024       True     jvp_attn   0.80 ms         N/A             N/A
2048       False    jvp_attn   1.93 ms         1.61 ms (0.83x) 2.14 ms (1.11x)
2048       True     jvp_attn   1.31 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 3.08x
Min speedup: 0.95x
Max speedup: 9.03x

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
  version = {0.0.2},
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
