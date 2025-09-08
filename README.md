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
      # attn_mask=attn_mask,  # NOTE: Attention masking is temporarily unsupported
      # dropout_p=attn_dropout_p if self.training else 0.0,  # NOTE: Attention dropout is currently unsupported
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
32         False    additive   sdpa       0.785        0.64           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.475        0.23           0.0 TFLOP/s 1.95e-03     ✓

32         False    boolean    sdpa       0.834        0.65           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.469        0.22           0.0 TFLOP/s 1.95e-03     ✓

32         False    none       sdpa       0.546        0.64           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.465        0.22           0.0 TFLOP/s 1.95e-03     ✓

32         True     none       sdpa       0.838        0.65           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.464        0.22           0.0 TFLOP/s 1.95e-03     ✓

64         False    additive   sdpa       0.790        1.41           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.487        0.47           0.0 TFLOP/s 9.77e-04     ✓

64         False    boolean    sdpa       0.820        1.45           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.473        0.43           0.0 TFLOP/s 9.77e-04     ✓

64         False    none       sdpa       0.685        1.41           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.474        0.43           0.0 TFLOP/s 9.77e-04     ✓

64         True     none       sdpa       0.854        1.42           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.499        0.43           0.0 TFLOP/s 1.95e-03     ✓

128        False    additive   sdpa       0.769        3.28           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.494        1.02           0.1 TFLOP/s 9.77e-04     ✓

128        False    boolean    sdpa       0.831        3.44           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.468        0.86           0.1 TFLOP/s 9.77e-04     ✓

128        False    none       sdpa       0.533        3.28           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.470        0.86           0.1 TFLOP/s 9.77e-04     ✓

128        True     none       sdpa       0.984        3.35           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.483        0.86           0.0 TFLOP/s 1.95e-03     ✓

256        False    additive   sdpa       1.142        9.69           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.473        2.35           0.4 TFLOP/s 9.77e-04     ✓

256        False    boolean    sdpa       0.886        10.32          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.466        1.72           0.4 TFLOP/s 9.77e-04     ✓

256        False    none       sdpa       0.715        9.69           0.1 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.472        1.72           0.4 TFLOP/s 9.77e-04     ✓

256        True     none       sdpa       0.976        9.94           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.464        1.72           0.2 TFLOP/s 1.95e-03     ✓

512        False    additive   sdpa       1.399        31.88          0.2 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.481        5.95           1.4 TFLOP/s 4.88e-04     ✓

512        False    boolean    sdpa       1.222        34.38          0.3 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.489        3.45           1.4 TFLOP/s 4.88e-04     ✓

512        False    none       sdpa       1.106        31.88          0.3 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.475        3.45           1.4 TFLOP/s 4.88e-04     ✓

512        True     none       sdpa       1.354        32.88          0.1 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.493        3.45           0.7 TFLOP/s 1.95e-03     ✓

1024       False    additive   sdpa       2.430        113.77         0.6 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.480        16.89          5.7 TFLOP/s 4.88e-04     ✓

1024       False    boolean    sdpa       2.889        123.77         0.5 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.483        6.89           5.7 TFLOP/s 4.88e-04     ✓

1024       False    none       sdpa       2.457        113.77         0.6 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.467        6.89           5.9 TFLOP/s 4.88e-04     ✓

1024       True     none       sdpa       2.670        117.77         0.3 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.500        6.89           2.7 TFLOP/s 1.95e-03     ✓

2048       False    additive   sdpa       7.791        427.54         0.7 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   0.696        53.79         15.7 TFLOP/s 2.44e-04     ✓

2048       False    boolean    sdpa       7.673        467.54         0.7 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   0.755        13.79         14.5 TFLOP/s 2.44e-04     ✓

2048       False    none       sdpa       7.773        427.54         0.7 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   0.614        13.79         17.8 TFLOP/s 2.44e-04     ✓

2048       True     none       sdpa       8.609        443.54         0.3 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.464        13.79         11.8 TFLOP/s 1.95e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.47 ms         0.47 ms (1.01x) 0.48 ms (1.02x)
32         True     jvp_attn   0.46 ms         N/A             N/A
64         False    jvp_attn   0.47 ms         0.47 ms (1.00x) 0.49 ms (1.03x)
64         True     jvp_attn   0.50 ms         N/A             N/A
128        False    jvp_attn   0.47 ms         0.47 ms (1.00x) 0.49 ms (1.05x)
128        True     jvp_attn   0.48 ms         N/A             N/A
256        False    jvp_attn   0.47 ms         0.47 ms (0.99x) 0.47 ms (1.00x)
256        True     jvp_attn   0.46 ms         N/A             N/A
512        False    jvp_attn   0.47 ms         0.49 ms (1.03x) 0.48 ms (1.01x)
512        True     jvp_attn   0.49 ms         N/A             N/A
1024       False    jvp_attn   0.47 ms         0.48 ms (1.03x) 0.48 ms (1.03x)
1024       True     jvp_attn   0.50 ms         N/A             N/A
2048       False    jvp_attn   0.61 ms         0.75 ms (1.23x) 0.70 ms (1.13x)
2048       True     jvp_attn   0.46 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 4.00x
Min speedup: 1.13x
Max speedup: 18.54x

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
32         False    additive   sdpa       0.900        0.64           0.0 TFLOP/s baseline     N/A
32         False    additive   jvp_attn   0.661        0.23           0.0 TFLOP/s 1.56e-02     ✓

32         False    boolean    sdpa       0.883        0.65           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.578        0.22           0.0 TFLOP/s 1.56e-02     ✓

32         False    none       sdpa       0.603        0.64           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.513        0.22           0.0 TFLOP/s 1.56e-02     ✓

32         True     none       sdpa       0.915        0.65           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.519        0.22           0.0 TFLOP/s 1.56e-02     ✓

64         False    additive   sdpa       0.833        1.41           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.531        0.47           0.0 TFLOP/s 7.81e-03     ✓

64         False    boolean    sdpa       0.870        1.45           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.545        0.43           0.0 TFLOP/s 7.81e-03     ✓

64         False    none       sdpa       0.565        1.41           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.508        0.43           0.0 TFLOP/s 7.81e-03     ✓

64         True     none       sdpa       0.939        1.42           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.520        0.43           0.0 TFLOP/s 1.56e-02     ✓

128        False    additive   sdpa       0.864        3.28           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.476        1.02           0.1 TFLOP/s 7.81e-03     ✓

128        False    boolean    sdpa       0.839        3.44           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.460        0.86           0.1 TFLOP/s 7.81e-03     ✓

128        False    none       sdpa       0.798        3.28           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.519        0.86           0.1 TFLOP/s 7.81e-03     ✓

128        True     none       sdpa       0.886        3.35           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.504        0.86           0.0 TFLOP/s 1.56e-02     ✓

256        False    additive   sdpa       1.164        9.69           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.471        2.35           0.4 TFLOP/s 7.81e-03     ✓

256        False    boolean    sdpa       0.918        10.32          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.468        1.72           0.4 TFLOP/s 7.81e-03     ✓

256        False    none       sdpa       0.780        9.69           0.1 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.463        1.72           0.4 TFLOP/s 3.91e-03     ✓

256        True     none       sdpa       1.187        9.94           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.463        1.72           0.2 TFLOP/s 1.56e-02     ✓

512        False    additive   sdpa       1.059        31.88          0.3 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.476        5.95           1.4 TFLOP/s 3.91e-03     ✓

512        False    boolean    sdpa       1.053        34.38          0.3 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.463        3.45           1.5 TFLOP/s 3.91e-03     ✓

512        False    none       sdpa       1.131        31.88          0.3 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.466        3.45           1.5 TFLOP/s 3.91e-03     ✓

512        True     none       sdpa       1.651        32.88          0.1 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.470        3.45           0.7 TFLOP/s 1.56e-02     ✓

1024       False    additive   sdpa       2.525        113.77         0.5 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.489        16.89          5.6 TFLOP/s 3.91e-03     ✓

1024       False    boolean    sdpa       2.775        123.77         0.5 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.471        6.89           5.8 TFLOP/s 3.91e-03     ✓

1024       False    none       sdpa       2.393        113.77         0.6 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.482        6.89           5.7 TFLOP/s 3.91e-03     ✓

1024       True     none       sdpa       2.319        117.77         0.3 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.463        6.89           3.0 TFLOP/s 1.56e-02     ✓

2048       False    additive   sdpa       8.504        427.54         0.6 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   0.804        53.79         13.6 TFLOP/s 1.95e-03     ✓

2048       False    boolean    sdpa       8.944        467.54         0.6 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   0.852        13.79         12.9 TFLOP/s 1.95e-03     ✓

2048       False    none       sdpa       6.810        427.54         0.8 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   0.900        13.79         12.2 TFLOP/s 1.95e-03     ✓

2048       True     none       sdpa       8.846        443.54         0.3 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.508        13.79         10.8 TFLOP/s 3.12e-02     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.51 ms         0.58 ms (1.13x) 0.66 ms (1.29x)
32         True     jvp_attn   0.52 ms         N/A             N/A
64         False    jvp_attn   0.51 ms         0.54 ms (1.07x) 0.53 ms (1.05x)
64         True     jvp_attn   0.52 ms         N/A             N/A
128        False    jvp_attn   0.52 ms         0.46 ms (0.89x) 0.48 ms (0.92x)
128        True     jvp_attn   0.50 ms         N/A             N/A
256        False    jvp_attn   0.46 ms         0.47 ms (1.01x) 0.47 ms (1.02x)
256        True     jvp_attn   0.46 ms         N/A             N/A
512        False    jvp_attn   0.47 ms         0.46 ms (0.99x) 0.48 ms (1.02x)
512        True     jvp_attn   0.47 ms         N/A             N/A
1024       False    jvp_attn   0.48 ms         0.47 ms (0.98x) 0.49 ms (1.01x)
1024       True     jvp_attn   0.46 ms         N/A             N/A
2048       False    jvp_attn   0.90 ms         0.85 ms (0.95x) 0.80 ms (0.89x)
2048       True     jvp_attn   0.51 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 3.75x
Min speedup: 1.11x
Max speedup: 17.40x

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
32         False    additive   jvp_attn   0.523        0.45           0.0 TFLOP/s 7.21e-03     ✓

32         False    boolean    sdpa       0.764        0.53           0.0 TFLOP/s baseline     N/A
32         False    boolean    jvp_attn   0.500        0.43           0.0 TFLOP/s 7.21e-03     ✓

32         False    none       sdpa       0.454        0.51           0.0 TFLOP/s baseline     N/A
32         False    none       jvp_attn   0.521        0.43           0.0 TFLOP/s 7.22e-03     ✓

32         True     none       sdpa       0.771        0.51           0.0 TFLOP/s baseline     N/A
32         True     none       jvp_attn   0.530        0.43           0.0 TFLOP/s 6.18e-03     ✓

64         False    additive   sdpa       0.731        1.09           0.0 TFLOP/s baseline     N/A
64         False    additive   jvp_attn   0.503        0.94           0.0 TFLOP/s 7.17e-03     ✓

64         False    boolean    sdpa       0.760        1.17           0.0 TFLOP/s baseline     N/A
64         False    boolean    jvp_attn   0.501        0.86           0.0 TFLOP/s 7.17e-03     ✓

64         False    none       sdpa       0.447        1.09           0.0 TFLOP/s baseline     N/A
64         False    none       jvp_attn   0.497        0.86           0.0 TFLOP/s 7.03e-03     ✓

64         True     none       sdpa       0.790        1.11           0.0 TFLOP/s baseline     N/A
64         True     none       jvp_attn   0.507        0.86           0.0 TFLOP/s 6.18e-03     ✓

128        False    additive   sdpa       0.702        2.81           0.0 TFLOP/s baseline     N/A
128        False    additive   jvp_attn   0.494        2.03           0.1 TFLOP/s 5.41e-03     ✓

128        False    boolean    sdpa       0.826        3.13           0.0 TFLOP/s baseline     N/A
128        False    boolean    jvp_attn   0.478        1.72           0.1 TFLOP/s 5.41e-03     ✓

128        False    none       sdpa       0.579        2.81           0.0 TFLOP/s baseline     N/A
128        False    none       jvp_attn   0.514        1.72           0.1 TFLOP/s 5.07e-03     ✓

128        True     none       sdpa       0.837        2.88           0.0 TFLOP/s baseline     N/A
128        True     none       jvp_attn   0.537        1.72           0.0 TFLOP/s 6.18e-03     ✓

256        False    additive   sdpa       0.687        8.75           0.1 TFLOP/s baseline     N/A
256        False    additive   jvp_attn   0.481        4.69           0.4 TFLOP/s 3.41e-03     ✓

256        False    boolean    sdpa       0.797        10.00          0.1 TFLOP/s baseline     N/A
256        False    boolean    jvp_attn   0.506        3.44           0.3 TFLOP/s 3.41e-03     ✓

256        False    none       sdpa       0.466        8.75           0.2 TFLOP/s baseline     N/A
256        False    none       jvp_attn   0.474        3.44           0.4 TFLOP/s 3.67e-03     ✓

256        True     none       sdpa       1.024        9.00           0.0 TFLOP/s baseline     N/A
256        True     none       jvp_attn   0.496        3.44           0.2 TFLOP/s 5.78e-03     ✓

512        False    additive   sdpa       0.982        30.01          0.3 TFLOP/s baseline     N/A
512        False    additive   jvp_attn   0.515        11.88          1.3 TFLOP/s 3.09e-03     ✓

512        False    boolean    sdpa       1.413        35.01          0.2 TFLOP/s baseline     N/A
512        False    boolean    jvp_attn   0.487        6.88           1.4 TFLOP/s 3.09e-03     ✓

512        False    none       sdpa       1.362        30.01          0.3 TFLOP/s baseline     N/A
512        False    none       jvp_attn   0.481        6.88           1.4 TFLOP/s 2.88e-03     ✓

512        True     none       sdpa       0.968        31.01          0.2 TFLOP/s baseline     N/A
512        True     none       jvp_attn   0.481        6.88           0.7 TFLOP/s 5.13e-03     ✓

1024       False    additive   sdpa       2.381        110.02         0.6 TFLOP/s baseline     N/A
1024       False    additive   jvp_attn   0.708        33.77          3.9 TFLOP/s 2.84e-03     ✓

1024       False    boolean    sdpa       3.475        130.02         0.4 TFLOP/s baseline     N/A
1024       False    boolean    jvp_attn   0.593        13.77          4.6 TFLOP/s 2.84e-03     ✓

1024       False    none       sdpa       2.246        110.02         0.6 TFLOP/s baseline     N/A
1024       False    none       jvp_attn   0.488        13.77          5.6 TFLOP/s 2.61e-03     ✓

1024       True     none       sdpa       2.700        115.02         0.3 TFLOP/s baseline     N/A
1024       True     none       jvp_attn   0.531        13.77          2.6 TFLOP/s 5.61e-03     ✓

2048       False    additive   sdpa       8.094        420.04         0.7 TFLOP/s baseline     N/A
2048       False    additive   jvp_attn   1.274        107.54         8.6 TFLOP/s 1.57e-03     ✓

2048       False    boolean    sdpa       7.980        500.04         0.7 TFLOP/s baseline     N/A
2048       False    boolean    jvp_attn   1.332        27.54          8.2 TFLOP/s 1.57e-03     ✓

2048       False    none       sdpa       6.471        420.04         0.8 TFLOP/s baseline     N/A
2048       False    none       jvp_attn   1.345        27.54          8.1 TFLOP/s 1.56e-03     ✓

2048       True     none       sdpa       8.308        436.04         0.3 TFLOP/s baseline     N/A
2048       True     none       jvp_attn   0.749        27.54          7.3 TFLOP/s 6.47e-03     ✓


================================================================================
MASK TYPE PERFORMANCE COMPARISON
================================================================================
Seq Len    Causal   Method     No Mask         Boolean Mask    Additive Mask
--------------------------------------------------------------------------------
32         False    jvp_attn   0.52 ms         0.50 ms (0.96x) 0.52 ms (1.00x)
32         True     jvp_attn   0.53 ms         N/A             N/A
64         False    jvp_attn   0.50 ms         0.50 ms (1.01x) 0.50 ms (1.01x)
64         True     jvp_attn   0.51 ms         N/A             N/A
128        False    jvp_attn   0.51 ms         0.48 ms (0.93x) 0.49 ms (0.96x)
128        True     jvp_attn   0.54 ms         N/A             N/A
256        False    jvp_attn   0.47 ms         0.51 ms (1.07x) 0.48 ms (1.01x)
256        True     jvp_attn   0.50 ms         N/A             N/A
512        False    jvp_attn   0.48 ms         0.49 ms (1.01x) 0.52 ms (1.07x)
512        True     jvp_attn   0.48 ms         N/A             N/A
1024       False    jvp_attn   0.49 ms         0.59 ms (1.22x) 0.71 ms (1.45x)
1024       True     jvp_attn   0.53 ms         N/A             N/A
2048       False    jvp_attn   1.34 ms         1.33 ms (0.99x) 1.27 ms (0.95x)
2048       True     jvp_attn   0.75 ms         N/A             N/A

============================================================
STATISTICS
============================================================
Average speedup: 2.83x
Min speedup: 0.87x
Max speedup: 11.09x

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
  version = {0.0.3},
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
