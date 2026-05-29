"""ADC 2D LUT indirect-load kernel extracted from adc_tilelang_npu.py.

Input layout matches the ADC kernel:
  LUT:   [S, K] float32
  codes: [N, S] int32
  out:   [S, N] float32

The kernel keeps the same N-block partitioning and code staging path as
adc_tilelang_npu.py, then isolates only the 2D LUT access:
  LUT_VAL_UB[m] = LUT[s, CODES_UB[s, m]]
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
os.environ.setdefault("TILELANG_ENABLE_SIMT", "1")

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


@tilelang.jit(out_idx=[-1], target="npuir")
def adc_lut2d_indirect_kernel(block_M, num_subspaces=64,
                              codebook_size=256, dtype="float32",
                              code_dtype="int32"):
    N = T.symbolic("N")

    @T.prim_func
    def lut2d_func(
        LUT: T.Tensor((num_subspaces, codebook_size), dtype),
        codes: T.Tensor((N, num_subspaces), code_dtype),
        out: T.Tensor((num_subspaces, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_M), is_npu=True) as (by, _):
            start = by * block_M
            valid = T.min(block_M, N - start)

            CODES_TILE_UB = T.alloc_ub((block_M, num_subspaces), code_dtype)
            CODES_UB = T.alloc_ub((num_subspaces, block_M), code_dtype)
            LUT_VAL_UB = T.alloc_ub((block_M,), dtype)

            value_zero = 0
            T.npuir_brc(value_zero, LUT_VAL_UB)

            T.copy(
                codes[start:start + valid, 0:num_subspaces],
                CODES_TILE_UB[0:valid, 0:num_subspaces],
            )
            T.npuir_transpose(CODES_TILE_UB, CODES_UB, [1, 0])

            for s in T.serial(num_subspaces):
                for m in T.Parallel(block_M):
                    if m < valid:
                        LUT_VAL_UB[m] = LUT[s, CODES_UB[s, m]]

                T.copy(LUT_VAL_UB[0:valid], out[s, start:start + valid])

    return lut2d_func


def make_lut2d_kernel(block_M=256):
    return adc_lut2d_indirect_kernel(block_M)


def main(n, block_M, num_subspaces, codebook_size):
    if n <= 0 or block_M <= 0:
        raise ValueError("n and block_M must be positive")

    torch.manual_seed(42)
    # torch.npu.set_device(0)

    lut_cpu = torch.rand(
        num_subspaces, codebook_size, dtype=torch.float32)
    codes_cpu = torch.randint(
        0, codebook_size, (n, num_subspaces), dtype=torch.int32)
    assert codes_cpu.shape == (n, num_subspaces)

    sub_idx = torch.arange(num_subspaces, dtype=torch.long).unsqueeze(0)
    partial = lut_cpu[sub_idx, codes_cpu.long()]
    ref = partial.transpose(0, 1).contiguous()

    lut = lut_cpu.to("npu")
    codes = codes_cpu.to("npu")

    kernel = make_lut2d_kernel(block_M)
    result = kernel(lut, codes)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), ref, rtol=1e-3, atol=1e-3)
    print(f"PASS n={n} block_M={block_M}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=env_int("TILELANG_ADC_N", 77300*2))
    parser.add_argument("--block-m", type=int,
                        default=env_int("TILELANG_ADC_BLOCK_M", 256))
    parser.add_argument("--num-subspaces", type=int,
                        default=env_int("TILELANG_ADC_S", 64))
    parser.add_argument("--codebook-size", type=int,
                        default=env_int("TILELANG_ADC_K", 256))
    args = parser.parse_args()
    main(args.n, args.block_m, args.num_subspaces, args.codebook_size)
