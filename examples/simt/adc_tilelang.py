"""ADC distance kernel using split SIMD/SIMT loops for NPUIR.

Input layout:
  LUT:     [S, K] float32
  codes_t: [S, N] int32, transposed from the usual [N, S] layout
  out:     [N] float32

Each NPU program handles one contiguous block of documents.  For every
subspace, the kernel separates the original mixed loop into:
  1. continuous code load into UB by T.copy/MTE2;
  2. 2D LUT indirect load, expected to be rewritten by NpuSimtIndirectLoad;
  3. vector accumulation by explicit NPUIR vector add.
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
def adc_distance_kernel(block_M, num_threads, num_subspaces=64,
                        codebook_size=256, dtype="float32",
                        code_dtype="int32"):
    del num_threads
    N = T.symbolic("N")

    @T.prim_func
    def adc_func(
        LUT: T.Tensor((num_subspaces, codebook_size), dtype),
        codes_t: T.Tensor((num_subspaces, N), code_dtype),
        out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_M), is_npu=True) as (by, _):
            start = by * block_M
            valid = T.min(block_M, N - start)

            CODES_UB = T.alloc_ub((num_subspaces, block_M), code_dtype)
            LUT_VAL_UB = T.alloc_ub((block_M,), dtype)
            acc = T.alloc_ub((block_M,), dtype)
            out_ub = T.alloc_ub((block_M,), dtype)

            value_zero = 0
            T.npuir_brc(value_zero, acc)
            T.npuir_brc(value_zero, LUT_VAL_UB)

            for s in T.serial(num_subspaces):
                T.copy(codes_t[s, start:start + valid], CODES_UB[s, 0:valid])

                for m in T.Parallel(block_M):
                    if m < valid:
                        LUT_VAL_UB[m] = LUT[s, CODES_UB[s, m]]

                T.npuir_add(acc, LUT_VAL_UB, acc)

            T.vsqrt(acc, out_ub)
            T.copy(out_ub[0:valid], out[start:start + valid])

    return adc_func


def make_adc_kernel(block_M=256, num_threads=128):
    return adc_distance_kernel(block_M, num_threads)


def main(n, block_M, num_subspaces, codebook_size):
    if n <= 0 or block_M <= 0:
        raise ValueError("n and block_M must be positive")

    torch.manual_seed(42)
    torch.npu.set_device(0)

    lut = torch.randn(
        num_subspaces, codebook_size, dtype=torch.float32, device="npu")
    codes_t = torch.randint(
        0, codebook_size, (num_subspaces, n), dtype=torch.int32, device="npu")

    kernel = make_adc_kernel(block_M, 128)
    result = kernel(lut, codes_t)

    lut_cpu = lut.cpu()
    codes_cpu = codes_t.cpu().long()
    partial = torch.gather(lut_cpu, dim=1, index=codes_cpu)
    ref = torch.sqrt(partial.sum(dim=0))

    torch.testing.assert_close(result.cpu(), ref, rtol=1e-3, atol=1e-3)
    print(f"PASS n={n} block_M={block_M}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=env_int("TILELANG_ADC_N", 200000))
    parser.add_argument("--block-m", type=int,
                        default=env_int("TILELANG_ADC_BLOCK_M", 256))
    parser.add_argument("--num-subspaces", type=int,
                        default=env_int("TILELANG_ADC_S", 64))
    parser.add_argument("--codebook-size", type=int,
                        default=env_int("TILELANG_ADC_K", 256))
    args = parser.parse_args()
    main(args.n, args.block_m, args.num_subspaces, args.codebook_size)
