"""ADC distance kernel with uint8 codes cast before transpose.

Input layout:
  LUT:     [S, K] float32
  codes:   [N, S] uint8
  out:     [N] float32

This variant keeps the public input contract of adc_tilelang_npu_ub_gather_u8.py
but avoids the u8-transpose-then-cast lowering path. Each subspace tile is
first cast from uint8 to int32 in its original [block_M, S_TILE] layout, then the
int32 tile is transposed to [S_TILE, block_M] for contiguous gather indices.
"""

import argparse
import os

os.environ.setdefault("TILELANG_ASCEND_MODE", "Developer")
os.environ.setdefault("TILELANG_ENABLE_SIMT", "0")

import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.language as T


def env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


@tilelang.jit(out_idx=[-1], target="npuir")
def adc_distance_ub_gather_u8_cast_before_transpose_kernel(
    block_M,
    num_subspaces=64,
    codebook_size=256,
    dtype="float32",
    code_dtype="uint8",
    num_kernels=56,
    subspace_tile=16,
):
    if code_dtype != "uint8":
        raise ValueError("this variant expects uint8 codes")
    if codebook_size > 256:
        raise ValueError("uint8 codes require codebook_size <= 256")
    if subspace_tile <= 0:
        raise ValueError("subspace_tile must be positive")
    if num_subspaces % subspace_tile != 0:
        raise ValueError("num_subspaces must be divisible by subspace_tile")

    N = T.symbolic("N")
    num_logic_kernels = T.ceildiv(N, block_M)
    num_subspace_tiles = num_subspaces // subspace_tile

    @T.prim_func
    def adc_func(
        LUT: T.Tensor((num_subspaces, codebook_size), dtype),
        codes: T.Tensor((N, num_subspaces), code_dtype),
        out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(num_kernels, is_npu=True) as (kernel_id, _):
            LUT_UB = T.alloc_ub((num_subspaces, codebook_size), dtype)
            CODES_TILE_U8_UB = T.alloc_ub((block_M, subspace_tile), code_dtype)
            CODES_TILE_I32_UB = T.alloc_ub((block_M, subspace_tile), "int32")
            CODES_I32_UB = T.alloc_ub((subspace_tile, block_M), "int32")
            LUT_ROW_UB = T.alloc_ub((1, block_M), dtype)
            acc = T.alloc_ub((1, block_M), dtype)
            out_ub = T.alloc_ub((1, block_M), dtype)

            value_zero = 0
            T.copy(LUT, LUT_UB)

            for task_id in T.serial(T.ceildiv(num_logic_kernels, num_kernels)):
                logic_kernel_id = task_id * num_kernels
                logic_kernel_id = logic_kernel_id + kernel_id
                if logic_kernel_id < num_logic_kernels:
                    start = logic_kernel_id * block_M
                    valid = T.min(block_M, N - start)

                    T.vbrc(value_zero, CODES_TILE_U8_UB)
                    T.vbrc(value_zero, acc)

                    for sg in T.serial(num_subspace_tiles):
                        s_base = sg * subspace_tile
                        T.copy(
                            codes[start:start + valid,
                                  s_base:s_base + subspace_tile],
                            CODES_TILE_U8_UB[0:valid, 0:subspace_tile],
                        )
                        T.vcast(CODES_TILE_U8_UB, CODES_TILE_I32_UB)
                        T.transpose(CODES_TILE_I32_UB, CODES_I32_UB, [1, 0])

                        for ss in T.serial(subspace_tile):
                            s = s_base + ss
                            T.gather(
                                LUT_UB[s:s + 1, 0:codebook_size],
                                LUT_ROW_UB,
                                CODES_I32_UB[ss:ss + 1, 0:block_M],
                            )
                            T.vadd(acc, LUT_ROW_UB, acc)

                    T.vsqrt(acc, out_ub)
                    T.copy(out_ub[0, 0:valid], out[start:start + valid])

    return adc_func


def make_adc_kernel(block_M=256, num_subspaces=64,
                    codebook_size=256, num_kernels=56, subspace_tile=16):
    return adc_distance_ub_gather_u8_cast_before_transpose_kernel(
        block_M,
        num_subspaces=num_subspaces,
        codebook_size=codebook_size,
        num_kernels=num_kernels,
        subspace_tile=subspace_tile,
    )


def main(n, block_M, num_subspaces, codebook_size, num_kernels, subspace_tile):
    if n <= 0 or block_M <= 0:
        raise ValueError("n and block_M must be positive")
    if num_kernels <= 0:
        raise ValueError("num_kernels must be positive")
    if codebook_size > 256:
        raise ValueError("uint8 codes require codebook_size <= 256")
    if subspace_tile <= 0:
        raise ValueError("subspace_tile must be positive")
    if num_subspaces % subspace_tile != 0:
        raise ValueError("num_subspaces must be divisible by subspace_tile")

    torch.manual_seed(42)
    # torch.npu.set_device(0)

    lut_cpu = torch.rand(num_subspaces, codebook_size, dtype=torch.float32)
    codes_cpu = torch.randint(
        0, codebook_size, (n, num_subspaces), dtype=torch.uint8)
    assert codes_cpu.shape == (n, num_subspaces)
    sub_idx = torch.arange(num_subspaces, dtype=torch.long).unsqueeze(0)
    partial = lut_cpu[sub_idx, codes_cpu.long()]
    ref = torch.sqrt(partial.sum(dim=-1))

    lut = lut_cpu.to("npu")
    codes = codes_cpu.to("npu")

    kernel = make_adc_kernel(
        block_M,
        num_subspaces=num_subspaces,
        codebook_size=codebook_size,
        num_kernels=num_kernels,
        subspace_tile=subspace_tile,
    )
    result = kernel(lut, codes)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), ref, rtol=1e-3, atol=1e-3)
    print(
        f"PASS n={n} block_M={block_M} num_kernels={num_kernels} "
        f"subspace_tile={subspace_tile} code_dtype=uint8 "
        "cast_before_transpose"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=env_int("TILELANG_ADC_N", 77300 * 2))
    parser.add_argument("--block-m", type=int,
                        default=env_int("TILELANG_ADC_BLOCK_M", 256))
    parser.add_argument("--num-subspaces", type=int,
                        default=env_int("TILELANG_ADC_S", 64))
    parser.add_argument("--codebook-size", type=int,
                        default=env_int("TILELANG_ADC_K", 256))
    parser.add_argument("--num-kernels", type=int,
                        default=env_int("TILELANG_ADC_NUM_KERNELS", 56))
    parser.add_argument("--subspace-tile", type=int,
                        default=env_int("TILELANG_ADC_S_TILE", 16))
    args = parser.parse_args()
    main(args.n, args.block_m, args.num_subspaces, args.codebook_size,
         args.num_kernels, args.subspace_tile)
