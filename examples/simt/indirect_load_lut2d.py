# Copyright (c) Huawei Technologies Co., Ltd. 2025.
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


def indirect_load_lut2d(n, block, num_subspaces, codebook_size):
    @T.prim_func
    def main(
        LUT: T.Tensor((num_subspaces, codebook_size), "float32"),
        CODES: T.Tensor((num_subspaces, block), "int32"),
        OUT: T.Tensor((num_subspaces, n), "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (pid, vid):
            CODES_UB = T.alloc_ub((num_subspaces, block), "int32")
            O_UB = T.alloc_ub((block,), "float32")

            T.copy(CODES[0, 0], CODES_UB)

            for s in T.serial(num_subspaces):
                for m in T.Parallel(block):
                    if m < n:
                        O_UB[m] = LUT[s, CODES_UB[s, m]]
                T.copy(O_UB[0:n], OUT[s, 0:n])

    return main


def main(n, block, num_subspaces, codebook_size):
    if n <= 0 or block <= 0:
        raise ValueError("n and block must be positive")
    if n > block:
        raise ValueError("this minimal experiment uses one program, so n must be <= block")

    torch.manual_seed(0)
    torch.npu.set_device(0)

    lut = torch.randn(num_subspaces, codebook_size, device="npu", dtype=torch.float32)
    codes = torch.randint(
        0, codebook_size, (num_subspaces, block), device="npu", dtype=torch.int32)
    out = torch.empty(num_subspaces, n, device="npu", dtype=torch.float32)

    kernel = tilelang.compile(
        indirect_load_lut2d(n, block, num_subspaces, codebook_size),
        target="npuir",
    )
    kernel(lut, codes, out)

    subspace = torch.arange(num_subspaces, device="npu")[:, None]
    ref = lut[subspace, codes[:, :n].long()]
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=env_int("TILELANG_SIMT_N", 128))
    parser.add_argument("--block", type=int, default=env_int("TILELANG_SIMT_BLOCK", 256))
    parser.add_argument("--num-subspaces", type=int, default=env_int("TILELANG_SIMT_S", 4))
    parser.add_argument("--codebook-size", type=int, default=env_int("TILELANG_SIMT_K", 256))
    args = parser.parse_args()
    main(args.n, args.block, args.num_subspaces, args.codebook_size)
