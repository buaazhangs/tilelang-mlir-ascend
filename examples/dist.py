import argparse
import os

import torch
import tilelang
import tilelang.language as T


def cdist_squared_pytorch(x1, x2):
    diff = x1.to(torch.float32).unsqueeze(1) - x2.to(torch.float32).unsqueeze(0)
    return torch.sum(diff * diff, dim=-1).squeeze(0)


def ref_program(A, B):
    return cdist_squared_pytorch(A, B)


def supply_prog(args):
    dtype = torch.bfloat16
    # NPU
    query_hidden_states = torch.ones(1, 256, dtype=dtype, device="npu")
    c_embs = torch.ones(2000000, 256, dtype=dtype, device="npu")
    return [query_hidden_states, c_embs]


def manual_check_prog(actual, expected):
    actual_tensor = actual[0] if isinstance(actual, (list, tuple)) else actual
    expected_tensor = expected[0] if isinstance(expected, (list, tuple)) else expected
    return torch.allclose(actual_tensor, expected_tensor, rtol=1e-2, atol=1e-2)


def get_configs():
    return [
        {"block_M": block_M, "block_N": block_N}
        for block_M in [16, 32, 64]
        for block_N in [128, 256]
    ]


# NPU
@tilelang.jit(out_idx=[-1], target="npuir")
def distance(M, N, block_N=256, block_M=32, dtype="bfloat16", accum_dtype="float32"):
    """Developer 模式 NPU kernel：计算 query 与 codebook 每一行的平方欧氏距离。

    输入:
      A: [1, N]，单条 query 向量
      B: [M, N]，M 条待比较向量
    输出:
      C: [M]，C[m] = sum_j((A[0, j] - B[m, j]) ** 2)
    """

    @T.prim_func
    def dist(
        A: T.Tensor((1, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (cid, _):
            row_offset = cid * block_M
            real_m = T.min(block_M, M - row_offset) #尾块

            A_shared = T.alloc_shared((1, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            # 改动点：输入可以是 bf16/fp16，但进入向量计算前统一转成 fp32。
            # 这样差值、平方、block_N 归约都走 fp32 逻辑，语义更接近 PyTorch 参考实现。
            A_f32 = T.alloc_shared((1, block_N), accum_dtype)
            B_f32 = T.alloc_shared((block_M, block_N), accum_dtype)
            diff = T.alloc_shared((block_M, block_N), accum_dtype)
            diff_sq = T.alloc_shared((block_M, block_N), accum_dtype)
            partial_sum = T.alloc_shared((block_M, 1), accum_dtype)
            total_sum = T.alloc_shared((block_M, 1), accum_dtype)

            T.clear(total_sum)

            # 不在逻辑核划分里面分blockN，避免跨逻辑核
            for n_tile in T.serial(T.ceildiv(N, block_N)):
                col_offset = n_tile * block_N
                real_n = T.min(block_N, N - col_offset) #尾块

                # 改动点：显式向量算子会处理完整 tile，先清零避免 M/N 尾块的脏数据参与计算。
                T.clear(A_shared)
                T.clear(B_shared)

                # 搬运当前 query/codebook tile 到 shared，和 dev 示例里的 GM->shared 一致。
                T.copy(
                    A[0:1, col_offset : col_offset + real_n],
                    A_shared[0:1, 0:real_n],
                )
                T.copy(
                    B[
                        row_offset : row_offset + real_m,
                        col_offset : col_offset + real_n,
                    ],
                    B_shared[0:real_m, 0:real_n],
                )


                T.vcast(A_shared, A_f32, round_mode="rint")
                T.vcast(B_shared, B_f32, round_mode="rint")
                T.vsub(A_f32, B_f32, diff)
                T.vmul(diff, diff, diff_sq)

                T.reduce_sum(diff_sq, partial_sum, dim=1) #npu dev模式，这里当前reduce维度不能变化
                T.vadd(total_sum, partial_sum, total_sum)

            for local_m in T.Parallel(block_M):
                if local_m < real_m:
                    C[row_offset + local_m] = total_sum[local_m, 0]

    return dist


def run_test(M=1024, N=256, block_M=32, block_N=256, dtype="bfloat16"):
    os.environ["TILELANG_ASCEND_MODE"] = "Developer"
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()

    torch_dtype = getattr(torch, dtype)
    query = torch.randn((1, N), dtype=torch_dtype, device="npu")
    codebook = torch.randn((M, N), dtype=torch_dtype, device="npu")

    kernel = distance(
        M,
        N,
        block_N=block_N,
        block_M=block_M,
        dtype=dtype,
        accum_dtype="float32",
    )
    output = kernel(query, codebook)
    expected = ref_program(query, codebook)

    print("NPU output:")
    print(output)
    print("Reference:")
    print(expected)
    print(output.shape)
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-2, atol=1e-2)
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CDist squared L2 NPU Developer kernel")
    parser.add_argument("--M", type=int, default=993257)
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--block_M", type=int, default=64)
    parser.add_argument("--block_N", type=int, default=256)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
    )
    args = parser.parse_args()
    run_test(args.M, args.N, args.block_M, args.block_N, args.dtype)
