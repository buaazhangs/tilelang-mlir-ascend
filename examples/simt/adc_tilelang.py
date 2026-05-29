"""
ADC (Asymmetric Distance Computation) TileLang Kernel

LUT: [S, K] float32, S=num_subspaces=64, K=codebook_size=256
codes: [N, S] uint8, N=num_docs
out: [N] float32 (L2 distance)

每个 block 处理 block_M 个文档:
  1. 将 LUT 加载到 shared memory (64*256*4B = 64KB)
  2. 流式读取 codes, 查 LUT 累加, 写出 sqrt(sum)
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def adc_distance_kernel(block_M, num_threads, num_subspaces=64, codebook_size=256,
                        dtype="float32", code_dtype="int32"):
    N = T.dynamic("N")

    @T.prim_func
    def adc_func(
        LUT: T.Tensor((num_subspaces, codebook_size), dtype),
        codes: T.Tensor((N, num_subspaces), code_dtype),
        out: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_M), threads=num_threads) as (by,):
            # 将整个 LUT 加载到 shared memory: 64*256*4B = 64KB
            LUT_shared = T.alloc_shared((num_subspaces, codebook_size), dtype)
            T.copy(LUT[0, 0], LUT_shared)

            acc = T.alloc_fragment((block_M,), dtype)
            T.clear(acc)

            # 对每个子空间查表累加
            for s in T.serial(num_subspaces):
                for m in T.Parallel(block_M):
                    global_m = by * block_M + m
                    if global_m < N:
                        code_val = codes[global_m, s]
                        acc[m] = acc[m] + LUT_shared[s, code_val]

            # sqrt 并写出
            for m in T.Parallel(block_M):
                global_m = by * block_M + m
                if global_m < N:
                    out[global_m] = T.sqrt(acc[m])

    return adc_func


def make_adc_kernel(block_M=256, num_threads=128):
    """创建 ADC kernel 实例"""
    return adc_distance_kernel(block_M, num_threads)


if __name__ == "__main__":
    # 正确性验证
    S, K = 64, 256
    N = 200000

    torch.manual_seed(42)
    lut = torch.randn(S, K, dtype=torch.float32, device="cuda")
    codes = torch.randint(0, K, (N, S), dtype=torch.int32, device="cuda")

    # PyTorch 参考
    codes_long = codes.to(dtype=torch.long)
    sub_idx = torch.arange(S, device="cuda")
    partial = lut[sub_idx.unsqueeze(0), codes_long]  # [N, S]
    ref = torch.sqrt(partial.sum(dim=-1))  # [N]

    # TileLang kernel
    kernel = make_adc_kernel(256, 128)
    result = kernel(lut, codes)

    if torch.allclose(result, ref, rtol=1e-3, atol=1e-3):
        print(f"✅ ADC kernel correct! N={N}, max_diff={torch.abs(result - ref).max().item():.6f}")
    else:
        print(f"❌ ADC kernel mismatch! max_diff={torch.abs(result - ref).max().item():.6f}")
        diff_idx = torch.argmax(torch.abs(result - ref))
        print(f"   result[{diff_idx}]={result[diff_idx].item()}, ref[{diff_idx}]={ref[diff_idx].item()}")
