2026-05-21 09:39:15  [TileLang:tilelang.env:WARNING]: CUTLASS is not installed or found in the expected path
[NpuSimtIndirectLoad] visit T.Parallel:
for m in T.parallel(256):
    if m < 128:
        O_UB = T.Buffer((256,), scope="shared")
        LUT_shared = T.Buffer((4, 256), scope="shared")
        s = T.int32()
        CODES = T.Buffer((4, 256), "int32")
        O_UB[m] = LUT_shared[s, CODES[s, m]]
[NpuSimtIndirectLoad] 1111:
O_UB = T.Buffer((256,), scope="shared")
LUT_shared = T.Buffer((4, 256), scope="shared")
s = T.int32()
CODES = T.Buffer((4, 256), "int32")
m = T.int32()
O_UB[m] = LUT_shared[s, CODES[s, m]]
[NpuSimtIndirectLoad] rewritten TIR:
LUT_shared = T.Buffer((4, 256), scope="shared")
s = T.int32()
CODES = T.Buffer((4, 256), "int32")
O_UB = T.Buffer((256,), scope="shared")
T.npuir_indirect_load(T.region(LUT_shared[s, 0], 1, 1, 256), T.region(CODES[s, 0], 1, 1, 256), T.region(O_UB[0], 2, 256), 128)
====== TVM IR ======
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(LUT: T.Buffer((4, 256), "float32"), CODES: T.Buffer((4, 256), "int32"), OUT: T.Buffer((4, 128), "float32")):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": [], "kind": "npuir", "tag": ""})})
        pid = T.launch_thread("blockIdx.x", 1)
        vid = T.launch_thread("blockIdx.y", 2)
        LUT_shared = T.decl_buffer((4, 256), scope="shared")
        T.copy(T.region(LUT[0, 0], 1, 4, 256), T.region(LUT_shared[0, 0], 2, 4, 256))
        for s in range(4):
            O_UB = T.decl_buffer((256,), scope="shared")
            T.npuir_indirect_load(T.region(LUT_shared[s, 0], 1, 1, 256), T.region(CODES[s, 0], 1, 1, 256), T.region(O_UB[0], 2, 256), 128)
            T.copy(T.region(O_UB[0], 1, 128), T.region(OUT[s, 0], 2, 1, 128))

====== npuir ======
#map = affine_map<(d0) -> (d0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func private @triton_indirect_load(memref<1024xf32>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
  func.func @main(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32>, %arg3: memref<?xi32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %c256_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [4, 256], strides: [%2, %0] : memref<?xf32> to memref<4x256xf32, strided<[256, 1]>>
    %c128_i32 = arith.constant 128 : i32
    %3 = arith.muli %c128_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 128], strides: [%4, %0] : memref<?xf32> to memref<4x128xf32, strided<[128, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 256], strides: [%2, %0] : memref<?xi32> to memref<4x256xi32, strided<[256, 1]>>
    %5 = hivm.hir.get_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = hivm.hir.get_sub_block_idx -> i64
    %8 = arith.trunci %7 : i64 to i32
    %9 = tensor.empty() : tensor<4x256xf32>
    %alloc = memref.alloc() : memref<4x256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x256xf32, strided<[256, 1]>> to memref<4x256xf32>
    %10 = bufferization.to_tensor %alloc restrict : memref<4x256xf32>
    %inserted_slice = tensor.insert_slice %10 into %9[0, 0] [4, 256] [1, 1] : tensor<4x256xf32> into tensor<4x256xf32>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32_2 = arith.constant 1 : i32
    scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32_2  : i32 {
      %11 = tensor.empty() : tensor<256xf32>
      %12 = bufferization.to_memref %inserted_slice : memref<4x256xf32>
      %collapse_shape = memref.collapse_shape %12 [[0, 1]] : memref<4x256xf32> into memref<1024xf32>
      %13 = arith.index_cast %arg11 : i32 to index
      %subview = memref.subview %reinterpret_cast_1[%13, 0] [1, 256] [1, 1] : memref<4x256xi32, strided<[256, 1]>> to memref<1x256xi32, strided<[256, 1], offset: ?>>
      %14 = bufferization.to_tensor %subview restrict : memref<1x256xi32, strided<[256, 1], offset: ?>>
      %collapsed = tensor.collapse_shape %14 [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
      %15 = arith.extsi %collapsed : tensor<256xi32> to tensor<256xi64>
      %c256_i32_3 = arith.constant 256 : i32
      %16 = arith.muli %arg11, %c256_i32_3 : i32
      %17 = arith.extsi %16 : i32 to i64
      %18 = tensor.empty() : tensor<256xi64>
      %19 = linalg.fill ins(%17 : i64) outs(%18 : tensor<256xi64>) -> tensor<256xi64>
      %20 = arith.addi %15, %19 : tensor<256xi64>
      %21 = tensor.empty() : tensor<256xi32>
      %22 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%21 : tensor<256xi32>) attrs =  {tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 256 : index} {
      ^bb0(%out: i32):
        %29 = linalg.index 0 : index
        %30 = arith.index_cast %29 : index to i32
        linalg.yield %30 : i32
      } -> tensor<256xi32>
      %c128_i32_4 = arith.constant 128 : i32
      %23 = tensor.empty() : tensor<256xi32>
      %24 = linalg.fill ins(%c128_i32_4 : i32) outs(%23 : tensor<256xi32>) -> tensor<256xi32>
      %25 = arith.cmpi slt, %22, %24 : tensor<256xi32>
      %cst = arith.constant 0.000000e+00 : f32
      %26 = tensor.empty() : tensor<256xf32>
      %27 = linalg.fill ins(%cst : f32) outs(%26 : tensor<256xf32>) -> tensor<256xf32>
      %28 = func.call @triton_indirect_load(%collapse_shape, %20, %25, %27) : (memref<1024xf32>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
      %extracted_slice = tensor.extract_slice %28[0] [128] [1] : tensor<256xf32> to tensor<128xf32>
      %subview_5 = memref.subview %reinterpret_cast_0[%13, 0] [1, 128] [1, 1] : memref<4x128xf32, strided<[128, 1]>> to memref<128xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_5 : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

AscendNPU IR OPT success
Detected 910_95/950 device, target=Ascend950PR_9579, enabling auto-bind-sub-block
AscendNPU IR:

#map = affine_map<(d0) -> (d0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func private @triton_indirect_load(memref<1024xf32>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
  func.func @main(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32>, %arg3: memref<?xi32>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %c256_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [4, 256], strides: [%2, %0] : memref<?xf32> to memref<4x256xf32, strided<[256, 1]>>
    %c128_i32 = arith.constant 128 : i32
    %3 = arith.muli %c128_i32, %c1_i32 : i32
    %4 = arith.index_cast %3 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [4, 128], strides: [%4, %0] : memref<?xf32> to memref<4x128xf32, strided<[128, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [4, 256], strides: [%2, %0] : memref<?xi32> to memref<4x256xi32, strided<[256, 1]>>
    %5 = hivm.hir.get_block_idx -> i64
    %6 = arith.trunci %5 : i64 to i32
    %7 = hivm.hir.get_sub_block_idx -> i64
    %8 = arith.trunci %7 : i64 to i32
    %9 = tensor.empty() : tensor<4x256xf32>
    %alloc = memref.alloc() : memref<4x256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x256xf32, strided<[256, 1]>> to memref<4x256xf32>
    %10 = bufferization.to_tensor %alloc restrict : memref<4x256xf32>
    %inserted_slice = tensor.insert_slice %10 into %9[0, 0] [4, 256] [1, 1] : tensor<4x256xf32> into tensor<4x256xf32>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32_2 = arith.constant 1 : i32
    scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32_2  : i32 {
      %11 = tensor.empty() : tensor<256xf32>
      %12 = bufferization.to_memref %inserted_slice : memref<4x256xf32>
      %collapse_shape = memref.collapse_shape %12 [[0, 1]] : memref<4x256xf32> into memref<1024xf32>
      %13 = arith.index_cast %arg11 : i32 to index
      %subview = memref.subview %reinterpret_cast_1[%13, 0] [1, 256] [1, 1] : memref<4x256xi32, strided<[256, 1]>> to memref<1x256xi32, strided<[256, 1], offset: ?>>
      %14 = bufferization.to_tensor %subview restrict : memref<1x256xi32, strided<[256, 1], offset: ?>>
      %collapsed = tensor.collapse_shape %14 [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
      %15 = arith.extsi %collapsed : tensor<256xi32> to tensor<256xi64>
      %c256_i32_3 = arith.constant 256 : i32
      %16 = arith.muli %arg11, %c256_i32_3 : i32
      %17 = arith.extsi %16 : i32 to i64
      %18 = tensor.empty() : tensor<256xi64>
      %19 = linalg.fill ins(%17 : i64) outs(%18 : tensor<256xi64>) -> tensor<256xi64>
      %20 = arith.addi %15, %19 : tensor<256xi64>
      %21 = tensor.empty() : tensor<256xi32>
      %22 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%21 : tensor<256xi32>) attrs =  {tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 256 : index} {
      ^bb0(%out: i32):
        %29 = linalg.index 0 : index
        %30 = arith.index_cast %29 : index to i32
        linalg.yield %30 : i32
      } -> tensor<256xi32>
      %c128_i32_4 = arith.constant 128 : i32
      %23 = tensor.empty() : tensor<256xi32>
      %24 = linalg.fill ins(%c128_i32_4 : i32) outs(%23 : tensor<256xi32>) -> tensor<256xi32>
      %25 = arith.cmpi slt, %22, %24 : tensor<256xi32>
      %cst = arith.constant 0.000000e+00 : f32
      %26 = tensor.empty() : tensor<256xf32>
      %27 = linalg.fill ins(%cst : f32) outs(%26 : tensor<256xf32>) -> tensor<256xf32>
      %28 = func.call @triton_indirect_load(%collapse_shape, %20, %25, %27) : (memref<1024xf32>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
      %extracted_slice = tensor.extract_slice %28[0] [128] [1] : tensor<256xf32> to tensor<128xf32>
      %subview_5 = memref.subview %reinterpret_cast_0[%13, 0] [1, 128] [1, 1] : memref<4x128xf32, strided<[128, 1]>> to memref<128xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice in writable %subview_5 : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

err cmd: /data/pri/Ascend/9.1.0.B010/cann-9.1.0/bin/bishengir-compile /tmp/tmpdtzxf7qo/kernel.npuir --target=Ascend950PR_9579 --enable-auto-multi-buffer=true --disable-ffts --enable-triton-kernel-compile=true --enable-hivm-compile=true --enable-vf-merge-level=1 --enable-hfusion-compile=true --enable-auto-bind-sub-block=true -o /tmp/tmpdtzxf7qo/kernel
err code: 1
err info: loc("/tmp/tmpdtzxf7qo/kernel.npuir":2:1): error: Failed to run buildFinalHIVMPipelines pipeline

loc("/tmp/tmpdtzxf7qo/kernel.npuir":43:13): error: 'func.call' op operand type mismatch: expected operand type 'memref<256xi32, #hivm.address_space<ub>>', but provided 'memref<256xi32, #hivm.address_space<gm>>' for operand number 0
[ERROR] Failed to run BiShengIR pipeline
[ERROR] Executing: /data/pri/Ascend/9.1.0.B010/cann-9.1.0/bin/bishengir-compile-a5 /tmp/tmpdtzxf7qo/kernel.npuir --target=Ascend950PR_9579 --enable-auto-multi-buffer=true --disable-ffts --enable-triton-kernel-compile=true --enable-hivm-compile=true --enable-vf-merge-level=1 --enable-hfusion-compile=true --enable-auto-bind-sub-block=true -o /tmp/tmpdtzxf7qo/kernel
