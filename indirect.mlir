2026-05-20 06:20:49  [TileLang:tilelang.env:WARNING]: CUTLASS is not installed or found in the expected path
====== TVM IR ======
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(X: T.Buffer((2000,), "float32"), Y: T.Buffer((1000,), "float32"), IDX_GM: T.Buffer((1000,), "int32"), OUT_GM: T.Buffer((1000,), "float32")):
        T.func_attr({"target": T.target({"host": {"keys": ["cpu"], "kind": "stackvm", "tag": ""}, "keys": [], "kind": "npuir", "tag": ""})})
        pid = T.launch_thread("blockIdx.x", 4)
        _ = T.launch_thread("blockIdx.y", 8)
        IDX_UB = T.decl_buffer((256,), "int32", scope="shared")
        O_UB = T.decl_buffer((256,), scope="shared")
        Y_UB = T.decl_buffer((256,), scope="shared")
        SUM_UB = T.decl_buffer((256,), scope="shared")
        T.npuir_brc(0, T.region(O_UB[0], 2, 256))
        T.npuir_brc(0, T.region(Y_UB[0], 2, 256))
        T.copy(T.region(IDX_GM[pid * 256], 1, T.min(256, 1000 - pid * 256)), T.region(IDX_UB[0], 2, T.min(256, 1000 - pid * 256)))
        T.copy(T.region(Y[pid * 256], 1, T.min(256, 1000 - pid * 256)), T.region(Y_UB[0], 2, T.min(256, 1000 - pid * 256)))
        T.npuir_indirect_load(T.region(X[0], 1, 1), T.region(IDX_UB[0], 1, 256), T.region(O_UB[0], 2, 256), 1000 - pid * 256)
        T.npuir_add(T.region(O_UB[0], 1, 256), T.region(Y_UB[0], 1, 256), T.region(SUM_UB[0], 2, 256))
        T.copy(T.region(SUM_UB[0], 1, T.min(256, 1000 - pid * 256)), T.region(OUT_GM[pid * 256], 2, T.min(256, 1000 - pid * 256)))

====== npuir ======
#map = affine_map<(d0) -> (d0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func private @triton_indirect_load(memref<2000xf32, strided<[1]>>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
  func.func @main(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xi32>, %arg5: memref<?xf32>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [2000], strides: [%0] : memref<?xf32> to memref<2000xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1000], strides: [%0] : memref<?xi32> to memref<1000xi32, strided<[1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1000], strides: [%0] : memref<?xf32> to memref<1000xf32, strided<[1]>>
    %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [0], sizes: [1000], strides: [%0] : memref<?xf32> to memref<1000xf32, strided<[1]>>
    %1 = hivm.hir.get_block_idx -> i64
    %2 = arith.trunci %1 : i64 to i32
    %3 = tensor.empty() : tensor<256xi32>
    %4 = tensor.empty() : tensor<256xf32>
    %5 = tensor.empty() : tensor<256xf32>
    %6 = tensor.empty() : tensor<256xf32>
    %c0_i32 = arith.constant 0 : i32
    %7 = arith.sitofp %c0_i32 : i32 to f32
    %8 = linalg.fill ins(%7 : f32) outs(%4 : tensor<256xf32>) -> tensor<256xf32>
    %9 = arith.sitofp %c0_i32 : i32 to f32
    %10 = linalg.fill ins(%9 : f32) outs(%5 : tensor<256xf32>) -> tensor<256xf32>
    %c256_i32 = arith.constant 256 : i32
    %11 = arith.muli %2, %c256_i32 : i32
    %12 = arith.index_cast %11 : i32 to index
    %c1000_i32 = arith.constant 1000 : i32
    %13 = arith.subi %c1000_i32, %11 : i32
    %14 = arith.minsi %c256_i32, %13 : i32
    %15 = arith.index_cast %14 : i32 to index
    %subview = memref.subview %reinterpret_cast_0[%12] [%15] [1] : memref<1000xi32, strided<[1]>> to memref<?xi32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<256xi32>
    %subview_3 = memref.subview %alloc[0] [%15] [1] : memref<256xi32> to memref<?xi32, strided<[1]>>
    memref.copy %subview, %subview_3 : memref<?xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1]>>
    %16 = bufferization.to_tensor %subview_3 restrict : memref<?xi32, strided<[1]>>
    %inserted_slice = tensor.insert_slice %16 into %3[0] [%15] [1] : tensor<?xi32> into tensor<256xi32>
    %subview_4 = memref.subview %reinterpret_cast_1[%12] [%15] [1] : memref<1000xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
    %alloc_5 = memref.alloc() : memref<256xf32>
    %subview_6 = memref.subview %alloc_5[0] [%15] [1] : memref<256xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_4, %subview_6 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %17 = bufferization.to_tensor %subview_6 restrict : memref<?xf32, strided<[1]>>
    %inserted_slice_7 = tensor.insert_slice %17 into %10[0] [%15] [1] : tensor<?xf32> into tensor<256xf32>
    %18 = arith.extsi %inserted_slice : tensor<256xi32> to tensor<256xi64>
    %19 = tensor.empty() : tensor<256xi32>
    %20 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%19 : tensor<256xi32>) attrs =  {tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 256 : index} {
    ^bb0(%out: i32):
      %28 = linalg.index 0 : index
      %29 = arith.index_cast %28 : index to i32
      linalg.yield %29 : i32
    } -> tensor<256xi32>
    %21 = tensor.empty() : tensor<256xi32>
    %22 = linalg.fill ins(%13 : i32) outs(%21 : tensor<256xi32>) -> tensor<256xi32>
    %23 = arith.cmpi slt, %20, %22 : tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %24 = tensor.empty() : tensor<256xf32>
    %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<256xf32>) -> tensor<256xf32>
    %26 = call @triton_indirect_load(%reinterpret_cast, %18, %23, %25) : (memref<2000xf32, strided<[1]>>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
    %27 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%26, %inserted_slice_7 : tensor<256xf32>, tensor<256xf32>) outs(%6 : tensor<256xf32>) -> tensor<256xf32>
    %extracted_slice = tensor.extract_slice %27[0] [%15] [1] : tensor<256xf32> to tensor<?xf32>
    %subview_8 = memref.subview %reinterpret_cast_2[%12] [%15] [1] : memref<1000xf32, strided<[1]>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_8 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

AscendNPU IR OPT success
Detected 910_95/950 device, target=Ascend950PR_9579, enabling auto-bind-sub-block
AscendNPU IR compile success: 
PASS
