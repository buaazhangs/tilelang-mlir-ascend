#map = affine_map<(d0) -> (d0)>
module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func private @triton_indirect_load(memref<?xf32>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
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
    %9 = tensor.empty() : tensor<4x256xi32>
    %alloc = memref.alloc() : memref<4x256xi32>
    memref.copy %reinterpret_cast_1, %alloc : memref<4x256xi32, strided<[256, 1]>> to memref<4x256xi32>
    %10 = bufferization.to_tensor %alloc restrict : memref<4x256xi32>
    %inserted_slice = tensor.insert_slice %10 into %9[0, 0] [4, 256] [1, 1] : tensor<4x256xi32> into tensor<4x256xi32>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32_2 = arith.constant 1 : i32
    scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32_2  : i32 {
      %11 = tensor.empty() : tensor<256xf32>
      %12 = arith.index_cast %arg11 : i32 to index
      %extracted_slice = tensor.extract_slice %inserted_slice[%12, 0] [1, 256] [1, 1] : tensor<4x256xi32> to tensor<1x256xi32>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
      %13 = arith.extsi %collapsed : tensor<256xi32> to tensor<256xi64>
      %c256_i32_3 = arith.constant 256 : i32
      %14 = arith.muli %arg11, %c256_i32_3 : i32
      %15 = arith.extsi %14 : i32 to i64
      %16 = tensor.empty() : tensor<256xi64>
      %17 = linalg.fill ins(%15 : i64) outs(%16 : tensor<256xi64>) -> tensor<256xi64>
      %18 = arith.addi %13, %17 : tensor<256xi64>
      %19 = tensor.empty() : tensor<256xi32>
      %20 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%19 : tensor<256xi32>) attrs =  {tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 256 : index} {
      ^bb0(%out: i32):
        %27 = linalg.index 0 : index
        %28 = arith.index_cast %27 : index to i32
        linalg.yield %28 : i32
      } -> tensor<256xi32>
      %c128_i32_4 = arith.constant 128 : i32
      %21 = tensor.empty() : tensor<256xi32>
      %22 = linalg.fill ins(%c128_i32_4 : i32) outs(%21 : tensor<256xi32>) -> tensor<256xi32>
      %23 = arith.cmpi slt, %20, %22 : tensor<256xi32>
      %cst = arith.constant 0.000000e+00 : f32
      %24 = tensor.empty() : tensor<256xf32>
      %25 = linalg.fill ins(%cst : f32) outs(%24 : tensor<256xf32>) -> tensor<256xf32>
      %26 = func.call @triton_indirect_load(%arg2, %18, %23, %25) : (memref<?xf32>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
      %extracted_slice_5 = tensor.extract_slice %26[0] [128] [1] : tensor<256xf32> to tensor<128xf32>
      %subview = memref.subview %reinterpret_cast_0[%12, 0] [1, 128] [1, 1] : memref<4x128xf32, strided<[128, 1]>> to memref<128xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_5 in writable %subview : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
