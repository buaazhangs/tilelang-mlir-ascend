module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>, memref.memref_as_ptr} {
  func.func @adc_func(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf32>, %arg3: memref<?xi8>, %arg4: memref<?xf32>, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.index_cast %c1_i32 : i32 to index
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %c256_i32, %c1_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [64, 256], strides: [%2, %0] : memref<?xf32> to memref<64x256xf32, strided<[256, 1]>>
    %3 = arith.index_cast %arg5 : i32 to index
    %reinterpret_cast_0 = memref.reinterpret_cast %arg4 to offset: [0], sizes: [%3], strides: [%0] : memref<?xf32> to memref<?xf32, strided<[?]>>
    %c64_i32 = arith.constant 64 : i32
    %4 = arith.muli %c64_i32, %c1_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [0], sizes: [%3, 64], strides: [%5, %0] : memref<?xi8> to memref<?x64xi8, strided<[?, ?]>>
    %6 = hivm.hir.get_block_idx -> i64
    %7 = arith.trunci %6 : i64 to i32
    %8 = tensor.empty() : tensor<64x256xf32>
    %alloc = memref.alloc() : memref<64x256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<64x256xf32, strided<[256, 1]>> to memref<64x256xf32>
    %9 = bufferization.to_tensor %alloc restrict : memref<64x256xf32>
    %expanded = tensor.expand_shape %9 [[0], [1]] output_shape [64, 256] : tensor<64x256xf32> into tensor<64x256xf32>
    %inserted_slice = tensor.insert_slice %expanded into %8[0, 0] [64, 256] [1, 1] : tensor<64x256xf32> into tensor<64x256xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1791_i32 = arith.constant 1791 : i32
    %10 = arith.addi %arg5, %c1791_i32 : i32
    %c1792_i32 = arith.constant 1792 : i32
    %11 = arith.divsi %10, %c1792_i32 : i32
    %c1_i32_2 = arith.constant 1 : i32
    scf.for %arg12 = %c0_i32 to %11 step %c1_i32_2  : i32 {
      %12 = tensor.empty() : tensor<32x16xi8>
      %13 = tensor.empty() : tensor<32xf32>
      %14 = tensor.empty() : tensor<32xf32>
      %c56_i32 = arith.constant 56 : i32
      %15 = arith.muli %arg12, %c56_i32 : i32
      %16 = arith.addi %15, %7 : i32
      %c31_i32 = arith.constant 31 : i32
      %17 = arith.addi %arg5, %c31_i32 : i32
      %c32_i32 = arith.constant 32 : i32
      %18 = arith.divsi %17, %c32_i32 : i32
      %19 = arith.cmpi slt, %16, %18 : i32
      %20:3 = scf.if %19 -> (tensor<32x16xi8>, tensor<32xf32>, tensor<32xf32>) {
        %c0_i32_3 = arith.constant 0 : i32
        %21 = arith.trunci %c0_i32_3 : i32 to i8
        %22 = linalg.fill ins(%21 : i8) outs(%12 : tensor<32x16xi8>) -> tensor<32x16xi8>
        %23 = arith.sitofp %c0_i32_3 : i32 to f32
        %24 = linalg.fill ins(%23 : f32) outs(%13 : tensor<32xf32>) -> tensor<32xf32>
        %c4_i32 = arith.constant 4 : i32
        %c1_i32_4 = arith.constant 1 : i32
        %25:2 = scf.for %arg13 = %c0_i32_3 to %c4_i32 step %c1_i32_4 iter_args(%arg14 = %22, %arg15 = %24) -> (tensor<32x16xi8>, tensor<32xf32>)  : i32 {
          %35 = tensor.empty() : tensor<32x16xi32>
          %c1792_i32_7 = arith.constant 1792 : i32
          %36 = arith.muli %arg12, %c1792_i32_7 : i32
          %c32_i32_8 = arith.constant 32 : i32
          %37 = arith.muli %7, %c32_i32_8 : i32
          %38 = arith.addi %36, %37 : i32
          %39 = arith.index_cast %38 : i32 to index
          %40 = arith.subi %arg5, %37 : i32
          %41 = arith.subi %40, %36 : i32
          %42 = arith.minsi %c32_i32_8, %41 : i32
          %43 = arith.index_cast %42 : i32 to index
          %c16_i32 = arith.constant 16 : i32
          %44 = arith.muli %arg13, %c16_i32 : i32
          %45 = arith.index_cast %44 : i32 to index
          %subview_9 = memref.subview %reinterpret_cast_1[%39, %45] [%43, 16] [1, 1] : memref<?x64xi8, strided<[?, ?]>> to memref<?x16xi8, strided<[?, ?], offset: ?>>
          %alloc_10 = memref.alloc() : memref<32x16xi8>
          %subview_11 = memref.subview %alloc_10[0, 0] [%43, 16] [1, 1] : memref<32x16xi8> to memref<?x16xi8, strided<[16, 1]>>
          memref.copy %subview_9, %subview_11 : memref<?x16xi8, strided<[?, ?], offset: ?>> to memref<?x16xi8, strided<[16, 1]>>
          %46 = bufferization.to_tensor %subview_11 restrict : memref<?x16xi8, strided<[16, 1]>>
          %c0 = arith.constant 0 : index
          %dim = tensor.dim %46, %c0 : tensor<?x16xi8>
          %c1 = arith.constant 1 : index
          %expanded_12 = tensor.expand_shape %46 [[0], [1]] output_shape [%dim, 16] : tensor<?x16xi8> into tensor<?x16xi8>
          %inserted_slice_13 = tensor.insert_slice %expanded_12 into %arg14[0, 0] [%43, 16] [1, 1] : tensor<?x16xi8> into tensor<32x16xi8>
          %47 = tensor.empty() : tensor<32x16xi8>
          %48 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<ui2si>} ins(%inserted_slice_13 : tensor<32x16xi8>) outs(%35 : tensor<32x16xi32>) -> tensor<32x16xi32>
          %c0_i32_14 = arith.constant 0 : i32
          %c1_i32_15 = arith.constant 1 : i32
          %49 = scf.for %arg16 = %c0_i32_14 to %c16_i32 step %c1_i32_15 iter_args(%arg17 = %arg15) -> (tensor<32xf32>)  : i32 {
            %50 = tensor.empty() : tensor<32xi32>
            %51 = tensor.empty() : tensor<256xf32>
            %52 = tensor.empty() : tensor<32xf32>
            %c16_i32_16 = arith.constant 16 : i32
            %53 = arith.muli %arg13, %c16_i32_16 : i32
            %54 = arith.addi %53, %arg16 : i32
            %55 = arith.index_cast %54 : i32 to index
            %extracted_slice_17 = tensor.extract_slice %inserted_slice[%55, 0] [1, 256] [1, 1] : tensor<64x256xf32> to tensor<256xf32>
            %expanded_18 = tensor.expand_shape %extracted_slice_17 [[0]] output_shape [256] : tensor<256xf32> into tensor<256xf32>
            %inserted_slice_19 = tensor.insert_slice %expanded_18 into %51[0] [256] [1] : tensor<256xf32> into tensor<256xf32>
            %56 = arith.index_cast %arg16 : i32 to index
            %extracted_slice_20 = tensor.extract_slice %48[0, %56] [32, 1] [1, 1] : tensor<32x16xi32> to tensor<32xi32>
            %expanded_21 = tensor.expand_shape %extracted_slice_20 [[0]] output_shape [32] : tensor<32xi32> into tensor<32xi32>
            %inserted_slice_22 = tensor.insert_slice %expanded_21 into %50[0] [32] [1] : tensor<32xi32> into tensor<32xi32>
            %57 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%inserted_slice_19, %inserted_slice_22 : tensor<256xf32>, tensor<32xi32>) outs(%52 : tensor<32xf32>) axis = 0 -> tensor<32xf32>
            %58 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg17, %57 : tensor<32xf32>, tensor<32xf32>) outs(%arg17 : tensor<32xf32>) -> tensor<32xf32>
            scf.yield %58 : tensor<32xf32>
          }
          scf.yield %inserted_slice_13, %49 : tensor<32x16xi8>, tensor<32xf32>
        }
        %26 = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%25#1 : tensor<32xf32>) outs(%14 : tensor<32xf32>) -> tensor<32xf32>
        %c32_i32_5 = arith.constant 32 : i32
        %27 = arith.muli %7, %c32_i32_5 : i32
        %28 = arith.subi %arg5, %27 : i32
        %c1792_i32_6 = arith.constant 1792 : i32
        %29 = arith.muli %arg12, %c1792_i32_6 : i32
        %30 = arith.subi %28, %29 : i32
        %31 = arith.minsi %c32_i32_5, %30 : i32
        %32 = arith.index_cast %31 : i32 to index
        %33 = arith.addi %29, %27 : i32
        %34 = arith.index_cast %33 : i32 to index
        %extracted_slice = tensor.extract_slice %26[0] [%32] [1] : tensor<32xf32> to tensor<?xf32>
        %subview = memref.subview %reinterpret_cast_0[%34] [%32] [1] : memref<?xf32, strided<[?]>> to memref<?xf32, strided<[?], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xf32>, memref<?xf32, strided<[?], offset: ?>>) -> ()
        scf.yield %25#0, %25#1, %26 : tensor<32x16xi8>, tensor<32xf32>, tensor<32xf32>
      } else {
        scf.yield %12, %13, %14 : tensor<32x16xi8>, tensor<32xf32>, tensor<32xf32>
      }
    }
    return
  }
}