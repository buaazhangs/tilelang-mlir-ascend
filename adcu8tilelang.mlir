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
    %c46591_i32 = arith.constant 46591 : i32
    %10 = arith.addi %arg5, %c46591_i32 : i32
    %c46592_i32 = arith.constant 46592 : i32
    %11 = arith.divsi %10, %c46592_i32 : i32
    %c1_i32_2 = arith.constant 1 : i32
    scf.for %arg12 = %c0_i32 to %11 step %c1_i32_2  : i32 {
      %12 = tensor.empty() : tensor<832x8xi8>
      %13 = tensor.empty() : tensor<1x832xf32>
      %14 = tensor.empty() : tensor<1x832xf32>
      %c56_i32 = arith.constant 56 : i32
      %15 = arith.muli %arg12, %c56_i32 : i32
      %16 = arith.addi %15, %7 : i32
      %c831_i32 = arith.constant 831 : i32
      %17 = arith.addi %arg5, %c831_i32 : i32
      %c832_i32 = arith.constant 832 : i32
      %18 = arith.divsi %17, %c832_i32 : i32
      %19 = arith.cmpi slt, %16, %18 : i32
      %20:3 = scf.if %19 -> (tensor<832x8xi8>, tensor<1x832xf32>, tensor<1x832xf32>) {
        %c0_i32_3 = arith.constant 0 : i32
        %21 = arith.trunci %c0_i32_3 : i32 to i8
        %22 = linalg.fill ins(%21 : i8) outs(%12 : tensor<832x8xi8>) -> tensor<832x8xi8>
        %23 = arith.sitofp %c0_i32_3 : i32 to f32
        %24 = linalg.fill ins(%23 : f32) outs(%13 : tensor<1x832xf32>) -> tensor<1x832xf32>
        %c8_i32 = arith.constant 8 : i32
        %c1_i32_4 = arith.constant 1 : i32
        %25:2 = scf.for %arg13 = %c0_i32_3 to %c8_i32 step %c1_i32_4 iter_args(%arg14 = %22, %arg15 = %24) -> (tensor<832x8xi8>, tensor<1x832xf32>)  : i32 {
          %35 = tensor.empty() : tensor<8x832xi8>
          %36 = tensor.empty() : tensor<8x832xi32>
          %c46592_i32_7 = arith.constant 46592 : i32
          %37 = arith.muli %arg12, %c46592_i32_7 : i32
          %c832_i32_8 = arith.constant 832 : i32
          %38 = arith.muli %7, %c832_i32_8 : i32
          %39 = arith.addi %37, %38 : i32
          %40 = arith.index_cast %39 : i32 to index
          %41 = arith.subi %arg5, %38 : i32
          %42 = arith.subi %41, %37 : i32
          %43 = arith.minsi %c832_i32_8, %42 : i32
          %44 = arith.index_cast %43 : i32 to index
          %c8_i32_9 = arith.constant 8 : i32
          %45 = arith.muli %arg13, %c8_i32_9 : i32
          %46 = arith.index_cast %45 : i32 to index
          %subview_10 = memref.subview %reinterpret_cast_1[%40, %46] [%44, 8] [1, 1] : memref<?x64xi8, strided<[?, ?]>> to memref<?x8xi8, strided<[?, ?], offset: ?>>
          %alloc_11 = memref.alloc() : memref<832x8xi8>
          %subview_12 = memref.subview %alloc_11[0, 0] [%44, 8] [1, 1] : memref<832x8xi8> to memref<?x8xi8, strided<[8, 1]>>
          memref.copy %subview_10, %subview_12 : memref<?x8xi8, strided<[?, ?], offset: ?>> to memref<?x8xi8, strided<[8, 1]>>
          %47 = bufferization.to_tensor %subview_12 restrict : memref<?x8xi8, strided<[8, 1]>>
          %c0 = arith.constant 0 : index
          %dim = tensor.dim %47, %c0 : tensor<?x8xi8>
          %c1 = arith.constant 1 : index
          %expanded_13 = tensor.expand_shape %47 [[0], [1]] output_shape [%dim, 8] : tensor<?x8xi8> into tensor<?x8xi8>
          %inserted_slice_14 = tensor.insert_slice %expanded_13 into %arg14[0, 0] [%44, 8] [1, 1] : tensor<?x8xi8> into tensor<832x8xi8>
          %transposed = linalg.transpose ins(%inserted_slice_14 : tensor<832x8xi8>) outs(%35 : tensor<8x832xi8>) permutation = [1, 0] 
          %48 = tensor.empty() : tensor<8x832xi8>
          %49 = hfusion.cast {enable_overflow = true, round_mode = #hfusion.round_mode<rint>, type_fn = #hfusion.type_fn<cast_unsigned>, unsigned_mode = #hfusion.unsigned_mode<ui2si>} ins(%transposed : tensor<8x832xi8>) outs(%36 : tensor<8x832xi32>) -> tensor<8x832xi32>
          %c0_i32_15 = arith.constant 0 : i32
          %c1_i32_16 = arith.constant 1 : i32
          %50 = scf.for %arg16 = %c0_i32_15 to %c8_i32_9 step %c1_i32_16 iter_args(%arg17 = %arg15) -> (tensor<1x832xf32>)  : i32 {
            %51 = tensor.empty() : tensor<1x832xf32>
            %c8_i32_17 = arith.constant 8 : i32
            %52 = arith.muli %arg13, %c8_i32_17 : i32
            %53 = arith.addi %52, %arg16 : i32
            %54 = arith.index_cast %53 : i32 to index
            %extracted_slice_18 = tensor.extract_slice %inserted_slice[%54, 0] [1, 256] [1, 1] : tensor<64x256xf32> to tensor<1x256xf32>
            %55 = arith.index_cast %arg16 : i32 to index
            %extracted_slice_19 = tensor.extract_slice %49[%55, 0] [1, 832] [1, 1] : tensor<8x832xi32> to tensor<1x832xi32>
            %56 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%extracted_slice_18, %extracted_slice_19 : tensor<1x256xf32>, tensor<1x832xi32>) outs(%51 : tensor<1x832xf32>) axis = 1 -> tensor<1x832xf32>
            %57 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg17, %56 : tensor<1x832xf32>, tensor<1x832xf32>) outs(%arg17 : tensor<1x832xf32>) -> tensor<1x832xf32>
            scf.yield %57 : tensor<1x832xf32>
          }
          scf.yield %inserted_slice_14, %50 : tensor<832x8xi8>, tensor<1x832xf32>
        }
        %26 = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%25#1 : tensor<1x832xf32>) outs(%14 : tensor<1x832xf32>) -> tensor<1x832xf32>
        %c832_i32_5 = arith.constant 832 : i32
        %27 = arith.muli %7, %c832_i32_5 : i32
        %28 = arith.subi %arg5, %27 : i32
        %c46592_i32_6 = arith.constant 46592 : i32
        %29 = arith.muli %arg12, %c46592_i32_6 : i32
        %30 = arith.subi %28, %29 : i32
        %31 = arith.minsi %c832_i32_5, %30 : i32
        %32 = arith.index_cast %31 : i32 to index
        %33 = arith.addi %29, %27 : i32
        %34 = arith.index_cast %33 : i32 to index
        %extracted_slice = tensor.extract_slice %26[0, 0] [1, %32] [1, 1] : tensor<1x832xf32> to tensor<?xf32>
        %subview = memref.subview %reinterpret_cast_0[%34] [%32] [1] : memref<?xf32, strided<[?]>> to memref<?xf32, strided<[?], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xf32>, memref<?xf32, strided<[?], offset: ?>>) -> ()
        scf.yield %25#0, %25#1, %26 : tensor<832x8xi8>, tensor<1x832xf32>, tensor<1x832xf32>
      } else {
        scf.yield %12, %13, %14 : tensor<832x8xi8>, tensor<1x832xf32>, tensor<1x832xf32>
      }
    }
    return
  }
}