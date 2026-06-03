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
    %c3583_i32 = arith.constant 3583 : i32
    %10 = arith.addi %arg5, %c3583_i32 : i32
    %c3584_i32 = arith.constant 3584 : i32
    %11 = arith.divsi %10, %c3584_i32 : i32
    %c1_i32_2 = arith.constant 1 : i32
    scf.for %arg12 = %c0_i32 to %11 step %c1_i32_2  : i32 {
      %12 = tensor.empty() : tensor<64x16xi8>
      %13 = tensor.empty() : tensor<64xf32>
      %14 = tensor.empty() : tensor<64xf32>
      %c56_i32 = arith.constant 56 : i32
      %15 = arith.muli %arg12, %c56_i32 : i32
      %16 = arith.addi %15, %7 : i32
      %c63_i32 = arith.constant 63 : i32
      %17 = arith.addi %arg5, %c63_i32 : i32
      %c64_i32_3 = arith.constant 64 : i32
      %18 = arith.divsi %17, %c64_i32_3 : i32
      %19 = arith.cmpi slt, %16, %18 : i32
      %20:3 = scf.if %19 -> (tensor<64x16xi8>, tensor<64xf32>, tensor<64xf32>) {
        %c64_i32_4 = arith.constant 64 : i32
        %21 = arith.muli %7, %c64_i32_4 : i32
        %22 = arith.subi %arg5, %21 : i32
        %c3584_i32_5 = arith.constant 3584 : i32
        %23 = arith.muli %arg12, %c3584_i32_5 : i32
        %24 = arith.subi %22, %23 : i32
        %25 = arith.cmpi slt, %24, %c64_i32_4 : i32
        scf.if %25 {
          %c0_i32_8 = arith.constant 0 : i32
          %34 = arith.trunci %c0_i32_8 : i32 to i8
          %35 = linalg.fill ins(%34 : i8) outs(%12 : tensor<64x16xi8>) -> tensor<64x16xi8>
        }
        %c0_i32_6 = arith.constant 0 : i32
        %26 = arith.sitofp %c0_i32_6 : i32 to f32
        %27 = linalg.fill ins(%26 : f32) outs(%13 : tensor<64xf32>) -> tensor<64xf32>
        %c4_i32 = arith.constant 4 : i32
        %c1_i32_7 = arith.constant 1 : i32
        %28:2 = scf.for %arg13 = %c0_i32_6 to %c4_i32 step %c1_i32_7 iter_args(%arg14 = %12, %arg15 = %27) -> (tensor<64x16xi8>, tensor<64xf32>)  : i32 {
          %c3584_i32_8 = arith.constant 3584 : i32
          %34 = arith.muli %arg12, %c3584_i32_8 : i32
          %c64_i32_9 = arith.constant 64 : i32
          %35 = arith.muli %7, %c64_i32_9 : i32
          %36 = arith.addi %34, %35 : i32
          %37 = arith.index_cast %36 : i32 to index
          %38 = arith.subi %arg5, %35 : i32
          %39 = arith.subi %38, %34 : i32
          %40 = arith.minsi %c64_i32_9, %39 : i32
          %41 = arith.index_cast %40 : i32 to index
          %c16_i32 = arith.constant 16 : i32
          %42 = arith.muli %arg13, %c16_i32 : i32
          %43 = arith.index_cast %42 : i32 to index
          %subview_10 = memref.subview %reinterpret_cast_1[%37, %43] [%41, 16] [1, 1] : memref<?x64xi8, strided<[?, ?]>> to memref<?x16xi8, strided<[?, ?], offset: ?>>
          %alloc_11 = memref.alloc() : memref<64x16xi8>
          %subview_12 = memref.subview %alloc_11[0, 0] [%41, 16] [1, 1] : memref<64x16xi8> to memref<?x16xi8, strided<[16, 1]>>
          memref.copy %subview_10, %subview_12 : memref<?x16xi8, strided<[?, ?], offset: ?>> to memref<?x16xi8, strided<[16, 1]>>
          %44 = bufferization.to_tensor %subview_12 restrict : memref<?x16xi8, strided<[16, 1]>>
          %c0 = arith.constant 0 : index
          %dim = tensor.dim %44, %c0 : tensor<?x16xi8>
          %c1 = arith.constant 1 : index
          %expanded_13 = tensor.expand_shape %44 [[0], [1]] output_shape [%dim, 16] : tensor<?x16xi8> into tensor<?x16xi8>
          %inserted_slice_14 = tensor.insert_slice %expanded_13 into %arg14[0, 0] [%41, 16] [1, 1] : tensor<?x16xi8> into tensor<64x16xi8>
          %c0_i32_15 = arith.constant 0 : i32
          %c1_i32_16 = arith.constant 1 : i32
          %45 = scf.for %arg16 = %c0_i32_15 to %c16_i32 step %c1_i32_16 iter_args(%arg17 = %arg15) -> (tensor<64xf32>)  : i32 {
            %46 = tensor.empty() : tensor<64xi8>
            %47 = tensor.empty() : tensor<64xi32>
            %48 = tensor.empty() : tensor<256xf32>
            %49 = tensor.empty() : tensor<64xf32>
            %c64_i32_17 = arith.constant 64 : i32
            %50 = arith.muli %7, %c64_i32_17 : i32
            %51 = arith.subi %arg5, %50 : i32
            %c3584_i32_18 = arith.constant 3584 : i32
            %52 = arith.muli %arg12, %c3584_i32_18 : i32
            %53 = arith.subi %51, %52 : i32
            %54 = arith.cmpi slt, %53, %c64_i32_17 : i32
            scf.if %54 {
              %c0_i32_26 = arith.constant 0 : i32
              %63 = arith.trunci %c0_i32_26 : i32 to i8
              %64 = linalg.fill ins(%63 : i8) outs(%46 : tensor<64xi8>) -> tensor<64xi8>
            }
            %55 = arith.index_cast %arg16 : i32 to index
            %extracted_slice_19 = tensor.extract_slice %inserted_slice_14[0, %55] [64, 1] [1, 1] : tensor<64x16xi8> to tensor<64xi8>
            %expanded_20 = tensor.expand_shape %extracted_slice_19 [[0]] output_shape [64] : tensor<64xi8> into tensor<64xi8>
            %inserted_slice_21 = tensor.insert_slice %expanded_20 into %46[0] [64] [1] : tensor<64xi8> into tensor<64xi8>
            %56 = tensor.empty() : tensor<64xi8>
            %57 = hfusion.cast {cast = #hfusion.type_fn<cast_unsigned>, enable_overflow = true, round_mode = #hfusion.round_mode<rint>, unsigned_mode = #hfusion.unsigned_mode<ui2si>} ins(%inserted_slice_21 : tensor<64xi8>) outs(%47 : tensor<64xi32>) -> tensor<64xi32>
            %c16_i32_22 = arith.constant 16 : i32
            %58 = arith.muli %arg13, %c16_i32_22 : i32
            %59 = arith.addi %58, %arg16 : i32
            %60 = arith.index_cast %59 : i32 to index
            %extracted_slice_23 = tensor.extract_slice %inserted_slice[%60, 0] [1, 256] [1, 1] : tensor<64x256xf32> to tensor<256xf32>
            %expanded_24 = tensor.expand_shape %extracted_slice_23 [[0]] output_shape [256] : tensor<256xf32> into tensor<256xf32>
            %inserted_slice_25 = tensor.insert_slice %expanded_24 into %48[0] [256] [1] : tensor<256xf32> into tensor<256xf32>
            %61 = hfusion.gather {operandSegmentSizes = array<i32: 2, 1>} ins(%inserted_slice_25, %57 : tensor<256xf32>, tensor<64xi32>) outs(%49 : tensor<64xf32>) axis = 0 -> tensor<64xf32>
            %62 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%arg17, %61 : tensor<64xf32>, tensor<64xf32>) outs(%arg17 : tensor<64xf32>) -> tensor<64xf32>
            scf.yield %62 : tensor<64xf32>
          }
          scf.yield %inserted_slice_14, %45 : tensor<64x16xi8>, tensor<64xf32>
        }
        %29 = linalg.elemwise_unary {fun = #linalg.unary_fn<sqrt>} ins(%28#1 : tensor<64xf32>) outs(%14 : tensor<64xf32>) -> tensor<64xf32>
        %30 = arith.minsi %c64_i32_4, %24 : i32
        %31 = arith.index_cast %30 : i32 to index
        %32 = arith.addi %23, %21 : i32
        %33 = arith.index_cast %32 : i32 to index
        %extracted_slice = tensor.extract_slice %29[0] [%31] [1] : tensor<64xf32> to tensor<?xf32>
        %subview = memref.subview %reinterpret_cast_0[%33] [%31] [1] : memref<?xf32, strided<[?]>> to memref<?xf32, strided<[?], offset: ?>>
        bufferization.materialize_in_destination %extracted_slice in writable %subview : (tensor<?xf32>, memref<?xf32, strided<[?], offset: ?>>) -> ()
        scf.yield %28#0, %28#1, %29 : tensor<64x16xi8>, tensor<64xf32>, tensor<64xf32>
      } else {
        scf.yield %12, %13, %14 : tensor<64x16xi8>, tensor<64xf32>, tensor<64xf32>
      }
    }
    return
  }
}