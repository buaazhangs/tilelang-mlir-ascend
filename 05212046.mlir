2026-05-21 12:39:17  [TileLang:tilelang.env:WARNING]: CUTLASS is not installed or found in the expected path
[NpuSimtIndirectLoad] visit T.Parallel:
for m in T.parallel(256):
    if m < 128:
        O_UB = T.Buffer((256,), scope="shared")
        LUT_shared = T.Buffer((4, 256), scope="shared")
        s = T.int32()
        CODES_UB = T.Buffer((4, 256), "int32", scope="shared")
        O_UB[m] = LUT_shared[s, CODES_UB[s, m]]
[NpuSimtIndirectLoad] 1111:
O_UB = T.Buffer((256,), scope="shared")
LUT_shared = T.Buffer((4, 256), scope="shared")
s = T.int32()
CODES_UB = T.Buffer((4, 256), "int32", scope="shared")
m = T.int32()
O_UB[m] = LUT_shared[s, CODES_UB[s, m]]
[NpuSimtIndirectLoad] rewritten TIR:
LUT_shared = T.Buffer((4, 256), scope="shared")
s = T.int32()
CODES_UB = T.Buffer((4, 256), "int32", scope="shared")
O_UB = T.Buffer((256,), scope="shared")
T.npuir_indirect_load(T.region(LUT_shared[s, 0], 1, 1, 256), T.region(CODES_UB[s, 0], 1, 1, 256), T.region(O_UB[0], 2, 256), 128)
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
        CODES_UB = T.decl_buffer((4, 256), "int32", scope="shared")
        T.copy(T.region(LUT[0, 0], 1, 4, 256), T.region(LUT_shared[0, 0], 2, 4, 256))
        T.copy(T.region(CODES[0, 0], 1, 4, 256), T.region(CODES_UB[0, 0], 2, 4, 256))
        for s in range(4):
            O_UB = T.decl_buffer((256,), scope="shared")
            T.npuir_indirect_load(T.region(LUT_shared[s, 0], 1, 1, 256), T.region(CODES_UB[s, 0], 1, 1, 256), T.region(O_UB[0], 2, 256), 128)
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
    %10 = tensor.empty() : tensor<4x256xi32>
    %alloc = memref.alloc() : memref<4x256xf32>
    memref.copy %reinterpret_cast, %alloc : memref<4x256xf32, strided<[256, 1]>> to memref<4x256xf32>
    %11 = bufferization.to_tensor %alloc restrict : memref<4x256xf32>
    %inserted_slice = tensor.insert_slice %11 into %9[0, 0] [4, 256] [1, 1] : tensor<4x256xf32> into tensor<4x256xf32>
    %alloc_2 = memref.alloc() : memref<4x256xi32>
    memref.copy %reinterpret_cast_1, %alloc_2 : memref<4x256xi32, strided<[256, 1]>> to memref<4x256xi32>
    %12 = bufferization.to_tensor %alloc_2 restrict : memref<4x256xi32>
    %inserted_slice_3 = tensor.insert_slice %12 into %10[0, 0] [4, 256] [1, 1] : tensor<4x256xi32> into tensor<4x256xi32>
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32_4 = arith.constant 1 : i32
    scf.for %arg11 = %c0_i32 to %c4_i32 step %c1_i32_4  : i32 {
      %13 = tensor.empty() : tensor<256xf32>
      %14 = bufferization.to_memref %inserted_slice : memref<4x256xf32>
      %collapse_shape = memref.collapse_shape %14 [[0, 1]] : memref<4x256xf32> into memref<1024xf32>
      %15 = arith.index_cast %arg11 : i32 to index
      %extracted_slice = tensor.extract_slice %inserted_slice_3[%15, 0] [1, 256] [1, 1] : tensor<4x256xi32> to tensor<1x256xi32>
      %collapsed = tensor.collapse_shape %extracted_slice [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
      %16 = arith.extsi %collapsed : tensor<256xi32> to tensor<256xi64>
      %c256_i32_5 = arith.constant 256 : i32
      %17 = arith.muli %arg11, %c256_i32_5 : i32
      %18 = arith.extsi %17 : i32 to i64
      %19 = tensor.empty() : tensor<256xi64>
      %20 = linalg.fill ins(%18 : i64) outs(%19 : tensor<256xi64>) -> tensor<256xi64>
      %21 = arith.addi %16, %20 : tensor<256xi64>
      %22 = tensor.empty() : tensor<256xi32>
      %23 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%22 : tensor<256xi32>) attrs =  {tt.from_make_range, tt.make_range_offset = 0 : index, tt.make_range_size = 256 : index} {
      ^bb0(%out: i32):
        %30 = linalg.index 0 : index
        %31 = arith.index_cast %30 : index to i32
        linalg.yield %31 : i32
      } -> tensor<256xi32>
      %c128_i32_6 = arith.constant 128 : i32
      %24 = tensor.empty() : tensor<256xi32>
      %25 = linalg.fill ins(%c128_i32_6 : i32) outs(%24 : tensor<256xi32>) -> tensor<256xi32>
      %26 = arith.cmpi slt, %23, %25 : tensor<256xi32>
      %cst = arith.constant 0.000000e+00 : f32
      %27 = tensor.empty() : tensor<256xf32>
      %28 = linalg.fill ins(%cst : f32) outs(%27 : tensor<256xf32>) -> tensor<256xf32>
      %29 = func.call @triton_indirect_load(%collapse_shape, %21, %26, %28) : (memref<1024xf32>, tensor<256xi64>, tensor<256xi1>, tensor<256xf32>) -> tensor<256xf32>
      %extracted_slice_7 = tensor.extract_slice %29[0] [128] [1] : tensor<256xf32> to tensor<128xf32>
      %subview = memref.subview %reinterpret_cast_0[%15, 0] [1, 128] [1, 1] : memref<4x128xf32, strided<[128, 1]>> to memref<128xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %extracted_slice_7 in writable %subview : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}

AscendNPU IR OPT success
Detected 910_95/950 device, target=Ascend950PR_9579, enabling auto-bind-sub-block
AscendNPU IR compile success: 
WARNING: synchronous waiting (in Close) failed; there might be a device assert failure or timeout.
[E521 12:39:23.758647926 OpParamMaker.cpp:454] operator():../third_party/op-plugin/op_plugin/ops/opapi/ArangeKernelNpuOpApi.cpp:36 NPU function error: device error type 3, error code is 507035
[ERROR] 2026-05-21-12:39:23 (PID:697, Device:0, RankID:-1) ERR00100 PTA call acl api failed
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
[PID: 697] 2026-05-21-12:39:23.669.889 AclNN_Runtime_Error(EZ9903): aclrtLaunchKernelWithHostArgs failed, return: 507035
        Solution: In this scenario, collect the plog when the fault occurs and locate the fault based on the plog.
        TraceBack (most recent call last):
        The error from device(chipId:0, dieId:0), serial number is 13, there is an aivec error exception, core id is 0, error code = 341, dump info: pc start: 0x120041006000, current: 0x120041006250, sc error info: 0xffffffffffff, su error info: 0xdfc5d7f52dfefffe,0xfebaefc4d800ff3f, mte error info: 0xba9fcfef0003aff1, vec error info: 0x410062d80039187b, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x12004c600378, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:592]
        The extend info: errcode:(341) errorStr: UB access address overflow. subErrType: 0x4.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:595]
        Kernel task happen error, retCode=0x31, [vector core exception].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1530]
        rtStreamSynchronize execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
        stream_id=40, retCode=0x715005e.[FUNC:StreamLaunchKernelV2][FILE:aix_c.cc][LINE:488]
        rtsLaunchKernelWithHostArgs execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
        rtsLaunchKernelWithHostArgs failed, runtime result = 507035.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]
        aclrtLaunchKernelWithHostArgs failed, return: 507035
        Launch kernel failed.
        KernelLaunch failed: /data/pri/Ascend/9.1.0.B010/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe//kernel/ascend950/ops_math/range_apt/Range_b2cb2b97546bcd85efbca38148db9aes_high_performance.o
        Kernel Run failed. opType: 7, Range
        launch failed for Range, errno:361001.

Exception raised from operator() at ../third_party/op-plugin/op_plugin/ops/opapi/ArangeKernelNpuOpApi.cpp:36 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >) + 0x98 (0x7ff8d0fa7218 in /usr/local/python3.11.13/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&) + 0xe0 (0x7ff8d0f3bf72 in /usr/local/python3.11.13/lib/python3.11/site-packages/torch/lib/libc10.so)
frame #2: <unknown function> + 0x168fae1 (0x7ff81068cae1 in /usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so)
frame #3: <unknown function> + 0x40c46e7 (0x7ff8130c16e7 in /usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so)
frame #4: <unknown function> + 0x113ca41 (0x7ff810139a41 in /usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so)
frame #5: <unknown function> + 0x113fa68 (0x7ff81013ca68 in /usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so)
frame #6: <unknown function> + 0x113b747 (0x7ff810138747 in /usr/local/python3.11.13/lib/python3.11/site-packages/torch_npu/lib/libtorch_npu.so)
frame #7: <unknown function> + 0xdc253 (0x7ff8d0dbc253 in /lib/x86_64-linux-gnu/libstdc++.so.6)
frame #8: <unknown function> + 0x94ac3 (0x7ff8e6a25ac3 in /lib/x86_64-linux-gnu/libc.so.6)
frame #9: <unknown function> + 0x126850 (0x7ff8e6ab7850 in /lib/x86_64-linux-gnu/libc.so.6)

Traceback (most recent call last):
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 1221, in not_close_error_metas
    pair.compare()
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 700, in compare
    self._compare_values(actual, expected)
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 830, in _compare_values
    compare_fn(
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 1012, in _compare_regular_values_close
    if torch.all(matches):
RuntimeError: The Inner error is reported as above. The process exits for this inner error, and the current working operator name is aclnnArange.
Since the operator is called asynchronously, the stacktrace may be inaccurate. If you want to get the accurate stacktrace, please set the environment variable ASCEND_LAUNCH_BLOCKING=1.
Note: ASCEND_LAUNCH_BLOCKING=1 will force ops to run in synchronous mode, resulting in performance degradation. Please unset ASCEND_LAUNCH_BLOCKING in time after debugging.
[ERROR] 2026-05-21-12:39:23 (PID:697, Device:0, RankID:-1) ERR00100 PTA call acl api failed.


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/z00910011/tilelang-mlir-ascend/examples/simt/indirect_load_lut2d.py", line 77, in <module>
    main(args.n, args.block, args.num_subspaces, args.codebook_size)
  File "/data/z00910011/tilelang-mlir-ascend/examples/simt/indirect_load_lut2d.py", line 66, in main
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 1497, in assert_close
    error_metas = not_close_error_metas(
                  ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 1228, in not_close_error_metas
    f"Comparing\n\n"
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 367, in __repr__
    body = [
           ^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/testing/_comparison.py", line 368, in <listcomp>
    f"    {name}={value!s},"
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/_tensor.py", line 590, in __repr__
    return torch._tensor_str._str(self, tensor_contents=tensor_contents)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/_tensor_str.py", line 710, in _str
    return _str_intern(self, tensor_contents=tensor_contents)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/_tensor_str.py", line 631, in _str_intern
    tensor_str = _tensor_str(self, indent)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/_tensor_str.py", line 363, in _tensor_str
    formatter = _Formatter(get_summarized_data(self) if summarize else self)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.11.13/lib/python3.11/site-packages/torch/_tensor_str.py", line 145, in __init__
    nonzero_finite_vals = torch.masked_select(
                          ^^^^^^^^^^^^^^^^^^^^
RuntimeError: RunOpApiV2:../torch_npu/csrc/framework/OpCommand.cpp:281 NPU function error: device error type 3, error code is 507035
[ERROR] 2026-05-21-12:39:23 (PID:697, Device:0, RankID:-1) ERR00100 PTA call acl api failed
[Error]: The vector core execution is abnormal. 
        Rectify the fault based on the error information in the ascend log.
EE9999: Inner Error!
EE9999[PID: 697] 2026-05-21-12:39:23.687.932 (EE9999):  rtStreamSynchronizeWithTimeout execution failed, reason=vector core exception[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
        TraceBack (most recent call last):
       synchronize stream with timeout failed, runtime result = 507035[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]