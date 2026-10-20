// RUN: bishengir-opt %s --triton-global-kernel-args-to-hivm-op --one-shot-bufferize="bufferize-function-boundaries" --convert-tensor-view-to-template --canonicalize | FileCheck %s --implicit-check-not=tv.
#view = #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>
func.func @kernel(%base: !tv.ptr<f32>, %n: index, %stride: index, %nx: i32, %ny: i32, %nz: i32, %ix: i32, %iy: i32, %iz: i32) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
  %i = arith.index_cast %ix : i32 to index
  %v = tv.make_tensor_view %base, sizes = [%n], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
  %p = tv.make_partition_view %v : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #view>
  %t = tv.view_load %p[%i] : !tv.tensor_view<?xf32, strides=[?], #view>, index -> tensor<4xf32>
  tv.view_store %p[%i], %t : !tv.tensor_view<?xf32, strides=[?], #view>, tensor<4xf32>, index
  return
}
// CHECK-LABEL: func.func @kernel
// CHECK-SAME: memref<?xf32, #hivm.address_space<gm>>
// CHECK-SAME: func_dyn_memref_args = dense<[true, false, false, false, false, false]> : vector<6xi1>
// CHECK: call @_mlir_ciface_tensor_view_load_r1
// CHECK: call @_mlir_ciface_tensor_view_store_r1
