// RUN: bishengir-opt %s --one-shot-bufferize="bufferize-function-boundaries" | FileCheck %s --check-prefix=BUFFER
// RUN: bishengir-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-tensor-view-to-template --canonicalize | FileCheck %s --check-prefix=CALL --implicit-check-not=tv.

#partition = #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>
#strided = #tv.strided_view<tile = [4], dim_map = [0], traversal_strides = [8], padding_value = zero>
#gather = #tv.gather_scatter_view<tile = [4], sparse_dim = [0], padding_value = zero>

func.func @partition(%base: !tv.ptr<f32>, %size: index, %stride: index, %i: index) {
  %v = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
  %p = tv.make_partition_view %v : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #partition>
  %t = tv.view_load %p[%i] : !tv.tensor_view<?xf32, strides=[?], #partition>, index -> tensor<4xf32>
  tv.view_store %p[%i], %t : !tv.tensor_view<?xf32, strides=[?], #partition>, tensor<4xf32>, index
  return
}
// BUFFER-LABEL: func.func @partition
// BUFFER: tv.make_tensor_view
// BUFFER: tv.make_partition_view
// BUFFER: memref.alloc
// BUFFER: tv.view_load_into
// BUFFER: tv.view_store_from
// CALL-LABEL: func.func @partition
// CALL-SAME: memref<?xf32, #hivm.address_space<gm>>
// CALL: call @_mlir_ciface_tensor_view_load_r1
// CALL: call @_mlir_ciface_tensor_view_store_r1

func.func @strided(%base: !tv.ptr<f32>, %size: index, %stride: index, %i: index) {
  %v = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
  %p = tv.make_strided_view %v : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #strided>
  %t = tv.view_load %p[%i] : !tv.tensor_view<?xf32, strides=[?], #strided>, index -> tensor<4xf32>
  tv.view_store %p[%i], %t : !tv.tensor_view<?xf32, strides=[?], #strided>, tensor<4xf32>, index
  return
}
// BUFFER-LABEL: func.func @strided
// BUFFER: traversal_strides = [8]
// BUFFER: tv.view_load_into
// BUFFER: tv.view_store_from
// CALL-LABEL: func.func @strided
// CALL: arith.constant 8 : i64
// CALL: call @_mlir_ciface_tensor_view_load_r1
// CALL: call @_mlir_ciface_tensor_view_store_r1

func.func @gather(%base: !tv.ptr<f32>, %size: index, %stride: index, %indices: tensor<4xi32>) {
  %v = tv.make_tensor_view %base, sizes = [%size], strides = [%stride] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
  %p = tv.make_gather_scatter_view %v : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #gather>
  %t = tv.view_load %p[%indices] : !tv.tensor_view<?xf32, strides=[?], #gather>, tensor<4xi32> -> tensor<4xf32>
  tv.view_store %p[%indices], %t : !tv.tensor_view<?xf32, strides=[?], #gather>, tensor<4xf32>, tensor<4xi32>
  return
}
// BUFFER-LABEL: func.func @gather
// BUFFER: sparse_dim = [0]
// BUFFER: tv.view_load_into
// BUFFER: tv.view_store_from
// CALL-LABEL: func.func @gather
// CALL-SAME: memref<4xi32
// CALL: memref.extract_aligned_pointer_as_index
// CALL: call @_mlir_ciface_tensor_view_load_r1
// CALL: call @_mlir_ciface_tensor_view_store_r1
