// RUN: bishengir-opt %s --one-shot-bufferize="bufferize-function-boundaries" --convert-tensor-view-to-template --canonicalize | FileCheck %s --implicit-check-not=tv.
func.func @pointer_access(%pointers: tensor<8xi64>, %indices: tensor<4xindex>) {
  %t = tv.ptr_load %pointers, indices = [%indices] : tensor<8xi64>, tensor<4xindex> -> tensor<4xf32>
  tv.ptr_store %pointers, %t, indices = [%indices] : tensor<8xi64>, tensor<4xf32>, tensor<4xindex>
  return
}
// CHECK-LABEL: func.func @pointer_access
// CHECK: call @_mlir_ciface_tensor_view_ptr_load_r1
// CHECK: call @_mlir_ciface_tensor_view_ptr_store_r1
// CHECK: func.func private @_mlir_ciface_tensor_view_ptr_load_r1
// CHECK-SAME: hacc.always_inline
// CHECK-NOT: llvm.emit_c_interface
