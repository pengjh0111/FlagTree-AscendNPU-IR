// RUN: bishengir-opt %s --convert-tensor-view-to-template --verify-diagnostics
#view = #tv.partition_view<tile = [4], dim_map = [0], padding_value = zero>
func.func @unbufferized(%base: !tv.ptr<f32>, %n: index, %s: index, %i: index) -> tensor<4xf32> {
  %v = tv.make_tensor_view %base, sizes = [%n], strides = [%s] : !tv.ptr<f32> -> !tv.tensor_view<?xf32, strides=[?]>
  %p = tv.make_partition_view %v : !tv.tensor_view<?xf32, strides=[?]> -> !tv.tensor_view<?xf32, strides=[?], #view>
  // expected-error @+1 {{run one-shot-bufferize before TensorView template lowering}}
  %t = tv.view_load %p[%i] : !tv.tensor_view<?xf32, strides=[?], #view>, index -> tensor<4xf32>
  return %t : tensor<4xf32>
}
