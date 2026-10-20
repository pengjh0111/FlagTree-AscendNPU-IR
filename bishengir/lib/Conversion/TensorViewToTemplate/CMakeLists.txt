add_bishengir_conversion_library(BiShengIRTensorViewToTemplate
  TensorViewToTemplate.cpp
  BufferizableOpInterfaceImpl.cpp

  DEPENDS
  BiShengIRConversionPassIncGen

  LINK_LIBS PUBLIC
  TensorViewIR
  BiShengIRHIVMDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRTransforms
)
