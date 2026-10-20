//===- TensorViewToTemplate.h - TensorView library lowering -------*- C++
//-*-===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BISHENGIR_CONVERSION_TENSORVIEWTOTEMPLATE_H
#define BISHENGIR_CONVERSION_TENSORVIEWTOTEMPLATE_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class DialectRegistry;
class ModuleOp;
#define GEN_PASS_DECL_CONVERTTENSORVIEWTOTEMPLATE
#include "bishengir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTensorViewToTemplatePass();

namespace triton::tv {
void registerTensorViewBackendInterfaces(DialectRegistry &registry);
} // namespace triton::tv
} // namespace mlir

#endif
