//===- BufferizableOpInterfaceImpl.cpp ------------------------------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Conversion/TensorViewToTemplate/TensorViewToTemplate.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"

using namespace mlir;
using namespace mlir::bufferization;
namespace tv = mlir::triton::tv;

namespace {
static MemRefType getTileBufferType(RankedTensorType type) {
  auto space =
      hivm::AddressSpaceAttr::get(type.getContext(), hivm::AddressSpace::UB);
  return MemRefType::get(type.getShape(), type.getElementType(),
                         MemRefLayoutAttrInterface{}, space);
}

template <typename OpTy, typename BufferOpTy, bool IsLoad>
struct AccessModel : BufferizableOpInterface::ExternalModel<
                         AccessModel<OpTy, BufferOpTy, IsLoad>, OpTy> {
  bool bufferizesToAllocation(Operation *, Value) const { return IsLoad; }

  FailureOr<BaseMemRefType> getBufferType(Operation *, Value value,
                                          const BufferizationOptions &,
                                          SmallVector<Value> &) const {
    return getTileBufferType(cast<RankedTensorType>(value.getType()));
  }

  bool bufferizesToMemoryRead(Operation *, OpOperand &,
                              const AnalysisState &) const {
    return true;
  }
  bool bufferizesToMemoryWrite(Operation *, OpOperand &,
                               const AnalysisState &) const {
    return false;
  }
  AliasingValueList getAliasingValues(Operation *, OpOperand &,
                                      const AnalysisState &) const {
    return {};
  }
  LogicalResult bufferize(Operation *operation, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto op = cast<OpTy>(operation);
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      if (isa<TensorType>(operand.getType())) {
        auto buffer = getBuffer(rewriter, operand, options);
        if (failed(buffer))
          return failure();
        operands.push_back(*buffer);
      } else {
        operands.push_back(operand);
      }
    }
    Value buffer;
    if constexpr (IsLoad) {
      auto type = cast<RankedTensorType>(op->getResult(0).getType());
      if (!type.hasStaticShape())
        return op.emitError("TensorView load requires a static tile shape");
      auto allocation = options.createAlloc(rewriter, op.getLoc(),
                                            getTileBufferType(type), {});
      if (failed(allocation))
        return failure();
      buffer = *allocation;
      operands.insert(operands.begin() + 1, buffer);
    }
    OperationState operationState(op.getLoc(), BufferOpTy::getOperationName());
    operationState.addOperands(operands);
    operationState.addAttributes(op->getAttrs());
    rewriter.create(operationState);
    if constexpr (IsLoad)
      replaceOpWithBufferizedValues(rewriter, op, buffer);
    else
      rewriter.eraseOp(op);
    return success();
  }
};

template <typename OpTy>
struct VectorCoreModel
    : hivm::CoreTypeInterface::ExternalModel<VectorCoreModel<OpTy>, OpTy> {
  std::optional<hivm::TCoreType> getCoreType(Operation *) const {
    return hivm::TCoreType::VECTOR;
  }
};

template <typename... Ops> void attachCoreModels(MLIRContext *context) {
  (Ops::template attachInterface<VectorCoreModel<Ops>>(*context), ...);
}
} // namespace

void tv::registerTensorViewBackendInterfaces(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, tv::TensorViewDialect *) {
    context->getOrLoadDialect<hivm::HIVMDialect>();
    tv::ViewLoadOp::attachInterface<
        AccessModel<tv::ViewLoadOp, tv::ViewLoadIntoOp, true>>(*context);
    tv::ViewStoreOp::attachInterface<
        AccessModel<tv::ViewStoreOp, tv::ViewStoreFromOp, false>>(*context);
    tv::PtrLoadOp::attachInterface<
        AccessModel<tv::PtrLoadOp, tv::PtrLoadIntoOp, true>>(*context);
    tv::PtrStoreOp::attachInterface<
        AccessModel<tv::PtrStoreOp, tv::PtrStoreFromOp, false>>(*context);
    attachCoreModels<tv::ViewLoadOp, tv::ViewStoreOp, tv::PtrLoadOp,
                     tv::PtrStoreOp, tv::ViewLoadIntoOp, tv::ViewStoreFromOp,
                     tv::PtrLoadIntoOp, tv::PtrStoreFromOp>(context);
  });
}
