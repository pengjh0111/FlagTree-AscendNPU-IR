//===- TensorViewToTemplate.cpp - TensorView library calls
//-----------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Conversion/TensorViewToTemplate/TensorViewToTemplate.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Dialect/TensorView/IR/TensorViewDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORVIEWTOTEMPLATE
#include "bishengir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace tv = mlir::triton::tv;

namespace {
// The template ABI is entirely i64. Each rank adds ten descriptor fields per
// dimension. Addresses include the memref offset; strides remain in elements.
static Value constant(OpBuilder &builder, Location loc, int64_t value) {
  return builder.create<arith::ConstantIntOp>(loc, value, 64);
}

static Value toI64(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isInteger(64))
    return value;
  return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), value);
}

static int64_t elementCode(Type type) {
  if (type.isF16())
    return 16;
  if (type.isBF16())
    return 17;
  if (type.isF32())
    return 32;
  if (type.isF64())
    return 64;
  if (auto integer = dyn_cast<IntegerType>(type))
    return 128 + integer.getWidth();
  return 0;
}

static Value address(OpBuilder &builder, Location loc, Value memref) {
  auto type = cast<MemRefType>(memref.getType());
  auto metadata = builder.create<memref::ExtractStridedMetadataOp>(loc, memref);
  Value base =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memref);
  int64_t bytes = type.getElementType().isIndex()
                      ? 8
                      : llvm::divideCeil(type.getElementTypeBitWidth(), 8);
  Value byteOffset = builder.create<arith::MulIOp>(
      loc, toI64(builder, loc, metadata.getOffset()),
      constant(builder, loc, bytes));
  return builder.create<arith::AddIOp>(loc, toI64(builder, loc, base),
                                       byteOffset);
}

static LogicalResult createCall(Operation *op, OpBuilder &builder,
                                StringRef name, ValueRange operands) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto type = builder.getFunctionType(operands.getTypes(), {});
  auto function = module.lookupSymbol<func::FuncOp>(name);
  if (function && function.getFunctionType() != type)
    return op->emitError("inconsistent TensorView template signature");
  if (!function) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    function = builder.create<func::FuncOp>(op->getLoc(), name, type);
    function.setPrivate();
    // Scalar arguments already have the C ABI; no memref wrapper is needed.
    function->setAttr("hacc.always_inline", builder.getUnitAttr());
    function->setAttr(hivm::TFuncCoreTypeAttr::name,
                      hivm::TFuncCoreTypeAttr::get(builder.getContext(),
                                                   hivm::TFuncCoreType::AIV));
  }
  builder.create<func::CallOp>(op->getLoc(), function, operands);
  return success();
}

static LogicalResult lowerView(Operation *op, Value view, Value buffer,
                               ValueRange indices, bool isLoad) {
  auto viewType = cast<tv::TensorViewType>(view.getType());
  auto encoded =
      dyn_cast_or_null<tv::TensorViewEncodingOpInterface>(view.getDefiningOp());
  if (!encoded)
    return op->emitError("expected a directly constructed encoded view");
  auto base = encoded.getSource().getDefiningOp<tv::MakeTensorViewOp>();
  if (!base)
    return op->emitError("expected make_tensor_view as the view source");
  Value pointer = base.getSource();
  while (auto cast = pointer.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != 1)
      break;
    pointer = cast.getInputs().front();
  }
  if (!isa<MemRefType>(pointer.getType()))
    return op->emitError("view base must be a converted function pointer");
  int64_t rank = viewType.getRank();
  if (rank < 1 || rank > 4 || indices.size() != static_cast<size_t>(rank))
    return op->emitError("TensorView template stubs support ranks 1 through 4");
  int64_t code = elementCode(viewType.getElementType());
  if (!code)
    return op->emitError("unsupported TensorView template element type");

  OpBuilder builder(op);
  Location loc = op->getLoc();
  Attribute encoding = viewType.getEncoding();
  auto tile = tv::getEncodingTileShape(encoding);
  auto traversal = tv::getEncodingTraversal(encoding);
  auto sparse = tv::getEncodingSparseDims(encoding);
  SmallVector<int64_t> dimMap;
  if (auto partition = dyn_cast<tv::PartitionViewAttr>(encoding))
    dimMap.assign(partition.getDimMap().begin(), partition.getDimMap().end());
  else if (auto strided = dyn_cast<tv::StridedViewAttr>(encoding))
    dimMap.assign(strided.getDimMap().begin(), strided.getDimMap().end());
  else
    for (int64_t dim = 0; dim < rank; ++dim)
      dimMap.push_back(dim);
  int64_t sparseMask = 0;
  for (int64_t dim : sparse)
    sparseMask |= int64_t{1} << dim;
  SmallVector<Value> arguments{
      address(builder, loc, pointer), address(builder, loc, buffer),
      constant(builder, loc, code),
      constant(builder, loc,
               static_cast<int64_t>(tv::getEncodingPaddingValue(encoding))),
      constant(builder, loc, sparseMask)};
  auto local = builder.create<memref::ExtractStridedMetadataOp>(loc, buffer);
  for (int64_t dim = 0; dim < rank; ++dim) {
    Value coordinate = constant(builder, loc, 0);
    Value indexAddress = constant(builder, loc, 0);
    Value indexStride = constant(builder, loc, 0);
    Value indexBits = constant(builder, loc, 0);
    if (llvm::is_contained(sparse, dim)) {
      auto type = dyn_cast<MemRefType>(indices[dim].getType());
      if (!type || type.getRank() != 1)
        return op->emitError(
            "sparse coordinates must be bufferized rank-1 values");
      indexAddress = address(builder, loc, indices[dim]);
      auto metadata =
          builder.create<memref::ExtractStridedMetadataOp>(loc, indices[dim]);
      indexStride = toI64(builder, loc, metadata.getStrides().front());
      indexBits = constant(
          builder, loc,
          type.getElementType().isIndex() ? 64 : type.getElementTypeBitWidth());
    } else {
      coordinate = toI64(builder, loc, indices[dim]);
    }
    // dim_map selects the base dimension for partition/strided coordinates.
    arguments.append({toI64(builder, loc, base.getSizes()[dimMap[dim]]),
                      toI64(builder, loc, base.getStrides()[dimMap[dim]]),
                      constant(builder, loc, tile[dim]),
                      constant(builder, loc, traversal[dim]),
                      toI64(builder, loc, local.getSizes()[dim]),
                      toI64(builder, loc, local.getStrides()[dim]), coordinate,
                      indexAddress, indexStride, indexBits});
  }
  std::string name = (Twine("_mlir_ciface_tensor_view_") +
                      (isLoad ? "load_r" : "store_r") + Twine(rank))
                         .str();
  if (failed(createCall(op, builder, name, arguments)))
    return failure();
  op->erase();
  return success();
}

static LogicalResult lowerPointers(Operation *op, Value pointers, Value buffer,
                                   ValueRange indices, bool isLoad) {
  auto type = cast<MemRefType>(pointers.getType());
  int64_t rank = type.getRank();
  if (!type.getElementType().isInteger(64) || rank < 1 || rank > 4 ||
      indices.size() != static_cast<size_t>(rank))
    return op->emitError(
        "pointer template requires rank-1 through rank-4 i64 addresses");
  auto bufferType = cast<MemRefType>(buffer.getType());
  if (bufferType.getRank() != 1 || !elementCode(bufferType.getElementType()))
    return op->emitError(
        "pointer template requires a supported rank-1 data buffer");
  for (Value index : indices) {
    auto coordinateType = cast<MemRefType>(index.getType());
    if (coordinateType.getRank() != 1 ||
        !coordinateType.getElementType().isIndex())
      return op->emitError("pointer coordinates must be rank-1 index buffers");
  }
  OpBuilder builder(op);
  Location loc = op->getLoc();
  auto metadata =
      builder.create<memref::ExtractStridedMetadataOp>(loc, pointers);
  auto data = builder.create<memref::ExtractStridedMetadataOp>(loc, buffer);
  SmallVector<Value> arguments{
      address(builder, loc, pointers), address(builder, loc, buffer),
      constant(
          builder, loc,
          elementCode(cast<MemRefType>(buffer.getType()).getElementType())),
      toI64(builder, loc, data.getSizes().front()),
      toI64(builder, loc, data.getStrides().front())};
  for (auto [dim, index] : llvm::enumerate(indices)) {
    auto coordinate =
        builder.create<memref::ExtractStridedMetadataOp>(loc, index);
    arguments.append({toI64(builder, loc, metadata.getSizes()[dim]),
                      toI64(builder, loc, metadata.getStrides()[dim]),
                      address(builder, loc, index),
                      toI64(builder, loc, coordinate.getStrides().front())});
  }
  std::string name = (Twine("_mlir_ciface_tensor_view_ptr_") +
                      (isLoad ? "load_r" : "store_r") + Twine(rank))
                         .str();
  if (failed(createCall(op, builder, name, arguments)))
    return failure();
  op->erase();
  return success();
}

struct ConvertTensorViewToTemplatePass
    : impl::ConvertTensorViewToTemplateBase<ConvertTensorViewToTemplatePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult unbufferized = module.walk([](Operation *op) {
      if (isa<tv::ViewLoadOp, tv::ViewStoreOp, tv::PtrLoadOp, tv::PtrStoreOp>(
              op)) {
        op->emitError(
            "run one-shot-bufferize before TensorView template lowering");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (unbufferized.wasInterrupted())
      return signalPassFailure();
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](tv::PtrType type) -> Type {
      return MemRefType::get(
          {ShapedType::kDynamic}, type.getPointeeType(),
          MemRefLayoutAttrInterface{},
          hivm::AddressSpaceAttr::get(&getContext(), hivm::AddressSpace::GM));
    });
    converter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });
    converter.addTargetMaterialization([](OpBuilder &builder, Type type,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    });
    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType()) &&
             converter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation *op) { return converter.isLegal(op); });
    // Structural conversions use the same target as function conversion.
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns,
                                                         target);
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    SmallVector<Operation *> accesses;
    module.walk([&](Operation *op) {
      if (isa<tv::ViewLoadIntoOp, tv::ViewStoreFromOp, tv::PtrLoadIntoOp,
              tv::PtrStoreFromOp>(op))
        accesses.push_back(op);
    });
    for (Operation *op : accesses) {
      LogicalResult result =
          TypeSwitch<Operation *, LogicalResult>(op)
              .Case<tv::ViewLoadIntoOp>([](auto access) {
                return lowerView(access, access.getView(), access.getBuffer(),
                                 access.getIndices(), true);
              })
              .Case<tv::ViewStoreFromOp>([](auto access) {
                return lowerView(access, access.getView(), access.getBuffer(),
                                 access.getIndices(), false);
              })
              .Case<tv::PtrLoadIntoOp>([](auto access) {
                return lowerPointers(access, access.getPointers(),
                                     access.getBuffer(), access.getIndices(),
                                     true);
              })
              .Case<tv::PtrStoreFromOp>([](auto access) {
                return lowerPointers(access, access.getPointers(),
                                     access.getBuffer(), access.getIndices(),
                                     false);
              });
      if (failed(result))
        return signalPassFailure();
    }
    RewritePatternSet cleanup(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(cleanup))))
      return signalPassFailure();
    WalkResult remaining = module.walk([](Operation *op) {
      if (op->getName().getDialectNamespace() == "tv") {
        op->emitError(
            "TensorView template lowering left an unsupported operation");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (remaining.wasInterrupted())
      return signalPassFailure();
    module.walk([&](Operation *op) {
      auto checkType = [&](Type type) {
        type.walk([&](Type nested) {
          if (nested.getDialect().getNamespace() == "tv") {
            op->emitError("TensorView type remains after template lowering");
            signalPassFailure();
          }
        });
      };
      for (Type type : op->getOperandTypes())
        checkType(type);
      for (Type type : op->getResultTypes())
        checkType(type);
      if (auto function = dyn_cast<func::FuncOp>(op))
        checkType(function.getFunctionType());
    });
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTensorViewToTemplatePass() {
  return std::make_unique<ConvertTensorViewToTemplatePass>();
}
