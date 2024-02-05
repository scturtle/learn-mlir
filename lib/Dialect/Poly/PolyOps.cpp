#include "Dialect/Poly/PolyOps.h"

#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/PatternMatch.h"

#include "Dialect/Poly/PolyCanonicalize.cpp.inc"

namespace mlir {
namespace foo {
namespace poly {

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}

OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt>(
      adaptor.getOperands(), [](APInt a, APInt b) { return a + b; });
}

OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt>(
      adaptor.getOperands(), [](APInt a, APInt b) { return a - b; });
}

OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[0]);
  auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[1]);
  if (!lhs || !rhs)
    return nullptr;

  auto degree = getResult().getType().cast<PolynomialType>().getDegreeBound();
  auto maxIndex = lhs.size() + rhs.size() - 1;
  SmallVector<APInt, 8> result;
  result.reserve(maxIndex);
  for (int i = 0; i < maxIndex; ++i) {
    result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
  }

  int i = 0;
  for (auto lhsVal : lhs.getValues<APInt>()) {
    int j = 0;
    for (auto rhsVal : rhs.getValues<APInt>()) {
      result[(i + j) % degree] += rhsVal * lhsVal;
      ++j;
    }
    ++i;
  }

  return DenseIntElementsAttr::get(
      RankedTensorType::get(static_cast<int64_t>(result.size()),
                            IntegerType::get(getContext(), 32)),
      result);
}

// (x^2 - y^2) => (x+y)(x-y)
struct DifferenceOfSquares : public OpRewritePattern<SubOp> {
  DifferenceOfSquares(mlir::MLIRContext *context)
      : OpRewritePattern<SubOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(SubOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
      return failure();
    }
    auto rhsMul = rhs.getDefiningOp<MulOp>();
    auto lhsMul = lhs.getDefiningOp<MulOp>();
    if (!rhsMul || !lhsMul) {
      return failure();
    }
    if (rhsMul.getLhs() != rhsMul.getRhs() ||
        lhsMul.getLhs() != lhsMul.getRhs()) {
      return failure();
    }
    auto x = lhsMul.getLhs(), y = rhsMul.getLhs();
    AddOp newAdd = rewriter.create<AddOp>(op.getLoc(), x, y);
    SubOp newSub = rewriter.create<SubOp>(op.getLoc(), x, y);
    MulOp newMul = rewriter.create<MulOp>(op.getLoc(), newAdd, newSub);
    rewriter.replaceOp(op, newMul);
    return success();
  }
};

void AddOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context) {}

void SubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context) {
  // results.add<DifferenceOfSquares>(context);
  results.add<DifferenceOfSquaresV2>(context);
}

void MulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                        ::mlir::MLIRContext *context) {}

OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor) {
  return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
}

LogicalResult EvalOp::verify() {
  auto pointTy = getPoint().getType();
  bool isSignlessInteger = pointTy.isSignlessInteger(32);
  auto complexPt = llvm::dyn_cast<ComplexType>(pointTy);
  return isSignlessInteger || complexPt
             ? success()
             : emitOpError("argument point must be a 32-bit "
                           "integer, or a complex number");
}

void EvalOp::getCanonicalizationPatterns(::mlir::RewritePatternSet &results,
                                         ::mlir::MLIRContext *context) {
  // populateWithGenerated(results);
  results.add<LiftConjThroughEval>(context);
}

} // namespace poly
} // namespace foo
} // namespace mlir
