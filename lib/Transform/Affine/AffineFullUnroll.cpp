#include "Transform/Affine/AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

namespace mlir {
namespace foo {

void AffineFullUnrollPass::runOnOperation() {
  getOperation().walk([&](AffineForOp op) {
    if (failed(loopUnrollFull(op))) {
      op.emitError("unrolling failed");
      signalPassFailure();
    }
  });
}

struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp> {
  AffineFullUnrollPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AffineForOp>(context, /*benefit=*/1) {}
  LogicalResult matchAndRewrite(AffineForOp op,
                                PatternRewriter &rewriter) const override {
    return loopUnrollFull(op);
  }
};

void AffineFullUnrollPassAsPatternRewrite::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<AffineFullUnrollPattern>(&getContext());
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace foo
} // namespace mlir
