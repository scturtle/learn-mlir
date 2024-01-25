#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace foo {

class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;
  StringRef getArgument() const final { return "affine-full-unroll"; }
};

class AffineFullUnrollPassAsPatternRewrite
    : public PassWrapper<AffineFullUnrollPassAsPatternRewrite,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;
  StringRef getArgument() const final { return "affine-full-unroll-rewrite"; }
};


} // namespace foo
} // namespace mlir
