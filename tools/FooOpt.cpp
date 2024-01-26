#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Dialect/Poly/PolyDialect.h"
#include "Transform/Affine/AffineFullUnroll.h"
#include "Transform/Arith/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  registry.insert<mlir::foo::poly::PolyDialect>();

  mlir::PassRegistration<mlir::foo::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::foo::AffineFullUnrollPassAsPatternRewrite>();
  mlir::foo::registerArithPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "foo opt\n", registry));
}
