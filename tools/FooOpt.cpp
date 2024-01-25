#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Transform/Affine/AffineFullUnroll.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::PassRegistration<mlir::foo::AffineFullUnrollPass>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "foo opt\n", registry));
}
