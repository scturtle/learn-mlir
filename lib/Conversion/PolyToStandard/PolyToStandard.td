#ifndef LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD
#define LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD

include "mlir/Pass/PassBase.td"

def PolyToStandard : Pass<"poly-to-standard"> {
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
    "mlir::foo::poly::PolyDialect",
  ];
}

#endif
