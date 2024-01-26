#ifndef DIALECT_POLYBASE_TD
#define DIALECT_POLYBASE_TD

include "mlir/IR/OpBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let cppNamespace = "::mlir::foo::poly";
}

#endif // DIALECT_POLYBASE_TD
