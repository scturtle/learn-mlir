#ifndef DIALECT_POLYBASE_TD
#define DIALECT_POLYBASE_TD

include "mlir/IR/OpBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let cppNamespace = "::mlir::foo::poly";
  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

#endif // DIALECT_POLYBASE_TD
