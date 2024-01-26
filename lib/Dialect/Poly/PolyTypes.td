#ifndef DIALECT_POLYTYPES_TD
#define DIALECT_POLYTYPES_TD

include "PolyBase.td"
include "mlir/IR/AttrTypeBase.td"

class Poly_Type<string name, string typeMnemonic> : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Poly_Type<"Polynomial", "poly"> {
  let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";
}

#endif  // DIALECT_POLYTYPES_TD
