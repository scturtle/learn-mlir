#ifndef DIALECT_POLYOPS_TD
#define DIALECT_POLYOPS_TD

include "PolyBase.td"
include "PolyTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def PolyOrContainer : TypeOrContainer<Polynomial, "poly-or-container">;

class Poly_BinOp<string mnemoic> : Op<Poly_Dialect, mnemoic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($output)";
}

def Poly_AddOp : Poly_BinOp<"add"> {
}

def Poly_SubOp : Poly_BinOp<"sub"> {
}

def Poly_MulOp : Poly_BinOp<"mul"> {
}

def Poly_FromTensorOp : Op<Poly_Dialect, "from_tensor", [Pure]> {
  let arguments = (ins TensorOf<[AnyInteger]>:$input);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
}

def Poly_EvalOp : Op<Poly_Dialect, "eval"> {
  let arguments = (ins Polynomial:$input, AnyInteger:$point);
  let results = (outs AnyInteger:$output);
  let assemblyFormat = "$input `,` $point attr-dict `:` `(` type($input) `,` type($point) `)` `->` type($output)";
}

#endif // DIALECT_POLYOPS_TD
