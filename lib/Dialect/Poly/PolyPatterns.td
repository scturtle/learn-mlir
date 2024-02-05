#ifndef DIALECT_POLYPATTERNS_TD
#define DIALECT_POLYPATTERNS_TD

include "Poly.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<
  (Poly_EvalOp $f, (ConjOp $z, $fastmath)),
  (ConjOp (Poly_EvalOp $f, $z), $fastmath)
>;

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

def DifferenceOfSquaresV2 : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;

#endif
