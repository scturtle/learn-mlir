#ifndef LIB_TRANSFORM_ARITH_PASSES_TD_
#define LIB_TRANSFORM_ARITH_PASSES_TD_

include "mlir/Pass/PassBase.td"

def MulToAdd : Pass<"mul-to-add"> {
}

#endif  // LIB_TRANSFORM_ARITH_PASSES_TD_
