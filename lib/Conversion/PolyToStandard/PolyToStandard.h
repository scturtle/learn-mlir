#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace foo {
namespace poly {

#define GEN_PASS_DECL
#include "Conversion/PolyToStandard/PolyToStandard.h.inc"

#define GEN_PASS_REGISTRATION
#include "Conversion/PolyToStandard/PolyToStandard.h.inc"

} // namespace poly
} // namespace foo
} // namespace mlir
