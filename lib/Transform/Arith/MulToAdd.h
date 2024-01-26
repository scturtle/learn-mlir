#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace foo {

#define GEN_PASS_DECL_MULTOADD
#include "Transform/Arith/Passes.h.inc"

}
} // namespace mlir
