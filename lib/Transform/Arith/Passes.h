#pragma once
#include "Transform/Arith/MulToAdd.h"

namespace mlir {
namespace foo {

#define GEN_PASS_REGISTRATION
#include "Transform/Arith/Passes.h.inc"

} // namespace foo
} // namespace mlir
