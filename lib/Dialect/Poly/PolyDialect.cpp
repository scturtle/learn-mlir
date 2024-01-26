#include "Dialect/Poly/PolyDialect.h"
#include "Dialect/Poly/PolyTypes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Dialect/Poly/PolyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "Dialect/Poly/PolyTypes.cpp.inc"

namespace mlir {
namespace foo {
namespace poly {

void PolyDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Poly/PolyTypes.cpp.inc"
      >();
}

} // namespace poly
} // namespace foo
} // namespace mlir
