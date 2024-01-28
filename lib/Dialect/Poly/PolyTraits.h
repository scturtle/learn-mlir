#pragma once

#include "mlir/IR/OpDefinition.h"

namespace mlir::foo::poly {

template <typename ConcreteType>
class Has32BitArguments
    : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      if (!type.isIntOrIndex())
        continue;
      if (!type.isInteger(32)) {
        return op->emitOpError()
               << "requires each numeric operand to be a 32-bit integer";
      }
    }
    return success();
  }
};

} // namespace mlir::foo::poly
