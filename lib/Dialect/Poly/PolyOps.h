#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/Poly/PolyTraits.h"
#include "Dialect/Poly/PolyTypes.h"

#define GET_OP_CLASSES
#include "Dialect/Poly/Poly.h.inc"
