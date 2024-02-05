add_mlir_dialect(Poly poly)

set(LLVM_TARGET_DEFINITIONS PolyPatterns.td)
mlir_tablegen(PolyCanonicalize.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRPolyCanonicalizationIncGen)

add_mlir_dialect_library(MLIRPoly
  PolyDialect.cpp
  PolyOps.cpp
)
