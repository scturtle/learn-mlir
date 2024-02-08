set(LLVM_TARGET_DEFINITIONS PolyToStandard.td)
mlir_tablegen(PolyToStandard.h.inc -gen-pass-decls -name PolyToStandard)
add_public_tablegen_target(PolyToStandardPassIncGen)
add_dependencies(mlir-headers PolyToStandardPassIncGen)

add_mlir_library(PolyToStandard
  PolyToStandard.cpp

  DEPENDS
  PolyToStandardPassIncGen

  LINK_LIBS PUBLIC
  MLIRPoly
  MLIRArithDialect
  MLIRTensorDialect
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRPass
  MLIRTransforms
)
