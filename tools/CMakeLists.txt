get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
  ${dialect_libs}
  ${conversion_libs}
  MLIROptLib
  AffineFullUnroll
  MulToAdd
  MLIRPoly
  PolyToStandard
)

add_llvm_executable(foo-opt FooOpt.cpp)
llvm_update_compile_flags(foo-opt)
target_link_libraries(foo-opt PRIVATE ${LIBS})
mlir_check_all_link_libraries(foo-opt)
