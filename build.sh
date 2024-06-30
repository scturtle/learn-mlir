#!/bin/bash

if [ -d externals/llvm-project/build ]; then
  LLVM_DIR=externals/llvm-project/build
else
  LLVM_DIR=`brew --prefix llvm`
  LLVM_LIT=`which lit`
fi

cmake -H. -Bbuild -GNinja \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER=clang \
    -DLLVM_DIR="$LLVM_DIR/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" \
    -DLLVM_EXTERNAL_LIT=$LLVM_LIT \
    -DCMAKE_BUILD_TYPE=Debug

cmake --build build --target check-foo
