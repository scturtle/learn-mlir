#!/bin/sh

THIRDPARTY_LLVM_DIR=$PWD/externals/llvm-project
LLVM_DIR=$THIRDPARTY_LLVM_DIR/llvm
BUILD_DIR=$THIRDPARTY_LLVM_DIR/build
INSTALL_DIR=$THIRDPARTY_LLVM_DIR/install

cmake -H$LLVM_DIR -B$BUILD_DIR -GNinja \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DLLVM_LOCAL_RPATH=$INSTALL_DIR/lib \
      -DLLVM_BUILD_EXAMPLES=OFF \
      -DLLVM_INSTALL_UTILS=ON \
      -DCMAKE_BUILD_TYPE=Debug \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_CCACHE_BUILD=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DLLVM_ENABLE_PROJECTS='mlir' \
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64" \
      -DCMAKE_OSX_ARCHITECTURES="$(uname -m)" \
      -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)"

cmake --build $BUILD_DIR --target check-mlir
