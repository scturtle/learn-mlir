#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Conversion/PolyToStandard/PolyToStandard.h"
#include "Dialect/Poly/PolyDialect.h"
#include "Transform/Affine/AffineFullUnroll.h"
#include "Transform/Arith/Passes.h"

#include "mlir/Pass/PassManager.h"

void polyToLLVMPipelineBuilder(mlir::OpPassManager &manager) {
  manager.addPass(mlir::foo::poly::createPolyToStandard());
  manager.addPass(mlir::createCanonicalizerPass());

  // arith op on tensor to linalg
  manager.addPass(mlir::createConvertElementwiseToLinalgPass());
  // tensor.pad
  manager.addPass(mlir::createConvertTensorToLinalgPass());

  // One-shot bufferize
  // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
  mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  manager.addPass(
      mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
  mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
  mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                       deallocationOptions);

  // memref.subview
  manager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // linalg to scf
  manager.addPass(mlir::createConvertLinalgToLoopsPass());
  // scf to llvm
  manager.addPass(mlir::createConvertSCFToCFPass());
  manager.addPass(mlir::createConvertControlFlowToLLVMPass());
  // arith to llvm
  manager.addPass(mlir::createArithToLLVMConversionPass());

  // cannot be early or later
  manager.addPass(mlir::createConvertFuncToLLVMPass());

  // builtin.unrealized_conversion_cast
  manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  manager.addPass(mlir::createReconcileUnrealizedCastsPass());

  // cleanup
  manager.addPass(mlir::createCanonicalizerPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(mlir::createSymbolDCEPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  registry.insert<mlir::foo::poly::PolyDialect>();

  mlir::PassRegistration<mlir::foo::AffineFullUnrollPass>();
  mlir::PassRegistration<mlir::foo::AffineFullUnrollPassAsPatternRewrite>();
  mlir::foo::registerArithPasses();

  mlir::foo::poly::registerPolyToStandardPasses();

  mlir::PassPipelineRegistration<>(
      "poly-to-llvm", "Run passes to lower the poly dialect to LLVM",
      polyToLLVMPipelineBuilder);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "foo opt\n", registry));
}
