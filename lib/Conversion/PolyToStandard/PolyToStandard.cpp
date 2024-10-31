#include "Conversion/PolyToStandard/PolyToStandard.h"

#include "Dialect/Poly/PolyDialect.h"
#include "Dialect/Poly/PolyOps.h"
#include "Dialect/Poly/PolyTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace foo {
namespace poly {

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "Conversion/PolyToStandard/PolyToStandard.h.inc"

class PolyToStandardTypeConverter : public TypeConverter {
public:
  PolyToStandardTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](PolynomialType type) -> Type {
      int degreeBound = type.getDegreeBound();
      IntegerType elementTy =
          IntegerType::get(ctx, 32, IntegerType::SignednessSemantics::Signless);
      return RankedTensorType::get({degreeBound}, elementTy);
    });
    // Convert from a tensor type to a poly type: use from_tensor
    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<poly::FromTensorOp>(loc, type, inputs[0]);
    });

    // Convert from a poly type to a tensor type: use to_tensor
    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<poly::ToTensorOp>(loc, type, inputs[0]);
    });
  }
};

struct ConvertAdd : public OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  ConvertSub(mlir::MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, adaptor.getLhs(),
                                               adaptor.getRhs());
    return success();
  }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp> {
  ConvertFromTensor(mlir::MLIRContext *context)
      : OpConversionPattern<FromTensorOp>(context) {}
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(FromTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outputTensorTy = cast<RankedTensorType>(
        typeConverter->convertType(op.getResult().getType()));
    auto outputShape = outputTensorTy.getShape()[0];
    auto inputShape = op.getInput().getType().getShape()[0];
    // Zero pad the tensor
    auto coeff = adaptor.getInput();
    if (inputShape < outputShape) {
      SmallVector<OpFoldResult> low = {rewriter.getIndexAttr(0)},
                                high = {rewriter.getIndexAttr(outputShape -
                                                              inputShape)};
      auto zero = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(outputTensorTy.getElementType(), 0));
      coeff = rewriter.create<tensor::PadOp>(op.getLoc(), outputTensorTy, coeff,
                                             low, high, zero,
                                             /*nofold=*/false);
    }
    rewriter.replaceOp(op, coeff);
    return success();
  }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp> {
  ConvertToTensor(mlir::MLIRContext *context)
      : OpConversionPattern<ToTensorOp>(context) {}
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ToTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  ConvertConstant(mlir::MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto constOp = rewriter.create<arith::ConstantOp>(
        op.getLoc(), adaptor.getCoefficients());
    rewriter.replaceOpWithNewOp<poly::FromTensorOp>(
        op, op.getResult().getType(), constOp);
    return success();
  }
};

struct ConvertMul : public OpConversionPattern<MulOp> {
  ConvertMul(mlir::MLIRContext *context)
      : OpConversionPattern<MulOp>(context) {}
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto mulTensorType = cast<RankedTensorType>(adaptor.getLhs().getType());
    auto numTerms = mulTensorType.getShape()[0];
    // Create an all-zeros tensor to store the result
    auto polymulResult = b.create<arith::ConstantOp>(
        mulTensorType, DenseElementsAttr::get(mulTensorType, 0));
    // Loop bounds and step.
    auto low = b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(0));
    auto num =
        b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(numTerms));
    auto step =
        b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));
    // for i = 0, ..., N-1
    //   for j = 0, ..., N-1
    //     product[i+j (mod N)] += p0[i] * p1[j]
    auto outerLoop = b.create<scf::ForOp>(
        low, num, step, ValueRange{polymulResult},
        [&](OpBuilder &builder, Location loc, Value idx0,
            ValueRange loopState) {
          auto innerLoop = builder.create<scf::ForOp>(
              op.getLoc(), low, num, step, loopState,
              [&](OpBuilder &builder, Location loc, Value idx1,
                  ValueRange loopState) {
                ImplicitLocOpBuilder b(op.getLoc(), builder);
                Value accumTensor = loopState.front();
                Value destIndex = b.create<arith::RemUIOp>(
                    b.create<arith::AddIOp>(idx0, idx1), num);
                auto lval = b.create<tensor::ExtractOp>(adaptor.getLhs(),
                                                        ValueRange{idx0});
                auto rval = b.create<tensor::ExtractOp>(adaptor.getRhs(),
                                                        ValueRange{idx1});
                auto mulOp = b.create<arith::MulIOp>(lval, rval);
                auto result = b.create<arith::AddIOp>(
                    mulOp, b.create<tensor::ExtractOp>(accumTensor, destIndex));
                auto stored =
                    b.create<tensor::InsertOp>(result, accumTensor, destIndex);
                b.create<scf::YieldOp>(stored.getResult());
              });

          b.create<scf::YieldOp>(innerLoop.getResults());
        });
    rewriter.replaceOp(op, outerLoop);
    return success();
  }
};

struct ConvertEval : public OpConversionPattern<EvalOp> {
  ConvertEval(mlir::MLIRContext *context)
      : OpConversionPattern<EvalOp>(context) {}
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(EvalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto polyTensorType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto numTerms = polyTensorType.getShape()[0];
    auto lower =
        b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));
    auto num =
        b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(numTerms));
    auto upper = b.create<arith::ConstantOp>(b.getIndexType(),
                                             b.getIndexAttr(numTerms + 1));
    auto step = lower;
    // Horner's method:
    // accum = 0
    // for i = 1, 2, ..., N
    //   accum = point * accum + coeff[N - i]
    auto accum =
        b.create<arith::ConstantOp>(b.getI32Type(), b.getI32IntegerAttr(0));
    auto loop = b.create<scf::ForOp>(
        lower, upper, step, accum.getResult(),
        [&](OpBuilder &builder, Location loc, Value loopIndex,
            ValueRange loopState) {
          ImplicitLocOpBuilder b(op.getLoc(), builder);
          auto accum = loopState.front();
          auto coeffIndex = b.create<arith::SubIOp>(num, loopIndex);
          auto mulOp = b.create<arith::MulIOp>(adaptor.getPoint(), accum);
          auto result = b.create<arith::AddIOp>(
              mulOp, b.create<tensor::ExtractOp>(adaptor.getInput(),
                                                 coeffIndex.getResult()));
          b.create<scf::YieldOp>(result.getResult());
        });
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<PolyDialect>();

    RewritePatternSet patterns(context);
    PolyToStandardTypeConverter typeConverter(context);
    patterns.add<ConvertAdd, ConvertSub, ConvertFromTensor, ConvertToTensor,
                 ConvertConstant, ConvertMul, ConvertEval>(typeConverter,
                                                           context);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });

    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
             isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                              typeConverter) ||
             isLegalForReturnOpTypeConversionPattern(op, typeConverter);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace poly
} // namespace foo
} // namespace mlir
