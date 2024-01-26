// RUN: foo-opt %s

module {
  func.func @main(%arg0: !poly.poly<10>) -> !poly.poly<10> {
    return %arg0 : !poly.poly<10>
  }

  func.func @test_binop_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> i32 {
    %0 = poly.add %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    %1 = poly.sub %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    %2 = poly.mul %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    %3 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
    %4 = poly.from_tensor %3 : tensor<3xi32> -> !poly.poly<10>
    %5 = arith.constant 7 : i32
    %6 = poly.eval %4, %5 : (!poly.poly<10>, i32) -> i32
    return %6 : i32
  }
}
