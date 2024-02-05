// RUN: foo-opt --canonicalize %s | FileCheck %s

func.func @test_simple() -> !poly.poly<10> {
  %0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
  %2 = poly.add %p0, %p0 : !poly.poly<10>
  %3 = poly.mul %p0, %p0 : !poly.poly<10>
  %4 = poly.add %2, %3 : !poly.poly<10>
  return %2 : !poly.poly<10>
}

// CHECK-LABEL: @test_simple
// CHECK:       poly.constant dense<[2, 4, 6]>
// CHECK-NEXT:  return

func.func @test_difference_of_squares(
    %0: !poly.poly<3>, %1: !poly.poly<3>) -> !poly.poly<3> {
  %2 = poly.mul %0, %0 : !poly.poly<3>
  %3 = poly.mul %1, %1 : !poly.poly<3>
  %4 = poly.sub %2, %3 : !poly.poly<3>
  %5 = poly.add %4, %4 : !poly.poly<3>
  return %5 : !poly.poly<3>
}

// CHECK-LABEL: func.func @test_difference_of_squares
// CHECK-SAME: %[[x:.+]]: !poly.poly<3>,
// CHECK-SAME: %[[y:.+]]: !poly.poly<3>
// CHECK: %[[sum:.+]] = poly.add %[[x]], %[[y]]
// CHECK: %[[diff:.+]] = poly.sub %[[x]], %[[y]]
// CHECK: %[[mul:.+]] = poly.mul %[[sum]], %[[diff]]

func.func @test_normalize_conj_through_eval(
    %f: !poly.poly<3>, %z: complex<f64>) -> complex<f64> {
  %z_bar = complex.conj %z : complex<f64>
  %evaled = poly.eval %f, %z_bar : (!poly.poly<3>, complex<f64>) -> complex<f64>
  return %evaled : complex<f64>
}

// CHECK-LABEL: func.func @test_normalize_conj_through_eval
// CHECK-SAME: %[[f:.+]]: !poly.poly<3>,
// CHECK-SAME: %[[z:.+]]: complex<f64>
// CHECK: %[[evaled:.+]] = poly.eval %[[f]], %[[z]]
// CHECK-NEXT: %[[eval_bar:.+]] = complex.conj %[[evaled]]
// CHECK-NEXT: return %[[eval_bar]]
