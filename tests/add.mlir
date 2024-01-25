// RUN: foo-opt %s -canonicalize | FileCheck %s

func.func @add(%arg: i32) -> i32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1 : i32
  %2 = arith.addi %0, %1 : i32
  func.return %2 : i32
}

// CHECK-LABEL: func.func @add
// CHECK-NEXT: %[[C:.*]] = arith.constant 2
// CHECK-NEXT: return %[[C]]
