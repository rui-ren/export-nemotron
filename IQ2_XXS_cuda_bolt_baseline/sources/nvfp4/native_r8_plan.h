#pragma once
namespace native_r8_plan {
inline bool input(int rows, int n, int k) {
  return rows == 8 && n == 18560 && k == 4096;
}
inline bool output(int rows, int n, int k) {
  return rows == 8 && n == 4096 && k == 8192;
}
}  // namespace native_r8_plan
