#pragma once
// Only graph-observed attention and vocabulary projection dimensions.
inline bool source_q8_shape(int n, int k) {
  return k == 4096 && (n == 256 || n == 4096 || n == 131072);
}
