#include <cstdint>

extern "C" int input_launch_native_nvfp4_m1(
    const void*, const uint8_t*, const uint8_t*, const float*, void*,
    int, int, int, void*);
extern "C" int accepted_native_nvfp4_m1(
    const void*, const uint8_t*, const uint8_t*, const float*, void*,
    int, int, int, void*);
extern "C" int launch_out_lookahead4(
    const void*, const uint8_t*, const uint8_t*, const float*, void*,
    int, int, int, void*);
extern "C" int launch_r8_tile256(
    const void*, const uint8_t*, const uint8_t*, const float*, void*,
    int, int, int, void*);

extern "C" int launch_native_nvfp4_m1(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* stream) {
  // The byte-identical public wrapper admits these only for logical M=1.
  if (n == 18560 && k == 4096 && (rows == 1 || rows == 8))
    return input_launch_native_nvfp4_m1(
        x, packed, scales, global, output, rows, n, k, stream);
  if (n == 4096 && k == 8192) {
    if (rows == 1)
      return launch_out_lookahead4(x, packed, scales, global, output, rows, n, k, stream);
    if (rows == 8)
      return launch_r8_tile256(x, packed, scales, global, output, rows, n, k, stream);
  }
  return accepted_native_nvfp4_m1(x, packed, scales, global, output, rows, n, k, stream);
}
