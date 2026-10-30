#define launch_native_nvfp4_m1 unchanged_launch_native_nvfp4_m1
#include "baseline_dispatch.cu"
#undef launch_native_nvfp4_m1

extern "C" int launch_input_reuse8(
    const void*, const uint8_t*, const uint8_t*, const float*, void*,
    int, int, int, void*);
extern "C" int launch_output_record1024(
    const void*, const uint8_t*, const uint8_t*, const float*, void*,
    int, int, int, void*);

extern "C" int launch_native_nvfp4_m1(
    const void* x, const uint8_t* packed, const uint8_t* scales, const float* global,
    void* output, int rows, int n, int k, void* stream) {
  if (rows == 8 && n == 18560 && k == 4096)
    return launch_input_reuse8(x, packed, scales, global, output, rows, n, k, stream);
  if (rows == 8 && n == 4096 && k == 8192)
    return launch_output_record1024(x, packed, scales, global, output, rows, n, k, stream);
  return unchanged_launch_native_nvfp4_m1(
      x, packed, scales, global, output, rows, n, k, stream);
}
