#include "quant_blocks.cuh"

extern "C" void decode_blocks(const uint8_t* input, float* output, int rows, int k, int format) {
  const int row_bytes = k / (format == 12 ? 256 : 32) * (format == 12 ? 144 : format == 6 ? 22 : 34);
  for (int row = 0; row < rows; ++row)
    for (int column = 0; column < k; ++column)
      output[row * k + column] = DecodeWeight(input + row * row_bytes, column, format);
}
