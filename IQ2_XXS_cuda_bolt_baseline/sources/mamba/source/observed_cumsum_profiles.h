#pragma once
#include "cumsum_h1_cubin.h"
#include "cumsum_h2_cubin.h"
#include "cumsum_h4_cubin.h"
#include "cumsum_h8_cubin.h"
#include "cumsum_h16_cubin.h"
#include "cumsum_h32_cubin.h"
#include "cumsum_h64_cubin.h"
struct ObservedCumsumProfile {
  int head_tile;
  const unsigned char* image;
  unsigned int grid, threads, shared;
};
constexpr ObservedCumsumProfile kObservedCumsumProfiles[] = {
  {1, observed_cumsum_h1, 128, 128, 16},
  {2, observed_cumsum_h2, 64, 128, 1024},
  {4, observed_cumsum_h4, 32, 128, 2048},
  {8, observed_cumsum_h8, 16, 128, 2048},
  {16, observed_cumsum_h16, 8, 128, 4096},
  {32, observed_cumsum_h32, 4, 128, 8192},
  {64, observed_cumsum_h64, 2, 128, 16384},
};
