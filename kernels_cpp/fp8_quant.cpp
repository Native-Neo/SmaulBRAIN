// SmaulBRAIN native kernel: blockwise FP8 (E4M3) quantize/dequantize.
//
// Matches smaulbrain/precision.py block layout: each row is split into
// `tile`-wide blocks; block b of row r has FP32 scale s[r,b] = amax/448.
// Quantize: code = e4m3_round(x / s) stored as uint8 bit pattern.
// Dequantize: x ~= e4m3(code) * s. Row-block granularity keeps transients
// small: callers process one expert (or one row block) at a time.
//
// E4M3 encoding used here: sign(1) | exp(4, bias 8) | mantissa(3), with
// NaN/Inf saturating to the max code 0x7E/0xFE (finite-only storage).

#include <cmath>
#include <cstddef>
#include <cstdint>

static inline uint8_t f32_to_e4m3(float v) {
  if (!std::isfinite(v)) return v > 0 ? 0x7E : 0xFE;
  uint8_t sign = v < 0 ? 0x80 : 0x00;
  float a = std::fabs(v);
  if (a < 1e-9f) return sign;  // signed zero
  int exp2 = (int)std::floor(std::log2(a));
  // E4M3 normal range: 2^-8 .. 448; subnormals flush toward zero bins.
  int e = exp2 + 8;  // biased exponent
  if (e < 1) {
    float frac = a / (float)(1 << (exp2 + 8)) ;  // subnormal-ish path
    (void)frac;
    // Flush tiny values to the smallest representable mantissa steps.
    float step = std::ldexp(1.0f, -11);  // 2^-11 ~ min subnormal scale
    int m = (int)std::floor(a / step + 0.5f);
    if (m < 1) return sign;
    if (m > 7) m = 7;
    return (uint8_t)(sign | (unsigned)m);
  }
  if (e > 14) return (uint8_t)(sign | 0x7E);  // saturate, stay finite
  float base = std::ldexp(1.0f, exp2);
  int m = (int)std::floor((a / base - 1.0f) * 8.0f + 0.5f);
  if (m > 7) m = 7;
  if (m < 0) m = 0;
  return (uint8_t)(sign | (unsigned)((e << 3) | m));
}

static inline float e4m3_to_f32(uint8_t c) {
  uint8_t sign = (c & 0x80) ? 1 : 0;
  int e = (c >> 3) & 0x0F;
  int m = c & 0x07;
  float v;
  if (e == 0) {
    v = std::ldexp((float)m, -11);  // subnormal
  } else if (e == 15) {
    v = 448.0f;  // saturated codes read back as max (finite storage)
  } else {
    v = std::ldexp(1.0f + (float)m / 8.0f, e - 8);
  }
  return sign ? -v : v;
}

extern "C" void smaul_fp8_quant_row_block(
    const float* w, uint8_t* codes, float* scales,
    std::size_t rows, std::size_t cols, std::size_t tile) {
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0, nb = (cols + tile - 1) / tile; b < nb; ++b) {
      float amax = 1e-12f;
      std::size_t c0 = b * tile, c1 = c0 + tile < cols ? c0 + tile : cols;
      for (std::size_t c = c0; c < c1; ++c) {
        float a = std::fabs(w[r * cols + c]);
        if (a > amax) amax = a;
      }
      float s = amax / 448.0f;
      scales[r * ((cols + tile - 1) / tile) + b] = s;
      for (std::size_t c = c0; c < c1; ++c)
        codes[r * cols + c] = f32_to_e4m3(w[r * cols + c] / s);
    }
  }
}

extern "C" void smaul_fp8_dequant_row_block(
    const uint8_t* codes, const float* scales, float* w,
    std::size_t rows, std::size_t cols, std::size_t tile) {
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0, nb = (cols + tile - 1) / tile; b < nb; ++b) {
      float s = scales[r * ((cols + tile - 1) / tile) + b];
      std::size_t c0 = b * tile, c1 = c0 + tile < cols ? c0 + tile : cols;
      for (std::size_t c = c0; c < c1; ++c)
        w[r * cols + c] = e4m3_to_f32(codes[r * cols + c]) * s;
    }
  }
}
