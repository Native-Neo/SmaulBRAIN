// SmaulBRAIN native kernel: blockwise FP8 (E4M3) quantize/dequantize.
//
// Matches precision.py block layout: each row is split into
// `tile`-wide blocks; block b of row r has FP32 scale s[r,b] = amax/448.
// Quantize: code = e4m3_round(x / s) stored as uint8 bit pattern.
// Dequantize: x ~= e4m3(code) * s. Row-block granularity keeps transients
// small: callers process one expert (or one row block) at a time.
//
// E4M3 encoding used here (OCP: sign(1) | exp(4, bias 7) | mantissa(3)):
// normals (1+m/8)*2^(e-7) for e=1..14, extended normals e=15,m<=6
// (0x78..0x7E = 256..448), subnormals m*2^-9 for e=0. Out-of-range
// magnitudes saturate to 0x7E/0xFE (finite-only storage); 0x7F/0xFF (NaN)
// are never emitted.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

static inline uint8_t f32_to_e4m3(float v) {
  if (!std::isfinite(v)) return v > 0 ? 0x7E : 0xFE;
  uint8_t sign = v < 0 ? 0x80 : 0x00;
  float a = std::fabs(v);
  if (a < 1e-9f) return sign;  // signed zero
  int exp2 = (int)std::floor(std::log2(a));
  // E4M3 normal range: 2^-6 .. 448; tinier values use the subnormal bins.
  int e = exp2 + 7;  // biased exponent
  if (e < 1) {
    float step = std::ldexp(1.0f, -9);  // subnormal quantum m*2^-9
    int m = (int)std::floor(a / step + 0.5f);
    if (m < 1) return sign;
    if (m > 7) m = 7;
    return (uint8_t)(sign | (unsigned)m);
  }
  if (e > 15) return (uint8_t)(sign | 0x7E);  // saturate, stay finite
  float base = std::ldexp(1.0f, exp2);
  int m = (int)std::floor((a / base - 1.0f) * 8.0f + 0.5f);
  if (m < 0) m = 0;
  if (m > 7) {
    // Rounded past the top of this binade: carry into e+1,m=0 (the true
    // nearest), instead of clamping to m=7 (a whole step too low).
    m = 0;
    e += 1;
  }
  if (e > 15) return (uint8_t)(sign | 0x7E);  // saturate, stay finite
  if (e == 15 && m > 6) m = 6;  // 0x7F would be NaN; clamp to max finite
  return (uint8_t)(sign | (unsigned)((e << 3) | m));
}

static inline float e4m3_to_f32(uint8_t c) {
  uint8_t sign = (c & 0x80) ? 1 : 0;
  int e = (c >> 3) & 0x0F;
  int m = c & 0x07;
  float v;
  if (e == 0) {
    v = std::ldexp((float)m, -9);  // subnormal
  } else if (e == 15 && m == 7) {
    v = std::numeric_limits<float>::quiet_NaN();  // 0x7F/0xFF, never emitted
    if (sign) v = -v;
  } else {
    // Includes the e=15 extended normals (0x78..0x7D = 256..416);
    // 0x7E decodes to 448.0f, the saturation value, by the same rule.
    v = std::ldexp(1.0f + (float)m / 8.0f, e - 7);
  }
  return sign ? -v : v;
}

extern "C" void smaul_fp8_quant_row_block(
    const float* w, uint8_t* codes, float* scales,
    std::size_t rows, std::size_t cols, std::size_t tile) {
  // Safety: tile==0 would divide by zero; null/empty are no-ops.
  // Baseline x86-64 only: no AVX/intrinsics, safe on Ivy Bridge and later.
  // NaN/Inf inputs saturate to finite codes via f32_to_e4m3 (never emit NaN).
  if (!w || !codes || !scales) return;
  if (rows == 0 || cols == 0 || tile == 0) return;
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
  // Same guards as quantize: never divide by zero or fault on null/empty.
  if (!codes || !scales || !w) return;
  if (rows == 0 || cols == 0 || tile == 0) return;
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t b = 0, nb = (cols + tile - 1) / tile; b < nb; ++b) {
      float s = scales[r * ((cols + tile - 1) / tile) + b];
      std::size_t c0 = b * tile, c1 = c0 + tile < cols ? c0 + tile : cols;
      for (std::size_t c = c0; c < c1; ++c)
        w[r * cols + c] = e4m3_to_f32(codes[r * cols + c]) * s;
    }
  }
}
