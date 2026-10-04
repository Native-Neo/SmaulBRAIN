// SmaulBRAIN native kernel: RMSNorm forward (FP32 statistics).
//
// Numerics match rmsnorm.py exactly:
//   y[i] = (x[i] / sqrt(mean(x^2) + eps)) * w[i]   (accumulators in float)
// Compiles standalone (g++ -O2) or via torch.utils.cpp_extension.
// The Python path is already vectorized; this kernel exists for the
// measured hot loop on weak CPUs without a BLAS victory margin.

#include <cmath>
#include <cstddef>

extern "C" void smaul_rmsnorm_forward(
    const float* x, const float* w, float* y,
    std::size_t rows, std::size_t cols, float eps) {
  for (std::size_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    float acc = 0.0f;
    for (std::size_t c = 0; c < cols; ++c) acc += xr[c] * xr[c];
    float inv = 1.0f / std::sqrt(acc / (float)cols + eps);
    float* yr = y + r * cols;
    for (std::size_t c = 0; c < cols; ++c) yr[c] = xr[c] * inv * w[c];
  }
}
