// SmaulBRAIN native kernel: linear-attention recurrent step.
//
// One token update with ELU+1 feature map and per-step key normalization,
// matching linear_attention.py::linear_attn_step:
//   kf = (elu(k)+1) / ||elu(k)+1|| ; qf = elu(q)+1
//   S += kf^T v ; z += kf ; y = (qf^T S) / (qf^T z + eps)
// Memory per call is O(Dh^2) — no sequence-length dependence.
// Caller provides scratch[2*Dh] so this unit stays dependency-free.

#include <cmath>
#include <cstddef>

static inline float elu_f(float v) { return v >= 0.0f ? v : std::exp(v) - 1.0f; }

extern "C" void smaul_linear_attn_step_buf(
    float* S, float* z,           // [Dh,Dh] row-major accumulator, [Dh] normalizer (in/out)
    const float* q, const float* k, const float* v,
    float* y, float* scratch, std::size_t Dh, float eps) {
  // Safety: never fault on bad pointers/sizes (Python validates first).
  // Baseline x86-64 only: no AVX/intrinsics, safe on Ivy Bridge and later.
  if (!S || !z || !q || !k || !v || !y || !scratch) return;
  if (Dh == 0) return;
  if (!(eps >= 0.0f) || !std::isfinite(eps)) return;
  float* qf = scratch;
  float* kf = scratch + Dh;
  float nrm = 0.0f;
  for (std::size_t i = 0; i < Dh; ++i) {
    qf[i] = elu_f(q[i]) + 1.0f;
    kf[i] = elu_f(k[i]) + 1.0f;
    nrm += kf[i] * kf[i];
  }
  float root = std::sqrt(nrm);
  nrm = 1.0f / (root > 1e-6f ? root : 1e-6f);
  for (std::size_t i = 0; i < Dh; ++i) kf[i] *= nrm;
  for (std::size_t i = 0; i < Dh; ++i) {
    z[i] += kf[i];
    float* Si = S + i * Dh;
    for (std::size_t j = 0; j < Dh; ++j) Si[j] += kf[i] * v[j];
  }
  float den = eps;
  for (std::size_t i = 0; i < Dh; ++i) den += qf[i] * z[i];
  if (den < eps) den = eps;
  for (std::size_t j = 0; j < Dh; ++j) {
    float num = 0.0f;
    for (std::size_t i = 0; i < Dh; ++i) num += qf[i] * S[i * Dh + j];
    y[j] = num / den;
  }
}
