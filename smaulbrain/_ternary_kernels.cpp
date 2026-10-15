// Portable packed ternary kernels.
// Matrix W is row-major (din, dout), with four 2-bit codes per byte.
// Codes 0,1,2,3 map to -1,0,+1,+1 (3 is reserved by the packer).
// Compile with -mavx only when the host OS/CPU advertises AVX. AVX2 and
// AVX-512 are not used. The SSE2/scalar implementation remains available.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

static constexpr float kTritValue[4] = {-1.0f, 0.0f, 1.0f, 1.0f};

#if defined(__SSE2__)
using TritLut = std::array<std::array<float, 4>, 256>;
static const TritLut& trit_lut() noexcept {
    static const TritLut table = [] {
        TritLut t{};
        for (unsigned b = 0; b < 256; ++b) {
            for (unsigned lane = 0; lane < 4; ++lane) {
                const unsigned code = (b >> (lane * 2)) & 3u;
                t[b][lane] = kTritValue[code];
            }
        }
        return t;
    }();
    return table;
}
static inline __m128 trit_vector(unsigned byte) noexcept {
    return _mm_loadu_ps(trit_lut()[byte].data());
}
static inline float horizontal_sum(__m128 v) noexcept {
    const __m128 pair = _mm_add_ps(v, _mm_movehl_ps(v, v));
    const __m128 total = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1));
    return _mm_cvtss_f32(total);
}
#endif

#if defined(__AVX__)
static inline __m256 trit_vector8(unsigned b0, unsigned b1) noexcept {
    const __m128 lo = trit_vector(b0);
    const __m128 hi = trit_vector(b1);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1);
}
static inline float horizontal_sum(__m256 v) noexcept {
    const __m128 pair = _mm_add_ps(_mm256_castps256_ps128(v),
                                   _mm256_extractf128_ps(v, 1));
    return horizontal_sum(pair);
}
#endif

static inline unsigned ternary_code(const std::uint8_t* packed,
                                    std::size_t pos) noexcept {
    return (static_cast<unsigned>(packed[pos >> 2]) >>
            static_cast<unsigned>((pos & 3u) * 2u)) & 3u;
}

extern "C" int smaulbrain_ternary_matmat(
    const float* x, const std::uint8_t* packed, float scale, float* y,
    std::size_t batch, std::size_t din, std::size_t dout) noexcept {
    if (!x || !packed || !y || din == 0 || dout == 0) return 1;

#if defined(__AVX__)
    if (batch == 1 && (dout & 7u) == 0u) {
        // Block input rows outside output tiles so the current row group stays
        // cache-hot while scanning adjacent output columns. The output-tile
        // outer loop revisited widely spaced row cache lines for each tile.
        constexpr std::size_t kInputTile = 16;
        constexpr std::size_t kOutputTile = 64;
        std::fill(y, y + dout, 0.0f);
        for (std::size_t i0 = 0; i0 < din; i0 += kInputTile) {
            const std::size_t iend = std::min(i0 + kInputTile, din);
            for (std::size_t j0 = 0; j0 < dout; j0 += kOutputTile) {
                const std::size_t width = std::min(kOutputTile, dout - j0);
                const std::size_t nvec = width / 8;
                __m256 acc[8];
                for (std::size_t v = 0; v < nvec; ++v) {
                    acc[v] = _mm256_setzero_ps();
                }
                for (std::size_t i = i0; i < iend; ++i) {
                    const __m256 xv = _mm256_set1_ps(x[i]);
                    const std::size_t base = (i * dout + j0) >> 2;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
                    for (std::size_t v = 0; v < nvec; ++v) {
                        const __m256 wv = trit_vector8(packed[base + v * 2],
                                                       packed[base + v * 2 + 1]);
                        acc[v] = _mm256_add_ps(acc[v], _mm256_mul_ps(xv, wv));
                    }
                }
                for (std::size_t v = 0; v < nvec; ++v) {
                    const std::size_t out_idx = j0 + v * 8;
                    const __m256 prior = _mm256_loadu_ps(y + out_idx);
                    _mm256_storeu_ps(y + out_idx, _mm256_add_ps(prior, acc[v]));
                }
            }
        }
        const __m256 scalev = _mm256_set1_ps(scale);
        for (std::size_t j = 0; j < dout; j += 8) {
            const __m256 sum = _mm256_loadu_ps(y + j);
            _mm256_storeu_ps(y + j, _mm256_mul_ps(sum, scalev));
        }
        return 0;
    }
    if ((dout & 7u) == 0u) {
        // Process two tokens at once so packed bytes and decoded trit vectors
        // are reused for both rows. This is especially useful for prompt
        // prefill, where the same weights multiply multiple input positions.
        constexpr std::size_t kTile = 32;
        constexpr std::size_t kBatchTile = 2;
        for (std::size_t j0 = 0; j0 < dout; j0 += kTile) {
            const std::size_t width = std::min(kTile, dout - j0);
            const std::size_t nvec = width / 8;
            for (std::size_t b0 = 0; b0 < batch; b0 += kBatchTile) {
                const bool has_second = b0 + 1 < batch;
                const float* in0 = x + b0 * din;
                const float* in1 = has_second ? x + (b0 + 1) * din : nullptr;
                float* out0 = y + b0 * dout;
                float* out1 = has_second ? y + (b0 + 1) * dout : nullptr;
                __m256 acc0[4], acc1[4];
                for (std::size_t v = 0; v < nvec; ++v) {
                    acc0[v] = _mm256_setzero_ps();
                    if (has_second) acc1[v] = _mm256_setzero_ps();
                }
                for (std::size_t i = 0; i < din; ++i) {
                    const __m256 x0 = _mm256_set1_ps(in0[i]);
                    const __m256 x1 = has_second ? _mm256_set1_ps(in1[i]) : _mm256_setzero_ps();
                    const std::size_t base = (i * dout + j0) >> 2;
                    for (std::size_t v = 0; v < nvec; ++v) {
                        const __m256 wv = trit_vector8(packed[base + v * 2],
                                                       packed[base + v * 2 + 1]);
                        acc0[v] = _mm256_add_ps(acc0[v], _mm256_mul_ps(x0, wv));
                        if (has_second) {
                            acc1[v] = _mm256_add_ps(acc1[v], _mm256_mul_ps(x1, wv));
                        }
                    }
                }
                const __m256 scalev = _mm256_set1_ps(scale);
                for (std::size_t v = 0; v < nvec; ++v) {
                    _mm256_storeu_ps(out0 + j0 + v * 8, _mm256_mul_ps(acc0[v], scalev));
                    if (has_second) {
                        _mm256_storeu_ps(out1 + j0 + v * 8,
                                         _mm256_mul_ps(acc1[v], scalev));
                    }
                }
            }
        }
        return 0;
    }
#endif

    for (std::size_t b = 0; b < batch; ++b) {
        const float* in = x + b * din;
        float* out = y + b * dout;
#if defined(__SSE2__)
        if ((dout & 3u) == 0u) {
            constexpr std::size_t kTile = 32;
            for (std::size_t j0 = 0; j0 < dout; j0 += kTile) {
                const std::size_t width = std::min(kTile, dout - j0);
                const std::size_t nvec = width / 4;
                __m128 acc[8];
                for (std::size_t v = 0; v < nvec; ++v) acc[v] = _mm_setzero_ps();
                for (std::size_t i = 0; i < din; ++i) {
                    const __m128 xv = _mm_set1_ps(in[i]);
                    const std::size_t base = (i * dout + j0) >> 2;
                    for (std::size_t v = 0; v < nvec; ++v) {
                        const __m128 wv = trit_vector(packed[base + v]);
                        acc[v] = _mm_add_ps(acc[v], _mm_mul_ps(xv, wv));
                    }
                }
                const __m128 scalev = _mm_set1_ps(scale);
                for (std::size_t v = 0; v < nvec; ++v) {
                    _mm_storeu_ps(out + j0 + v * 4, _mm_mul_ps(acc[v], scalev));
                }
            }
            continue;
        }
#endif
        for (std::size_t j = 0; j < dout; ++j) {
            float acc = 0.0f;
            for (std::size_t i = 0; i < din; ++i) {
                const unsigned code = ternary_code(packed, i * dout + j);
                if (code == 0u) acc -= in[i];
                else if (code >= 2u) acc += in[i];
            }
            out[j] = acc * scale;
        }
    }
    return 0;
}

extern "C" int smaulbrain_ternary_transpose_matmat(
    const float* dy, const std::uint8_t* packed, float scale, float* dx,
    std::size_t batch, std::size_t din, std::size_t dout) noexcept {
    if (!dy || !packed || !dx || din == 0 || dout == 0) return 1;

    for (std::size_t b = 0; b < batch; ++b) {
        const float* grad_out = dy + b * dout;
        float* grad_in = dx + b * din;
        for (std::size_t i = 0; i < din; ++i) {
            const std::size_t row = i * dout;
            float acc = 0.0f;
            std::size_t j = 0;
#if defined(__AVX__)
            if ((dout & 7u) == 0u) {
                __m256 acc8 = _mm256_setzero_ps();
                for (; j < dout; j += 8) {
                    const std::size_t base = (row + j) >> 2;
                    const __m256 grads = _mm256_loadu_ps(grad_out + j);
                    const __m256 weights = trit_vector8(packed[base], packed[base + 1]);
                    acc8 = _mm256_add_ps(acc8, _mm256_mul_ps(grads, weights));
                }
                acc = horizontal_sum(acc8);
            } else
#endif
#if defined(__SSE2__)
            if ((dout & 3u) == 0u) {
                __m128 acc4 = _mm_setzero_ps();
                for (; j < dout; j += 4) {
                    const unsigned byte = packed[(row + j) >> 2];
                    const __m128 grads = _mm_loadu_ps(grad_out + j);
                    acc4 = _mm_add_ps(acc4, _mm_mul_ps(grads, trit_vector(byte)));
                }
                acc = horizontal_sum(acc4);
            } else
#endif
            {
                for (; j < dout; ++j) {
                    const unsigned code = ternary_code(packed, row + j);
                    if (code == 0u) acc -= grad_out[j];
                    else if (code >= 2u) acc += grad_out[j];
                }
            }
            grad_in[i] = acc * scale;
        }
    }
    return 0;
}


extern "C" int smaulbrain_pack_ternary(
    const float* weights, std::uint8_t* packed, std::size_t n,
    float* scale_out, float* threshold_out) noexcept {
    if (!weights || !packed || !scale_out || !threshold_out || n == 0) return 1;

    double sum_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_abs += std::fabs(static_cast<double>(weights[i]));
    }
    const double mean_abs = sum_abs / static_cast<double>(n);
    const double threshold = 0.7 * mean_abs;

    // Combine scale accumulation and code emission into one pass. The threshold
    // is already known, so we can classify each weight, accumulate the nonzero
    // absolute values, and write its packed trit without a third matrix scan.
    double selected_abs_sum = 0.0;
    std::size_t selected_count = 0;
    const std::size_t nbytes = (n + 3u) / 4u;
    for (std::size_t b = 0; b < nbytes; ++b) {
        unsigned byte = 0u;
        for (unsigned lane = 0; lane < 4; ++lane) {
            const std::size_t idx = b * 4u + lane;
            unsigned code = 1u;
            if (idx < n) {
                const double v = static_cast<double>(weights[idx]);
                if (v > threshold) {
                    code = 2u;
                    selected_abs_sum += v;
                    ++selected_count;
                } else if (v < -threshold) {
                    code = 0u;
                    selected_abs_sum -= v;
                    ++selected_count;
                }
            }
            byte |= code << (lane * 2u);
        }
        packed[b] = static_cast<std::uint8_t>(byte);
    }
    double scale = selected_count
        ? selected_abs_sum / static_cast<double>(selected_count)
        : mean_abs;
    if (!(scale > 0.0)) scale = 1.0;
    *scale_out = static_cast<float>(scale);
    *threshold_out = static_cast<float>(threshold);
    return 0;
}
