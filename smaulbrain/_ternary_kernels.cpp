// Portable packed ternary kernels.
// Matrix W is row-major (din, dout), with four 2-bit codes per byte.
// Codes 0,1,2,3 map to -1,0,+1,+1 (3 is reserved by the packer).
// Compile with -mavx only when the host OS/CPU advertises AVX. AVX2 and
// AVX-512 are not used. The SSE2/scalar implementation remains available.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

    for (std::size_t b = 0; b < batch; ++b) {
        const float* in = x + b * din;
        float* out = y + b * dout;

#if defined(__AVX__)
        if ((dout & 7u) == 0u) {
            // Four 256-bit accumulators hold a 32-column tile in registers.
            // Each packed byte pair yields eight ternary weights; the output
            // tile is written once, not reloaded/stored for every input row.
            constexpr std::size_t kTile = 32;
            for (std::size_t j0 = 0; j0 < dout; j0 += kTile) {
                const std::size_t width = std::min(kTile, dout - j0);
                const std::size_t nvec = width / 8;
                __m256 acc[4];
                for (std::size_t v = 0; v < nvec; ++v) acc[v] = _mm256_setzero_ps();
                for (std::size_t i = 0; i < din; ++i) {
                    const __m256 xv = _mm256_set1_ps(in[i]);
                    const std::size_t base = (i * dout + j0) >> 2;
                    for (std::size_t v = 0; v < nvec; ++v) {
                        const __m256 wv = trit_vector8(packed[base + v * 2],
                                                       packed[base + v * 2 + 1]);
                        acc[v] = _mm256_add_ps(acc[v], _mm256_mul_ps(xv, wv));
                    }
                }
                const __m256 scalev = _mm256_set1_ps(scale);
                for (std::size_t v = 0; v < nvec; ++v) {
                    _mm256_storeu_ps(out + j0 + v * 8, _mm256_mul_ps(acc[v], scalev));
                }
            }
            continue;
        }
#endif

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
