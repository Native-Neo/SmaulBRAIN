// Portable baseline C++17 kernels for 2-bit packed ternary matrices.
// Layout: row-major matrix shape (din, dout), four 2-bit codes per byte.
// Codes 0,1,2,3 map to -1,0,+1,+1 respectively (3 is reserved by the packer).
// Uses SSE2 only when the compiler target already supports it (standard on
// x86-64); otherwise the same source compiles to the scalar reference kernel.
#include <cstddef>
#include <cstdint>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

static inline unsigned ternary_code(const std::uint8_t* packed,
                                    std::size_t pos) noexcept {
    return (static_cast<unsigned>(packed[pos >> 2]) >>
            static_cast<unsigned>((pos & 3u) * 2u)) & 3u;
}

static constexpr float kTritValue[4] = {-1.0f, 0.0f, 1.0f, 1.0f};

#if defined(__SSE2__)
static inline __m128 trit_vector(unsigned byte) noexcept {
    return _mm_set_ps(kTritValue[(byte >> 6) & 3u],
                      kTritValue[(byte >> 4) & 3u],
                      kTritValue[(byte >> 2) & 3u],
                      kTritValue[byte & 3u]);
}

static inline float horizontal_sum(__m128 v) noexcept {
    const __m128 pair = _mm_add_ps(v, _mm_movehl_ps(v, v));
    const __m128 total = _mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1));
    return _mm_cvtss_f32(total);
}
#endif

extern "C" int smaulbrain_ternary_matmat(
    const float* x, const std::uint8_t* packed, float scale, float* y,
    std::size_t batch, std::size_t din, std::size_t dout) noexcept {
    if (!x || !packed || !y || din == 0 || dout == 0) return 1;
    for (std::size_t k = 0; k < batch * dout; ++k) y[k] = 0.0f;

    for (std::size_t b = 0; b < batch; ++b) {
        float* out = y + b * dout;
        const float* in = x + b * din;
        for (std::size_t i = 0; i < din; ++i) {
            const float xi = in[i];
            const std::size_t row = i * dout;
            std::size_t j = 0;
#if defined(__SSE2__)
            if ((dout & 3u) == 0u) {
                const __m128 xv = _mm_set1_ps(xi);
                for (; j < dout; j += 4) {
                    const unsigned byte = packed[(row + j) >> 2];
                    __m128 acc = _mm_loadu_ps(out + j);
                    acc = _mm_add_ps(acc, _mm_mul_ps(xv, trit_vector(byte)));
                    _mm_storeu_ps(out + j, acc);
                }
            } else
#endif
            {
                for (; j < dout; ++j) {
                    const unsigned code = ternary_code(packed, row + j);
                    if (code == 0u) out[j] -= xi;
                    else if (code >= 2u) out[j] += xi;
                }
            }
        }
        for (std::size_t j = 0; j < dout; ++j) out[j] *= scale;
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
#if defined(__SSE2__)
            if ((dout & 3u) == 0u) {
                __m128 acc4 = _mm_setzero_ps();
                for (; j < dout; j += 4) {
                    const unsigned byte = packed[(row + j) >> 2];
                    const __m128 grad4 = _mm_loadu_ps(grad_out + j);
                    acc4 = _mm_add_ps(acc4, _mm_mul_ps(grad4, trit_vector(byte)));
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
