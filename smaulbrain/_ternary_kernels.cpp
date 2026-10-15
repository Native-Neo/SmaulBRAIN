// Portable baseline C++17 kernels for 2-bit packed ternary matrices.
// Layout: row-major matrix shape (din, dout), four 2-bit codes per byte.
// Codes 0,1,2,3 map to -1,0,+1,+1 respectively (3 is reserved by the packer).
// SSE2 is baseline on x86-64; the same source retains scalar fallback support.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#if defined(__SSE2__)
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

#if defined(__SSE2__)
        if ((dout & 3u) == 0u) {
            // Accumulate a 32-column tile in registers and store it only once.
            // This removes a load/store of the output accumulator for every
            // input feature, which is costly for the narrow V2 CPU presets.
            constexpr std::size_t kTile = 32;
            for (std::size_t j0 = 0; j0 < dout; j0 += kTile) {
                const std::size_t width = std::min(kTile, dout - j0);
                const std::size_t nvec = width / 4;
                __m128 acc[8];
                for (std::size_t v = 0; v < nvec; ++v) acc[v] = _mm_setzero_ps();

                for (std::size_t i = 0; i < din; ++i) {
                    const __m128 xv = _mm_set1_ps(in[i]);
                    const std::size_t packed_row = (i * dout + j0) >> 2;
                    for (std::size_t v = 0; v < nvec; ++v) {
                        const unsigned byte = packed[packed_row + v];
                        const __m128 weights = trit_vector(byte);
                        acc[v] = _mm_add_ps(acc[v], _mm_mul_ps(xv, weights));
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
