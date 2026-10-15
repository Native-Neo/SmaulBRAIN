// Portable baseline C++17 kernels for 2-bit packed ternary matrices.
// Layout: row-major matrix shape (din, dout), four 2-bit codes per byte.
// Codes 0,1,2,3 map to -1,0,+1,+1 respectively (3 is reserved by the packer).
#include <cstddef>
#include <cstdint>

static inline unsigned ternary_code(const std::uint8_t* packed,
                                    std::size_t pos) noexcept {
    return (static_cast<unsigned>(packed[pos >> 2]) >>
            static_cast<unsigned>((pos & 3u) * 2u)) & 3u;
}

extern "C" int smaulbrain_ternary_matmat(
    const float* x, const std::uint8_t* packed, float scale, float* y,
    std::size_t batch, std::size_t din, std::size_t dout) noexcept {
    if (!x || !packed || !y || din == 0 || dout == 0) return 1;
    const std::size_t out_count = batch * dout;
    for (std::size_t k = 0; k < out_count; ++k) y[k] = 0.0f;

    for (std::size_t b = 0; b < batch; ++b) {
        float* out = y + b * dout;
        const float* in = x + b * din;
        for (std::size_t i = 0; i < din; ++i) {
            const float xi = in[i];
            const std::size_t row = i * dout;
            std::size_t j = 0;

            // All hardcoded V2 preset projection widths are multiples of four.
            // Process a whole byte per iteration in this common aligned case.
            if ((dout & 3u) == 0u) {
                for (; j < dout; j += 4) {
                    const unsigned v = packed[(row + j) >> 2];
                    const unsigned c0 = v & 3u;
                    const unsigned c1 = (v >> 2) & 3u;
                    const unsigned c2 = (v >> 4) & 3u;
                    const unsigned c3 = (v >> 6) & 3u;
                    if (c0 == 0u) out[j] -= xi;
                    else if (c0 >= 2u) out[j] += xi;
                    if (c1 == 0u) out[j + 1] -= xi;
                    else if (c1 >= 2u) out[j + 1] += xi;
                    if (c2 == 0u) out[j + 2] -= xi;
                    else if (c2 >= 2u) out[j + 2] += xi;
                    if (c3 == 0u) out[j + 3] -= xi;
                    else if (c3 >= 2u) out[j + 3] += xi;
                }
            } else {
                for (; j < dout; ++j) {
                    const unsigned c = ternary_code(packed, row + j);
                    if (c == 0u) out[j] -= xi;
                    else if (c >= 2u) out[j] += xi;
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
            if ((dout & 3u) == 0u) {
                for (; j < dout; j += 4) {
                    const unsigned v = packed[(row + j) >> 2];
                    const unsigned c0 = v & 3u;
                    const unsigned c1 = (v >> 2) & 3u;
                    const unsigned c2 = (v >> 4) & 3u;
                    const unsigned c3 = (v >> 6) & 3u;
                    if (c0 == 0u) acc -= grad_out[j];
                    else if (c0 >= 2u) acc += grad_out[j];
                    if (c1 == 0u) acc -= grad_out[j + 1];
                    else if (c1 >= 2u) acc += grad_out[j + 1];
                    if (c2 == 0u) acc -= grad_out[j + 2];
                    else if (c2 >= 2u) acc += grad_out[j + 2];
                    if (c3 == 0u) acc -= grad_out[j + 3];
                    else if (c3 >= 2u) acc += grad_out[j + 3];
                }
            } else {
                for (; j < dout; ++j) {
                    const unsigned c = ternary_code(packed, row + j);
                    if (c == 0u) acc -= grad_out[j];
                    else if (c >= 2u) acc += grad_out[j];
                }
            }
            grad_in[i] = acc * scale;
        }
    }
    return 0;
}
