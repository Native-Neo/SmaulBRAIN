#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cstring>
#include <thread>
#include <vector>

// Packed ternary code per weight: 00=zero, 01=+1, 10=-1, 11=reserved.
// Row scales are applied once per output rather than once per weight.
// Inputs/outputs may be FP32 or IEEE bfloat16 bit storage.
namespace {

template <bool BF16>
inline float load_value(const void* data, std::size_t index) {
    if constexpr (!BF16) {
        return static_cast<const float*>(data)[index];
    } else {
        const std::uint16_t bits = static_cast<const std::uint16_t*>(data)[index];
        const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &wide, sizeof(value));
        return value;
    }
}

template <bool BF16>
inline void store_value(void* data, std::size_t index, float value) {
    if constexpr (!BF16) {
        static_cast<float*>(data)[index] = value;
    } else {
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        // IEEE round-to-nearest-even when reducing FP32 to bfloat16.
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        static_cast<std::uint16_t*>(data)[index] =
            static_cast<std::uint16_t>(bits >> 16);
    }
}

inline std::uint8_t code_at(const std::uint8_t* packed, std::size_t index) {
    return static_cast<std::uint8_t>(
        (packed[index >> 2] >> ((index & 3u) * 2u)) & 0x3u
    );
}

inline float e4m3_to_f32(std::uint8_t c) {
    const int sign = (c & 0x80u) ? -1 : 1;
    const int e = (c >> 3) & 0x0F;
    const int m = c & 0x07;
    if (e == 0) return static_cast<float>(sign) * std::ldexp(static_cast<float>(m), -9);
    if (e == 15 && m == 7) return std::numeric_limits<float>::quiet_NaN();
    return static_cast<float>(sign) * std::ldexp(1.0f + static_cast<float>(m) / 8.0f, e - 7);
}

inline float round_bf16(float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    bits &= 0xFFFF0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template <class Function>
void parallel_for(std::size_t count, int requested_threads, Function function,
                  std::size_t minimum_chunk = 4096) {
    // Avoid thread launch overhead for small matrices. Thread creation here is
    // per native call, so never spawn more workers than there are useful chunks.
    int threads = std::max(1, requested_threads);
    threads = std::min<int>(threads, static_cast<int>(
        std::max<std::size_t>(1, (count + minimum_chunk - 1) / minimum_chunk)));
    if (threads <= 1) {
        function(0, count);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads - 1));
    for (int t = 1; t < threads; ++t) {
        const std::size_t begin = count * static_cast<std::size_t>(t) /
                                  static_cast<std::size_t>(threads);
        const std::size_t end = count * static_cast<std::size_t>(t + 1) /
                                static_cast<std::size_t>(threads);
        workers.emplace_back([&, begin, end]() { function(begin, end); });
    }
    function(0, count / static_cast<std::size_t>(threads));
    for (auto& worker : workers) {
        worker.join();
    }
}

template <bool BF16>
void forward_impl(const void* x, const std::uint8_t* packed,
                  const float* scales, void* y,
                  std::size_t rows, std::size_t in_features,
                  std::size_t out_features, int threads) {
    const std::size_t outputs = rows * out_features;
    parallel_for(outputs, threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t cell = begin; cell < end; ++cell) {
            const std::size_t row = cell / out_features;
            const std::size_t out = cell % out_features;
            const std::size_t weight_base = out * in_features;
            const std::size_t input_base = row * in_features;
            float sum = 0.0f;
            for (std::size_t d = 0; d < in_features; ++d) {
                const std::uint8_t code = code_at(packed, weight_base + d);
                const float value = load_value<BF16>(x, input_base + d);
                if (code == 1u) {
                    sum += value;
                } else if (code == 2u) {
                    sum -= value;
                }
            }
            store_value<BF16>(y, cell, sum * scales[out]);
        }
    });
}

}  // namespace

extern "C" void smaul_ternary_linear_forward(
    const void* x, const std::uint8_t* packed, const float* scales, void* y,
    std::size_t rows, std::size_t in_features, std::size_t out_features,
    int is_bfloat16, int threads
) {
    if (!x || !packed || !scales || !y || rows == 0 ||
        in_features == 0 || out_features == 0) {
        return;
    }
    if (is_bfloat16) {
        forward_impl<true>(x, packed, scales, y, rows, in_features,
                           out_features, threads);
    } else {
        forward_impl<false>(x, packed, scales, y, rows, in_features,
                            out_features, threads);
    }
}


// Input-gradient kernel for a ternary linear layer. The caller has already
// multiplied each output-gradient column by its corresponding ternary row scale,
// so the inner loop consists exclusively of signed additions/subtractions.
// packed stores the forward weight [out_features, in_features], row-major.
extern "C" void smaul_ternary_linear_input_grad(
    const float* grad_scaled, const std::uint8_t* packed, float* grad_input,
    std::size_t rows, std::size_t in_features, std::size_t out_features,
    int threads
) {
    if (!grad_scaled || !packed || !grad_input || rows == 0 ||
        in_features == 0 || out_features == 0) {
        return;
    }
    const std::size_t outputs = rows * in_features;
    parallel_for(outputs, threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t cell = begin; cell < end; ++cell) {
            const std::size_t row = cell / in_features;
            const std::size_t in = cell % in_features;
            float sum = 0.0f;
            for (std::size_t out = 0; out < out_features; ++out) {
                const std::uint8_t code = code_at(packed, out * in_features + in);
                const float value = grad_scaled[row * out_features + out];
                if (code == 1u) {
                    sum += value;
                } else if (code == 2u) {
                    sum -= value;
                }
            }
            grad_input[cell] = sum;
        }
    });
}


// Convert blockwise FP8 E4M3 weights directly to packed row-scaled ternary
// codes. The FP8 values are decoded one scalar at a time; no dense FP32/BF16
// weight matrix is created. The flattened 2-bit stream uses 00=zero,
// 01=positive, 10=negative. Row scales preserve the existing ternary-QAT rule.
extern "C" void smaul_fp8_to_packed_ternary(
    const std::uint8_t* fp8_codes, const float* fp8_block_scales,
    std::uint8_t* packed, float* ternary_row_scales,
    std::size_t rows, std::size_t cols, std::size_t tile,
    float threshold_ratio, int round_to_bf16
) {
    if (!fp8_codes || !fp8_block_scales || !packed || !ternary_row_scales ||
        rows == 0 || cols == 0 || tile == 0 ||
        !std::isfinite(threshold_ratio) || threshold_ratio < 0.0f ||
        threshold_ratio >= 1.0f) {
        return;
    }
    const std::size_t nblocks = (cols + tile - 1) / tile;
    const std::size_t total = rows * cols;
    const std::size_t packed_bytes = (total + 3) / 4;
    std::memset(packed, 0, packed_bytes);

    for (std::size_t r = 0; r < rows; ++r) {
        float abs_sum = 0.0f;
        for (std::size_t c = 0; c < cols; ++c) {
            const float block_scale = fp8_block_scales[r * nblocks + c / tile];
            float value = e4m3_to_f32(fp8_codes[r * cols + c]) * block_scale;
            if (round_to_bf16) value = round_bf16(value);
            abs_sum += std::fabs(value);
        }
        const float threshold = (abs_sum / static_cast<float>(cols)) * threshold_ratio;
        float selected_abs_sum = 0.0f;
        std::size_t selected_count = 0;
        for (std::size_t c = 0; c < cols; ++c) {
            const float block_scale = fp8_block_scales[r * nblocks + c / tile];
            float value = e4m3_to_f32(fp8_codes[r * cols + c]) * block_scale;
            if (round_to_bf16) value = round_bf16(value);
            if (value > threshold || value < -threshold) {
                selected_abs_sum += std::fabs(value);
                ++selected_count;
            }
        }
        const float row_scale = selected_count
            ? selected_abs_sum / static_cast<float>(selected_count) : 0.0f;
        ternary_row_scales[r] = row_scale;
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t index = r * cols + c;
            const float block_scale = fp8_block_scales[r * nblocks + c / tile];
            float value = e4m3_to_f32(fp8_codes[index]) * block_scale;
            if (round_to_bf16) value = round_bf16(value);
            const std::uint8_t trit = value > threshold ? 1u
                : (value < -threshold ? 2u : 0u);
            packed[index >> 2] |= static_cast<std::uint8_t>(
                trit << ((index & 3u) * 2u));
        }
    }
}
