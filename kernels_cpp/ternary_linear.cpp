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


// Convert dense floating-point master parameters to packed ternary codes
// directly. No same-shape quantized weight tensor is created.
template <bool BF16>
static void float_to_packed_ternary_impl(
    const void* weights, std::uint8_t* packed, float* row_scales,
    std::size_t rows, std::size_t cols, float threshold_ratio
) {
    const std::size_t total = rows * cols;
    std::memset(packed, 0, (total + 3) / 4);
    for (std::size_t r = 0; r < rows; ++r) {
        float abs_sum = 0.0f;
        for (std::size_t col = 0; col < cols; ++col)
            abs_sum += std::fabs(load_value<BF16>(weights, r * cols + col));
        const float threshold =
            (abs_sum / static_cast<float>(cols)) * threshold_ratio;
        float chosen_sum = 0.0f;
        std::size_t chosen_count = 0;
        for (std::size_t col = 0; col < cols; ++col) {
            const float value = load_value<BF16>(weights, r * cols + col);
            if (value > threshold || value < -threshold) {
                chosen_sum += std::fabs(value);
                ++chosen_count;
            }
        }
        float scale = chosen_count
            ? chosen_sum / static_cast<float>(chosen_count) : 0.0f;
        if (BF16) scale = round_bf16(scale);
        row_scales[r] = scale;
        for (std::size_t col = 0; col < cols; ++col) {
            const std::size_t index = r * cols + col;
            const float value = load_value<BF16>(weights, index);
            const std::uint8_t code = value > threshold ? 1u
                : (value < -threshold ? 2u : 0u);
            packed[index >> 2] |= static_cast<std::uint8_t>(
                code << ((index & 3u) * 2u));
        }
    }
}

extern "C" void smaul_float_to_packed_ternary(
    const void* weights, std::uint8_t* packed, float* row_scales,
    std::size_t rows, std::size_t cols, float threshold_ratio,
    int is_bfloat16
) {
    if (!weights || !packed || !row_scales || rows == 0 || cols == 0 ||
        !std::isfinite(threshold_ratio) || threshold_ratio < 0.0f ||
        threshold_ratio >= 1.0f) return;
    if (is_bfloat16) {
        float_to_packed_ternary_impl<true>(
            weights, packed, row_scales, rows, cols, threshold_ratio);
    } else {
        float_to_packed_ternary_impl<false>(
            weights, packed, row_scales, rows, cols, threshold_ratio);
    }
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
        float row_scale = selected_count
            ? selected_abs_sum / static_cast<float>(selected_count) : 0.0f;
        // Python QAT casts every scaled trit back to the compute dtype before
        // extracting its row scale. Mirror BF16 rounding at this boundary.
        if (round_to_bf16) row_scale = round_bf16(row_scale);
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

static inline std::uint8_t f32_to_e4m3(float value) {
    if (!std::isfinite(value)) return value > 0 ? 0x7Eu : 0xFEu;
    const std::uint8_t sign = value < 0.0f ? 0x80u : 0x00u;
    const float a = std::fabs(value);
    if (a < 1e-9f) return sign;
    int exp2 = static_cast<int>(std::floor(std::log2(a)));
    int e = exp2 + 7;
    if (e < 1) {
        const float step = std::ldexp(1.0f, -9);
        int m = static_cast<int>(std::floor(a / step + 0.5f));
        if (m < 1) return sign;
        if (m > 7) m = 7;
        return static_cast<std::uint8_t>(sign | static_cast<unsigned>(m));
    }
    if (e > 15) return static_cast<std::uint8_t>(sign | 0x7Eu);
    const float base = std::ldexp(1.0f, exp2);
    int m = static_cast<int>(std::floor((a / base - 1.0f) * 8.0f + 0.5f));
    if (m < 0) m = 0;
    if (m > 7) {
        m = 0;
        ++e;
    }
    if (e > 15) return static_cast<std::uint8_t>(sign | 0x7Eu);
    if (e == 15 && m > 6) m = 6;
    return static_cast<std::uint8_t>(sign | static_cast<unsigned>((e << 3) | m));
}

// Low-memory expert optimizer update. FP8 weights are decoded one scalar at a
// time and requantized one row/tile at a time. It never writes a dense FP32
// copy of the expert weights. Optimizer moments stay in their existing BF16 or
// FP32 tensors; gradients are FP32. v is the canonical factored layout.
template <bool STATE_BF16>
int fp8_factored_update_impl(
    std::uint8_t* codes, float* scales, const float* grad,
    void* m_data, void* vrow_data, void* vcol_data,
    std::size_t rows, std::size_t cols, std::size_t tile,
    int step, float beta_m, float beta_v, float lr, float wd,
    float eps, float update_clip, float grad_multiplier,
    float* mean_abs_update
) {
    if (!codes || !scales || !grad || !m_data || !vrow_data || !vcol_data ||
        !mean_abs_update || rows == 0 || cols == 0 || tile == 0 || step < 1 ||
        !std::isfinite(beta_m) || beta_m < 0.0f || beta_m >= 1.0f ||
        !std::isfinite(beta_v) || beta_v < 0.0f || beta_v >= 1.0f ||
        !std::isfinite(lr) || lr < 0.0f ||
        !std::isfinite(wd) || wd < 0.0f ||
        !std::isfinite(eps) || eps <= 0.0f ||
        !std::isfinite(update_clip) || update_clip <= 0.0f ||
        !std::isfinite(grad_multiplier) || grad_multiplier < 0.0f) {
        return 0;
    }

    const std::size_t nblocks = (cols + tile - 1) / tile;
    const float bc1 = 1.0f - std::pow(beta_m, static_cast<float>(step));
    const float bc2 = 1.0f - std::pow(beta_v, static_cast<float>(step));
    if (!(bc1 > 0.0f) || !(bc2 > 0.0f)) return 0;

    // These are row/column vectors, not weight-sized buffers.
    std::vector<float> vrow_new(rows, 0.0f);
    std::vector<float> vcol_abs_sum(cols, 0.0f);
    std::vector<float> vcol_new(cols, 0.0f);
    auto* m_ptr = m_data;
    auto* vr_ptr = vrow_data;
    auto* vc_ptr = vcol_data;

    for (std::size_t r = 0; r < rows; ++r) {
        float row_abs_sum = 0.0f;
        const std::size_t base = r * cols;
        for (std::size_t col = 0; col < cols; ++col) {
            const float g = grad[base + col] * grad_multiplier;
            const float a = std::fabs(g);
            row_abs_sum += a;
            vcol_abs_sum[col] += a;
        }
        const float old_vr = load_value<STATE_BF16>(vr_ptr, r);
        const float vr = old_vr * beta_v +
            (row_abs_sum / static_cast<float>(cols)) * (1.0f - beta_v);
        vrow_new[r] = vr;
        store_value<STATE_BF16>(vr_ptr, r, vr);
    }
    for (std::size_t col = 0; col < cols; ++col) {
        const float old_vc = load_value<STATE_BF16>(vc_ptr, col);
        const float vc = old_vc * beta_v +
            (vcol_abs_sum[col] / static_cast<float>(rows)) * (1.0f - beta_v);
        vcol_new[col] = vc;
        store_value<STATE_BF16>(vc_ptr, col, vc);
    }

    // SmaulOpt shares a reconstructed-v denominator per 256-row block.
    const std::size_t row_group = 256;
    std::vector<float> denom_group((rows + row_group - 1) / row_group, 1e-12f);
    for (std::size_t g = 0; g < denom_group.size(); ++g) {
        const std::size_t r0 = g * row_group;
        const std::size_t r1 = std::min(rows, r0 + row_group);
        float sum = 0.0f;
        for (std::size_t r = r0; r < r1; ++r) sum += vrow_new[r] / bc2;
        denom_group[g] = std::max(sum / static_cast<float>(r1 - r0), 1e-12f);
    }

    std::vector<float> old_tile(tile, 0.0f);
    std::vector<float> new_tile(tile, 0.0f);
    float update_abs_sum = 0.0f;
    const float decay = lr * wd;
    for (std::size_t r = 0; r < rows; ++r) {
        const float vr_hat = vrow_new[r] / bc2;
        const float denom = denom_group[r / row_group];
        const std::size_t row_base = r * cols;
        for (std::size_t b = 0; b < nblocks; ++b) {
            const std::size_t c0 = b * tile;
            const std::size_t c1 = std::min(cols, c0 + tile);
            const float old_block_scale = scales[r * nblocks + b];
            float amax = 1e-12f;
            for (std::size_t col = c0; col < c1; ++col) {
                const std::size_t i = row_base + col;
                const float g = grad[i] * grad_multiplier;
                const float old_m = load_value<STATE_BF16>(m_ptr, i);
                const float m_new = old_m * beta_m + g * (1.0f - beta_m);
                store_value<STATE_BF16>(m_ptr, i, m_new);
                const float m_hat = m_new / bc1;
                float v_mat = (vr_hat * (vcol_new[col] / bc2)) / denom;
                v_mat += eps;
                float u = m_hat / v_mat;
                if constexpr (STATE_BF16) {
                    u = std::max(-update_clip, std::min(update_clip, u));
                }
                const float old_w = e4m3_to_f32(codes[i]) * old_block_scale;
                const float new_w = old_w * (1.0f - decay) - lr * u;
                const std::size_t q = col - c0;
                old_tile[q] = old_w;
                new_tile[q] = new_w;
                amax = std::max(amax, std::fabs(new_w));
            }
            const float new_block_scale = amax / 448.0f;
            scales[r * nblocks + b] = new_block_scale;
            for (std::size_t col = c0; col < c1; ++col) {
                const std::size_t q = col - c0;
                codes[row_base + col] = f32_to_e4m3(new_tile[q] / new_block_scale);
                update_abs_sum += std::fabs(new_tile[q] - old_tile[q]);
            }
        }
    }
    *mean_abs_update = update_abs_sum / static_cast<float>(rows * cols);
    return 1;
}

extern "C" int smaul_fp8_factored_update(
    std::uint8_t* codes, float* scales, const float* grad,
    void* m, void* v_row, void* v_col,
    std::size_t rows, std::size_t cols, std::size_t tile,
    int state_bf16, int step, float beta_m, float beta_v,
    float lr, float wd, float eps, float update_clip,
    float grad_multiplier, float* mean_abs_update
) {
    if (state_bf16) {
        return fp8_factored_update_impl<true>(
            codes, scales, grad, m, v_row, v_col, rows, cols, tile, step,
            beta_m, beta_v, lr, wd, eps, update_clip, grad_multiplier,
            mean_abs_update);
    }
    return fp8_factored_update_impl<false>(
        codes, scales, grad, m, v_row, v_col, rows, cols, tile, step,
        beta_m, beta_v, lr, wd, eps, update_clip, grad_multiplier,
        mean_abs_update);
}


static inline std::uint64_t smaul_splitmix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

extern "C" int smaul_fp8_clone_perturb(
    const std::uint8_t* source_codes, const float* source_scales,
    std::uint8_t* output_codes, float* output_scales,
    std::size_t rows, std::size_t cols, std::size_t tile,
    float perturb, std::uint64_t seed
) {
    if (!source_codes || !source_scales || !output_codes || !output_scales ||
        rows == 0 || cols == 0 || tile == 0 ||
        !std::isfinite(perturb) || perturb < 0.0f || perturb >= 1.0f) {
        return 0;
    }
    const std::size_t nblocks = (cols + tile - 1) / tile;
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t b = 0; b < nblocks; ++b) {
            const std::size_t c0 = b * tile;
            const std::size_t c1 = std::min(cols, c0 + tile);
            const float source_scale = source_scales[r * nblocks + b];
            float amax = 1e-12f;
            for (std::size_t col = c0; col < c1; ++col) {
                const std::size_t index = r * cols + col;
                const float source = e4m3_to_f32(source_codes[index]) * source_scale;
                const std::uint64_t random_bits = smaul_splitmix64(
                    seed ^ (static_cast<std::uint64_t>(index) * 0xD6E8FEB86659FD93ULL)
                );
                const float sign = (random_bits >> 63) ? 1.0f : -1.0f;
                const float updated = source * (1.0f + perturb * sign);
                if (!std::isfinite(updated)) return 0;
                amax = std::max(amax, std::fabs(updated));
            }
            const float scale = amax / 448.0f;
            output_scales[r * nblocks + b] = scale;
            for (std::size_t col = c0; col < c1; ++col) {
                const std::size_t index = r * cols + col;
                const float source = e4m3_to_f32(source_codes[index]) * source_scale;
                const std::uint64_t random_bits = smaul_splitmix64(
                    seed ^ (static_cast<std::uint64_t>(index) * 0xD6E8FEB86659FD93ULL)
                );
                const float sign = (random_bits >> 63) ? 1.0f : -1.0f;
                const float updated = source * (1.0f + perturb * sign);
                output_codes[index] = f32_to_e4m3(updated / scale);
            }
        }
    }
    return 1;
}
