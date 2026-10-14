#include <algorithm>
#include <cstddef>
#include <cstdint>
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
