#include <cstddef>

// Computes dX = dY * W for row-major W[out_features, in_features].
// The weight values may be row-scaled ternary values. The inner loop walks
// contiguous memory so the compiler can vectorize the common accumulation
// path; exact zeros are skipped without changing the mathematical result.
extern "C" void smaul_ternary_backward_input(
    const float* grad_out,
    const float* weight,
    float* grad_input,
    std::size_t rows,
    std::size_t in_features,
    std::size_t out_features
) {
    if (!grad_out || !weight || !grad_input ||
        rows == 0 || in_features == 0 || out_features == 0) {
        return;
    }

    for (std::size_t r = 0; r < rows; ++r) {
        const float* go = grad_out + r * out_features;
        const float* w = weight;
        float* gi = grad_input + r * in_features;

        for (std::size_t d = 0; d < in_features; ++d) {
            gi[d] = 0.0f;
        }

        for (std::size_t o = 0; o < out_features; ++o) {
            const float grad = go[o];
            const float* wr = w + o * in_features;
            for (std::size_t d = 0; d < in_features; ++d) {
                const float value = wr[d];
                if (value != 0.0f) {
                    gi[d] += grad * value;
                }
            }
        }
    }
}
