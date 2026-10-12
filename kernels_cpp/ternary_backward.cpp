#include <cstddef>

// Computes dX = dY * W for W stored as row-scaled ternary values.
// Exact zero weights are skipped; nonzero ternary values are accumulated
// without changing the full-precision gradient result beyond FP32 summation.
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
        float* gi = grad_input + r * in_features;
        for (std::size_t d = 0; d < in_features; ++d) {
            float sum = 0.0f;
            for (std::size_t o = 0; o < out_features; ++o) {
                const float w = weight[o * in_features + d];
                if (w != 0.0f) {
                    sum += go[o] * w;
                }
            }
            gi[d] = sum;
        }
    }
}
