// SmaulBRAIN V2 portable native kernels (scalar C++; no AVX2/AVX-512 required).
// Build: python -m v2.kernels.build   (explicit, actionable errors on failure)
// All ops have Python scalar references in smaulbrain/math_ref.py; parity tested.
#include <cmath>
#include <cstdint>

extern "C" {

// y[i] = x[i] / sqrt(mean(x^2)+eps) * g[i]
void rmsnorm_fwd(const float* x, const float* g, float* y, int n, float eps) {
    double ms = 0.0;
    for (int i = 0; i < n; ++i) ms += (double)x[i] * x[i];
    ms /= n;
    double inv = 1.0 / std::sqrt(ms + eps);
    for (int i = 0; i < n; ++i) y[i] = (float)(x[i] * inv * g[i]);
}

// pack fp32 weights to 2-bit ternary codes, 4 per byte; returns scale+thresh
void ternary_pack(const float* W, uint8_t* packed, int n, float* scale_out, float* thresh_out) {
    double mean = 0.0;
    for (int i = 0; i < n; ++i) mean += std::fabs((double)W[i]);
    mean /= n;
    float th = (float)(0.7 * mean);
    double sabs = 0.0; int cnt = 0;
    for (int i = 0; i < n; ++i) {
        int q = (W[i] > th) ? 1 : ((W[i] < -th) ? -1 : 0);
        if (q) { sabs += std::fabs((double)W[i]); ++cnt; }
    }
    float s = cnt ? (float)(sabs / cnt) : (float)mean;
    if (!(s > 0)) s = 1.0f;
    int npad = (4 - (n % 4)) % 4;
    int total = n + npad;
    for (int i = 0; i < total / 4; ++i) {
        uint8_t b = 0;
        for (int k = 0; k < 4; ++k) {
            int idx = 4 * i + k;
            int q = 1; // code for 0
            if (idx < n) q = (W[idx] > th) ? 2 : ((W[idx] < -th) ? 0 : 1);
            b |= (uint8_t)(q << (2 * k));
        }
        packed[i] = b;
    }
    *scale_out = s; *thresh_out = th;
}

// y = x @ Wq * scale, W row-major (rows x cols), codes packed 2-bit
void packed_matvec(const float* x, const uint8_t* packed, float* y,
                   int rows, int cols, float scale) {
    for (int j = 0; j < cols; ++j) {
        double acc = 0.0;
        for (int i = 0; i < rows; ++i) {
            int pos = i * cols + j;
            int q = ((packed[pos >> 2] >> (2 * (pos & 3))) & 3) - 1;
            if (q > 1) q = 1;
            if (q) acc += (double)x[i] * q;
        }
        y[j] = (float)(acc * scale);
    }
}

// recurrent state update: s = alpha*s + k*v ; y = sigmoid(r)*(s + u*k*v)
void recurrent_step(const float* s_prev, const float* k, const float* v, const float* r,
                    const float* alpha, const float* u, float* s_new, float* y, int d) {
    for (int i = 0; i < d; ++i) {
        double kv = (double)k[i] * v[i];
        double s = (double)alpha[i] * s_prev[i] + kv;
        s_new[i] = (float)s;
        double g = 1.0 / (1.0 + std::exp(-(double)r[i]));
        y[i] = (float)(g * (s + (double)u[i] * kv));
    }
}

// expert clone with additive Gaussian noise (Box-Muller, seeded LCG)
void clone_perturb(const float* src, float* dst, int n, unsigned seed, float sigma, int mutate) {
    unsigned s = seed ? seed : 0x9E3779B9u;
    for (int i = 0; i < n; ++i) {
        float v = src[i];
        if (mutate) {
            s = 1664525u * s + 1013904223u;
            double u1 = ((s >> 8) + 1.0) / 16777217.0;
            s = 1664525u * s + 1013904223u;
            double u2 = ((s >> 8) + 1.0) / 16777217.0;
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
            v += (float)(z * sigma);
        }
        dst[i] = v;
    }
}

} // extern "C"
