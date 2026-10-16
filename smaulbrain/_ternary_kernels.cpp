// Portable packed ternary kernels.
// Matrix W is row-major (din, dout), with four 2-bit codes per byte.
// Codes 0,1,2,3 map to -1,0,+1,+1 (3 is reserved by the packer).
// Compile with -mavx only when the host OS/CPU advertises AVX. AVX2 and
// AVX-512 are not used. The SSE2/scalar implementation remains available.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

static constexpr float kTritValue[4] = {-1.0f, 0.0f, 1.0f, 1.0f};

// ---------------------------------------------------------------------------
// Tiny persistent thread pool. Worker count = SMAULBRAIN_THREADS if set, else
// hardware_concurrency (capped at 16). SMAULBRAIN_THREADS=1 disables threading.
// Workers spin briefly after a job (decode issues many back-to-back matmuls)
// and then sleep, so idle processes burn no CPU.
class Pool {
public:
    // Intentionally leaked: detached workers must outlive static destruction.
    static Pool& get() { static Pool* p = new Pool(); return *p; }
    unsigned size() const noexcept { return nthreads_; }

    void run(std::size_t n_items, const std::function<void(std::size_t)>& fn) {
        if (n_items == 0) return;
        if (nthreads_ <= 1 || n_items == 1) {
            for (std::size_t i = 0; i < n_items; ++i) fn(i);
            return;
        }
        std::lock_guard<std::mutex> run_lock(run_mu_);
        fn_ = &fn;
        n_items_ = n_items;
        next_.store(0, std::memory_order_relaxed);
        const unsigned helpers = static_cast<unsigned>(
            std::min<std::size_t>(nthreads_ - 1, n_items - 1));
        remaining_.store(helpers, std::memory_order_relaxed);
        wanted_.store(helpers, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            generation_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_all();
        work();
        while (remaining_.load(std::memory_order_acquire) != 0) {
            for (int k = 0; k < 64; ++k) cpu_relax();
            if (remaining_.load(std::memory_order_acquire) == 0) break;
            std::this_thread::yield();
        }
    }

private:
    Pool() {
        unsigned n = std::thread::hardware_concurrency();
        if (n == 0) n = 1;
        if (const char* e = std::getenv("SMAULBRAIN_THREADS")) {
            const long v = std::strtol(e, nullptr, 10);
            if (v >= 1) n = static_cast<unsigned>(v);
        }
        nthreads_ = std::min(n, 16u);
        for (unsigned i = 1; i < nthreads_; ++i)
            workers_.emplace_back([this] { loop(); });
        for (auto& t : workers_) t.detach();  // process exit must not block on workers
    }
    static inline void cpu_relax() noexcept {
#if defined(__SSE2__)
        _mm_pause();
#endif
    }
    void work() {
        for (;;) {
            const std::size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_items_) break;
            (*fn_)(i);
        }
    }
    void loop() {
        unsigned seen = 0;
        for (;;) {
            // Spin ~50us for the next job, then sleep on the condvar.
            bool got = false;
            for (int k = 0; k < 20000; ++k) {
                if (generation_.load(std::memory_order_acquire) != seen) { got = true; break; }
                cpu_relax();
            }
            if (!got) {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return generation_.load(std::memory_order_acquire) != seen; });
            }
            seen = generation_.load(std::memory_order_acquire);
            // Only `wanted_` helpers participate in each job.
            unsigned w = wanted_.load(std::memory_order_relaxed);
            while (w > 0 && !wanted_.compare_exchange_weak(w, w - 1, std::memory_order_acq_rel)) {}
            if (w == 0) continue;
            work();
            remaining_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }
    unsigned nthreads_ = 1;
    std::vector<std::thread> workers_;
    std::mutex mu_, run_mu_;
    std::condition_variable cv_;
    std::atomic<unsigned> generation_{0}, remaining_{0}, wanted_{0};
    std::atomic<std::size_t> next_{0};
    const std::function<void(std::size_t)>* fn_ = nullptr;
    std::size_t n_items_ = 0;
};

static inline bool worth_threads(std::size_t macs) noexcept {
    return macs >= (std::size_t{1} << 20) && Pool::get().size() > 1;
}

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

#if defined(__AVX__)
    if (batch == 1 && (dout & 7u) == 0u) {
        // Block input rows outside output tiles so the current row group stays
        // cache-hot while scanning adjacent output columns. The output-tile
        // outer loop revisited widely spaced row cache lines for each tile.
        constexpr std::size_t kInputTile = 16;
        constexpr std::size_t kOutputTile = 64;
        std::fill(y, y + dout, 0.0f);
        for (std::size_t i0 = 0; i0 < din; i0 += kInputTile) {
            const std::size_t iend = std::min(i0 + kInputTile, din);
            for (std::size_t j0 = 0; j0 < dout; j0 += kOutputTile) {
                const std::size_t width = std::min(kOutputTile, dout - j0);
                const std::size_t nvec = width / 8;
                __m256 acc[8];
                for (std::size_t v = 0; v < nvec; ++v) {
                    acc[v] = _mm256_setzero_ps();
                }
                for (std::size_t i = i0; i < iend; ++i) {
                    const __m256 xv = _mm256_set1_ps(x[i]);
                    const std::size_t base = (i * dout + j0) >> 2;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
                    for (std::size_t v = 0; v < nvec; ++v) {
                        const __m256 wv = trit_vector8(packed[base + v * 2],
                                                       packed[base + v * 2 + 1]);
                        acc[v] = _mm256_add_ps(acc[v], _mm256_mul_ps(xv, wv));
                    }
                }
                for (std::size_t v = 0; v < nvec; ++v) {
                    const std::size_t out_idx = j0 + v * 8;
                    const __m256 prior = _mm256_loadu_ps(y + out_idx);
                    _mm256_storeu_ps(y + out_idx, _mm256_add_ps(prior, acc[v]));
                }
            }
        }
        const __m256 scalev = _mm256_set1_ps(scale);
        for (std::size_t j = 0; j < dout; j += 8) {
            const __m256 sum = _mm256_loadu_ps(y + j);
            _mm256_storeu_ps(y + j, _mm256_mul_ps(sum, scalev));
        }
        return 0;
    }
    if ((dout & 7u) == 0u) {
        // Process two tokens at once so packed bytes and decoded trit vectors
        // are reused for both rows. This is especially useful for prompt
        // prefill, where the same weights multiply multiple input positions.
        constexpr std::size_t kTile = 32;
        constexpr std::size_t kBatchTile = 2;
        for (std::size_t j0 = 0; j0 < dout; j0 += kTile) {
            const std::size_t width = std::min(kTile, dout - j0);
            const std::size_t nvec = width / 8;
            for (std::size_t b0 = 0; b0 < batch; b0 += kBatchTile) {
                const bool has_second = b0 + 1 < batch;
                const float* in0 = x + b0 * din;
                const float* in1 = has_second ? x + (b0 + 1) * din : nullptr;
                float* out0 = y + b0 * dout;
                float* out1 = has_second ? y + (b0 + 1) * dout : nullptr;
                __m256 acc0[4], acc1[4];
                for (std::size_t v = 0; v < nvec; ++v) {
                    acc0[v] = _mm256_setzero_ps();
                    if (has_second) acc1[v] = _mm256_setzero_ps();
                }
                for (std::size_t i = 0; i < din; ++i) {
                    const __m256 x0 = _mm256_set1_ps(in0[i]);
                    const __m256 x1 = has_second ? _mm256_set1_ps(in1[i]) : _mm256_setzero_ps();
                    const std::size_t base = (i * dout + j0) >> 2;
                    for (std::size_t v = 0; v < nvec; ++v) {
                        const __m256 wv = trit_vector8(packed[base + v * 2],
                                                       packed[base + v * 2 + 1]);
                        acc0[v] = _mm256_add_ps(acc0[v], _mm256_mul_ps(x0, wv));
                        if (has_second) {
                            acc1[v] = _mm256_add_ps(acc1[v], _mm256_mul_ps(x1, wv));
                        }
                    }
                }
                const __m256 scalev = _mm256_set1_ps(scale);
                for (std::size_t v = 0; v < nvec; ++v) {
                    _mm256_storeu_ps(out0 + j0 + v * 8, _mm256_mul_ps(acc0[v], scalev));
                    if (has_second) {
                        _mm256_storeu_ps(out1 + j0 + v * 8,
                                         _mm256_mul_ps(acc1[v], scalev));
                    }
                }
            }
        }
        return 0;
    }
#endif

    for (std::size_t b = 0; b < batch; ++b) {
        const float* in = x + b * din;
        float* out = y + b * dout;
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


extern "C" int smaulbrain_pack_ternary(
    const float* weights, std::uint8_t* packed, std::size_t n,
    float* scale_out, float* threshold_out) noexcept {
    if (!weights || !packed || !scale_out || !threshold_out || n == 0) return 1;

    double sum_abs = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_abs += std::fabs(static_cast<double>(weights[i]));
    }
    const double mean_abs = sum_abs / static_cast<double>(n);
    const double threshold = 0.7 * mean_abs;

    // Combine scale accumulation and code emission into one pass. The threshold
    // is already known, so we can classify each weight, accumulate the nonzero
    // absolute values, and write its packed trit without a third matrix scan.
    double selected_abs_sum = 0.0;
    std::size_t selected_count = 0;
    const std::size_t nbytes = (n + 3u) / 4u;
    for (std::size_t b = 0; b < nbytes; ++b) {
        unsigned byte = 0u;
        for (unsigned lane = 0; lane < 4; ++lane) {
            const std::size_t idx = b * 4u + lane;
            unsigned code = 1u;
            if (idx < n) {
                const double v = static_cast<double>(weights[idx]);
                if (v > threshold) {
                    code = 2u;
                    selected_abs_sum += v;
                    ++selected_count;
                } else if (v < -threshold) {
                    code = 0u;
                    selected_abs_sum -= v;
                    ++selected_count;
                }
            }
            byte |= code << (lane * 2u);
        }
        packed[b] = static_cast<std::uint8_t>(byte);
    }
    double scale = selected_count
        ? selected_abs_sum / static_cast<double>(selected_count)
        : mean_abs;
    if (!(scale > 0.0)) scale = 1.0;
    *scale_out = static_cast<float>(scale);
    *threshold_out = static_cast<float>(threshold);
    return 0;
}

extern "C" int smaulbrain_recurrent_scan(
    const float* kv, const float* alpha, float* state, float* states,
    std::size_t T, std::size_t d) {
    if (!kv || !alpha || !state || !states || d == 0) return 1;
    // Time remains sequential, but each feature recurrence is independent.
    // Process one timestep at a time and vectorize across the feature axis.
    for (std::size_t t = 0; t < T; ++t) {
        const float* kvt = kv + t * d;
        float* st = states + t * d;
        std::size_t j = 0;
#if defined(__AVX__)
        for (; j + 8 <= d; j += 8) {
            const __m256 a = _mm256_loadu_ps(alpha + j);
            const __m256 s = _mm256_loadu_ps(state + j);
            const __m256 x = _mm256_loadu_ps(kvt + j);
            const __m256 next = _mm256_add_ps(_mm256_mul_ps(a, s), x);
            _mm256_storeu_ps(state + j, next);
            _mm256_storeu_ps(st + j, next);
        }
#elif defined(__SSE2__)
        for (; j + 4 <= d; j += 4) {
            const __m128 a = _mm_loadu_ps(alpha + j);
            const __m128 s = _mm_loadu_ps(state + j);
            const __m128 x = _mm_loadu_ps(kvt + j);
            const __m128 next = _mm_add_ps(_mm_mul_ps(a, s), x);
            _mm_storeu_ps(state + j, next);
            _mm_storeu_ps(st + j, next);
        }
#endif
        for (; j < d; ++j) {
            const float next = alpha[j] * state[j] + kvt[j];
            state[j] = next;
            st[j] = next;
        }
    }
    return 0;
}

extern "C" int smaulbrain_recurrent_backward_scan(
    const float* dS, const float* S, const float* alpha, float* dKV,
    double* dalpha, std::size_t T, std::size_t d) {
    if (!dS || !S || !alpha || !dKV || !dalpha || d == 0) return 1;
    // Use double recurrent accumulators to match the stable Python reference,
    // but keep all per-timestep loops inside native code instead of Python.
    std::vector<double> ds(d, 0.0);
    std::vector<double> dsdalpha(d, 0.0);
    std::vector<double> da(d, 0.0);
    for (std::size_t t = T; t-- > 0;) {
        const std::size_t base = t * d;
        for (std::size_t j = 0; j < d; ++j) {
            ds[j] = ds[j] * static_cast<double>(alpha[j]) +
                    static_cast<double>(dS[base + j]);
            dKV[base + j] = static_cast<float>(ds[j]);
        }
    }
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t base = t * d;
        const std::size_t prev = t > 0 ? (t - 1) * d : 0;
        for (std::size_t j = 0; j < d; ++j) {
            const double s_prev = t > 0 ? static_cast<double>(S[prev + j]) : 0.0;
            dsdalpha[j] = s_prev + static_cast<double>(alpha[j]) * dsdalpha[j];
            da[j] += static_cast<double>(dS[base + j]) * dsdalpha[j];
        }
    }
    for (std::size_t j = 0; j < d; ++j) dalpha[j] = da[j];
    return 0;
}

// ---------------------------------------------------------------------------
// Tiled layout (forward + input-gradient fast path).
//
// W (din, dout) is stored as ceil(dout/16) column tiles. Tile t holds, for
// every input row i, 4 consecutive bytes describing columns 16t..16t+15:
// [pos0, neg0, pos1, neg1] where pos/neg are 8-bit bitplanes (bit k = column
// 8v+k is +1 / -1). Each tile is one contiguous din*4 byte stream, so the
// kernels read memory strictly sequentially. Padding columns are 0 (zero
// weight). Set SB_PLANES=0 to use raw 2-bit codes in the same 4 bytes.
#ifndef SB_PLANES
#define SB_PLANES 0
#endif

static inline void tile_emit_codes(const unsigned* codes, std::uint8_t* o) noexcept {
#if SB_PLANES
    for (int v = 0; v < 2; ++v) {
        unsigned p = 0, n = 0;
        for (int k = 0; k < 8; ++k) {
            const unsigned c = codes[v * 8 + k];
            if (c >= 2u) p |= 1u << k;
            else if (c == 0u) n |= 1u << k;
        }
        o[2 * v] = static_cast<std::uint8_t>(p);
        o[2 * v + 1] = static_cast<std::uint8_t>(n);
    }
#else
    for (int k = 0; k < 4; ++k)
        o[k] = static_cast<std::uint8_t>(codes[4 * k] | (codes[4 * k + 1] << 2) |
                                         (codes[4 * k + 2] << 4) | (codes[4 * k + 3] << 6));
#endif
}

extern "C" int smaulbrain_retile_ternary(
    const std::uint8_t* lin, std::uint8_t* out, std::size_t din,
    std::size_t dout) noexcept {
    if (!lin || !out || din == 0 || dout == 0) return 1;
    const std::size_t ntiles = (dout + 15) / 16;
#if SB_PLANES
    static const auto nib = [] {
        std::array<std::array<std::uint8_t, 2>, 256> t{};
        for (unsigned b = 0; b < 256; ++b) {
            unsigned p = 0, n = 0;
            for (unsigned k = 0; k < 4; ++k) {
                const unsigned c = (b >> (2 * k)) & 3u;
                if (c >= 2u) p |= 1u << k;
                else if (c == 0u) n |= 1u << k;
            }
            t[b] = {static_cast<std::uint8_t>(p), static_cast<std::uint8_t>(n)};
        }
        return t;
    }();
#endif
    const bool aligned = (dout & 3u) == 0u;
    for (std::size_t t = 0; t < ntiles; ++t) {
        const std::size_t j0 = t * 16;
        const std::size_t width = std::min<std::size_t>(16, dout - j0);
        std::uint8_t* tile = out + t * din * 4;
        for (std::size_t i = 0; i < din; ++i) {
            std::uint8_t* o = tile + i * 4;
            if (aligned && width == 16) {
                const std::uint8_t* b = lin + ((i * dout + j0) >> 2);
#if SB_PLANES
                o[0] = static_cast<std::uint8_t>(nib[b[0]][0] | (nib[b[1]][0] << 4));
                o[1] = static_cast<std::uint8_t>(nib[b[0]][1] | (nib[b[1]][1] << 4));
                o[2] = static_cast<std::uint8_t>(nib[b[2]][0] | (nib[b[3]][0] << 4));
                o[3] = static_cast<std::uint8_t>(nib[b[2]][1] | (nib[b[3]][1] << 4));
#else
                o[0] = b[0]; o[1] = b[1]; o[2] = b[2]; o[3] = b[3];
#endif
            } else {
                unsigned codes[16];
                for (std::size_t c = 0; c < 16; ++c)
                    codes[c] = c < width ? ternary_code(lin, i * dout + j0 + c) : 1u;
                tile_emit_codes(codes, o);
            }
        }
    }
    return 0;
}

#if defined(__AVX__)
#if SB_PLANES
struct alignas(32) MaskLut { float v[256][8]; };
static const MaskLut& mask_lut() noexcept {
    static const MaskLut t = [] {
        MaskLut m{};
        for (unsigned b = 0; b < 256; ++b)
            for (unsigned k = 0; k < 8; ++k) m.v[b][k] = ((b >> k) & 1u) ? 1.0f : 0.0f;
        return m;
    }();
    return t;
}
// p points at [pos, neg] for 8 columns -> +1/0/-1 floats in one sub.
static inline __m256 dec8(const MaskLut& L, const std::uint8_t* p) noexcept {
    return _mm256_sub_ps(_mm256_load_ps(L.v[p[0]]), _mm256_load_ps(L.v[p[1]]));
}
#else
struct MaskLut {};
static inline const MaskLut& mask_lut() noexcept { static const MaskLut t{}; return t; }
static inline __m256 dec8(const MaskLut&, const std::uint8_t* p) noexcept {
    return trit_vector8(p[0], p[1]);
}
#endif

template <int BT>
static inline void tiled_fwd_tile(const MaskLut& L, const float* x, const std::uint8_t* tile,
                                  float scale, float* y, std::size_t b0, std::size_t din,
                                  std::size_t dout, std::size_t j0, std::size_t valid) noexcept {
    __m256 a0[BT], a1[BT];
    for (int b = 0; b < BT; ++b) { a0[b] = _mm256_setzero_ps(); a1[b] = _mm256_setzero_ps(); }
    const float* in = x + b0 * din;
    for (std::size_t i = 0; i < din; ++i) {
        const std::uint8_t* p = tile + i * 4;
        const __m256 w0 = dec8(L, p);
        const __m256 w1 = dec8(L, p + 2);
        for (int b = 0; b < BT; ++b) {
            const __m256 xb = _mm256_set1_ps(in[b * din + i]);
            a0[b] = _mm256_add_ps(a0[b], _mm256_mul_ps(xb, w0));
            a1[b] = _mm256_add_ps(a1[b], _mm256_mul_ps(xb, w1));
        }
    }
    const __m256 sv = _mm256_set1_ps(scale);
    for (int b = 0; b < BT; ++b) {
        float* out = y + (b0 + b) * dout + j0;
        const __m256 r0 = _mm256_mul_ps(a0[b], sv), r1 = _mm256_mul_ps(a1[b], sv);
        if (valid == 16) {
            _mm256_storeu_ps(out, r0);
            _mm256_storeu_ps(out + 8, r1);
        } else {
            alignas(32) float tmp[16];
            _mm256_store_ps(tmp, r0);
            _mm256_store_ps(tmp + 8, r1);
            for (std::size_t c = 0; c < valid; ++c) out[c] = tmp[c];
        }
    }
}
#endif  // __AVX__

static inline float tiled_weight(const std::uint8_t* tile, std::size_t i, unsigned c) noexcept {
    const std::uint8_t* p = tile + i * 4;
#if SB_PLANES
    const unsigned v = c >> 3, k = c & 7u;
    return static_cast<float>(((p[2 * v] >> k) & 1u)) - static_cast<float>(((p[2 * v + 1] >> k) & 1u));
#else
    const unsigned code = (static_cast<unsigned>(p[c >> 2]) >> ((c & 3u) * 2u)) & 3u;
    return kTritValue[code];
#endif
}

extern "C" int smaulbrain_tiled_matmat(
    const float* x, const std::uint8_t* tiled, float scale, float* y,
    std::size_t batch, std::size_t din, std::size_t dout) noexcept {
    if (!x || !tiled || !y || din == 0 || dout == 0) return 1;
    const std::size_t ntiles = (dout + 15) / 16;
#if defined(__AVX__)
    const MaskLut& L = mask_lut();
    auto do_tile = [&](std::size_t t) {
        const std::size_t j0 = t * 16;
        const std::size_t valid = std::min<std::size_t>(16, dout - j0);
        const std::uint8_t* tile = tiled + t * din * 4;
        for (std::size_t b0 = 0; b0 < batch;) {
            const std::size_t left = batch - b0;
            const std::size_t bt = left >= 4 ? 4 : left;
            switch (bt) {
                case 4: tiled_fwd_tile<4>(L, x, tile, scale, y, b0, din, dout, j0, valid); break;
                case 3: tiled_fwd_tile<3>(L, x, tile, scale, y, b0, din, dout, j0, valid); break;
                case 2: tiled_fwd_tile<2>(L, x, tile, scale, y, b0, din, dout, j0, valid); break;
                default: tiled_fwd_tile<1>(L, x, tile, scale, y, b0, din, dout, j0, valid); break;
            }
            b0 += bt;
        }
    };
#else
    auto do_tile = [&](std::size_t t) {
        const std::size_t j0 = t * 16;
        const unsigned valid = static_cast<unsigned>(std::min<std::size_t>(16, dout - j0));
        const std::uint8_t* tile = tiled + t * din * 4;
        for (std::size_t b = 0; b < batch; ++b) {
            for (unsigned c = 0; c < valid; ++c) {
                float acc = 0.0f;
                for (std::size_t i = 0; i < din; ++i) acc += x[b * din + i] * tiled_weight(tile, i, c);
                y[b * dout + j0 + c] = acc * scale;
            }
        }
    };
#endif
    if (worth_threads(batch * din * dout)) {
        Pool::get().run(ntiles, std::function<void(std::size_t)>(do_tile));
    } else {
        for (std::size_t t = 0; t < ntiles; ++t) do_tile(t);
    }
    return 0;
}

extern "C" int smaulbrain_tiled_transpose_matmat(
    const float* dy, const std::uint8_t* tiled, float scale, float* dx,
    std::size_t batch, std::size_t din, std::size_t dout) noexcept {
    if (!dy || !tiled || !dx || din == 0 || dout == 0) return 1;
    const std::size_t ntiles = (dout + 15) / 16;
    const std::size_t pw = ntiles * 16;
    // dy padded to whole tiles once: [batch][ntiles*16].
    std::vector<float> dyp(batch * pw, 0.0f);
    for (std::size_t b = 0; b < batch; ++b)
        std::copy(dy + b * dout, dy + (b + 1) * dout, dyp.data() + b * pw);
    // Each work item owns a disjoint range of input rows i, so dx writes never
    // overlap between threads. Every tile is streamed sequentially per range.
    constexpr std::size_t kRows = 64;
    const std::size_t nitems = (din + kRows - 1) / kRows;
    auto do_range = [&](std::size_t item) {
        const std::size_t i0 = item * kRows;
        const std::size_t i1 = std::min(din, i0 + kRows);
        for (std::size_t b = 0; b < batch; ++b)
            std::fill(dx + b * din + i0, dx + b * din + i1, 0.0f);
#if defined(__AVX__)
        const MaskLut& L = mask_lut();
        for (std::size_t t = 0; t < ntiles; ++t) {
            const std::uint8_t* tile = tiled + t * din * 4;
            const float* dyt = dyp.data() + t * 16;
            std::size_t i = i0;
            for (; i + 4 <= i1; i += 4) {
                __m256 w0[4], w1[4];
                for (int r = 0; r < 4; ++r) {
                    const std::uint8_t* p = tile + (i + r) * 4;
                    w0[r] = dec8(L, p);
                    w1[r] = dec8(L, p + 2);
                }
                for (std::size_t b = 0; b < batch; ++b) {
                    const __m256 d0 = _mm256_loadu_ps(dyt + b * pw);
                    const __m256 d1 = _mm256_loadu_ps(dyt + b * pw + 8);
                    __m256 a[4];
                    for (int r = 0; r < 4; ++r)
                        a[r] = _mm256_add_ps(_mm256_mul_ps(d0, w0[r]), _mm256_mul_ps(d1, w1[r]));
                    const __m256 h = _mm256_hadd_ps(_mm256_hadd_ps(a[0], a[1]),
                                                    _mm256_hadd_ps(a[2], a[3]));
                    const __m128 sm = _mm_add_ps(_mm256_castps256_ps128(h),
                                                 _mm256_extractf128_ps(h, 1));
                    float* o = dx + b * din + i;
                    _mm_storeu_ps(o, _mm_add_ps(_mm_loadu_ps(o), sm));
                }
            }
            for (; i < i1; ++i) {
                const std::uint8_t* p = tile + i * 4;
                const __m256 w0 = dec8(L, p), w1 = dec8(L, p + 2);
                for (std::size_t b = 0; b < batch; ++b) {
                    const __m256 d0 = _mm256_loadu_ps(dyt + b * pw);
                    const __m256 d1 = _mm256_loadu_ps(dyt + b * pw + 8);
                    dx[b * din + i] += horizontal_sum(
                        _mm256_add_ps(_mm256_mul_ps(d0, w0), _mm256_mul_ps(d1, w1)));
                }
            }
        }
#else
        for (std::size_t t = 0; t < ntiles; ++t) {
            const std::uint8_t* tile = tiled + t * din * 4;
            const float* dyt = dyp.data() + t * 16;
            for (std::size_t i = i0; i < i1; ++i)
                for (std::size_t b = 0; b < batch; ++b) {
                    float acc = 0.0f;
                    for (unsigned c = 0; c < 16; ++c) acc += dyt[b * pw + c] * tiled_weight(tile, i, c);
                    dx[b * din + i] += acc;
                }
        }
#endif
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t i = i0; i < i1; ++i) dx[b * din + i] *= scale;
    };
    if (worth_threads(batch * din * dout)) {
        Pool::get().run(nitems, std::function<void(std::size_t)>(do_range));
    } else {
        for (std::size_t item = 0; item < nitems; ++item) do_range(item);
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Fused fp32 -> tiled ternary packer. Equivalent to smaulbrain_pack_ternary()
// followed by smaulbrain_retile_ternary(), but SIMD, threaded, and without the
// intermediate row-major buffer. Float thresholds are rounded down so that
// (v > t_f) <=> (v > double threshold) for every float v: codes are identical
// to the scalar reference.
extern "C" int smaulbrain_pack_ternary_tiled(
    const float* w, std::uint8_t* out, std::size_t din, std::size_t dout,
    float* scale_out, float* threshold_out) noexcept {
    if (!w || !out || !scale_out || !threshold_out || din == 0 || dout == 0) return 1;
#if defined(__AVX__)
    const std::size_t n = din * dout;
    constexpr std::size_t kChunk = std::size_t{1} << 16;
    const std::size_t nchunks = (n + kChunk - 1) / kChunk;
    std::vector<double> part_abs(nchunks, 0.0);
    auto abs_chunk = [&](std::size_t c) {
        const std::size_t b = c * kChunk, e = std::min(n, b + kChunk);
        const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256d acc = _mm256_setzero_pd();
        std::size_t k = b;
        for (; k + 8 <= e; k += 8) {
            const __m256 a = _mm256_and_ps(_mm256_loadu_ps(w + k), absmask);
            acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_castps256_ps128(a)));
            acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1)));
        }
        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, acc);
        double sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        for (; k < e; ++k) sum += std::fabs(static_cast<double>(w[k]));
        part_abs[c] = sum;
    };
    const bool threaded = worth_threads(n);
    if (threaded) Pool::get().run(nchunks, std::function<void(std::size_t)>(abs_chunk));
    else for (std::size_t c = 0; c < nchunks; ++c) abs_chunk(c);
    double sum_abs = 0.0;
    for (double v : part_abs) sum_abs += v;
    const double mean_abs = sum_abs / static_cast<double>(n);
    const double threshold = 0.7 * mean_abs;
    float tf = static_cast<float>(threshold);
    if (static_cast<double>(tf) > threshold) tf = std::nextafterf(tf, 0.0f);

    static const auto luts = [] {
        std::array<std::array<std::uint16_t, 256>, 2> t{};
        for (unsigned b = 0; b < 256; ++b) {
            unsigned pos = 0, zero = 0;
            for (unsigned k = 0; k < 8; ++k) {
                if ((b >> k) & 1u) { pos |= 2u << (2 * k); zero |= 1u << (2 * k); }
            }
            t[0][b] = static_cast<std::uint16_t>(pos);   // bit set -> code 2 ('10')
            t[1][b] = static_cast<std::uint16_t>(zero);  // bit set -> code 1 ('01')
        }
        return t;
    }();
    const std::size_t ntiles = (dout + 15) / 16;
    std::vector<double> tile_sum(ntiles, 0.0);
    std::vector<std::size_t> tile_cnt(ntiles, 0);
    auto do_tile = [&](std::size_t t) {
        const std::size_t j0 = t * 16;
        const std::size_t valid = std::min<std::size_t>(16, dout - j0);
        std::uint8_t* tile = out + t * din * 4;
        const __m256 th = _mm256_set1_ps(tf), nth = _mm256_set1_ps(-tf);
        const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256d acc = _mm256_setzero_pd();
        std::size_t cnt = 0;
        alignas(32) float tmp[16] = {0};
        for (std::size_t i = 0; i < din; ++i) {
            const float* row = w + i * dout + j0;
            __m256 v0, v1;
            if (valid == 16) {
                v0 = _mm256_loadu_ps(row);
                v1 = _mm256_loadu_ps(row + 8);
            } else {
                for (std::size_t c = 0; c < valid; ++c) tmp[c] = row[c];
                v0 = _mm256_load_ps(tmp);
                v1 = _mm256_load_ps(tmp + 8);
            }
            std::uint8_t* o = tile + i * 4;
            for (int v = 0; v < 2; ++v) {
                const __m256 x = v ? v1 : v0;
                const __m256 mp = _mm256_cmp_ps(x, th, _CMP_GT_OQ);
                const __m256 mn = _mm256_cmp_ps(x, nth, _CMP_LT_OQ);
                const unsigned p = static_cast<unsigned>(_mm256_movemask_ps(mp));
                const unsigned q = static_cast<unsigned>(_mm256_movemask_ps(mn));
                const __m256 sel = _mm256_and_ps(_mm256_and_ps(x, absmask), _mm256_or_ps(mp, mn));
                acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_castps256_ps128(sel)));
                acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_extractf128_ps(sel, 1)));
                cnt += static_cast<std::size_t>(__builtin_popcount(p | q));
#if SB_PLANES
                o[2 * v] = static_cast<std::uint8_t>(p);
                o[2 * v + 1] = static_cast<std::uint8_t>(q);
#else
                const unsigned code = luts[0][p] | luts[1][(~(p | q)) & 0xFFu];
                o[2 * v] = static_cast<std::uint8_t>(code & 0xFFu);
                o[2 * v + 1] = static_cast<std::uint8_t>(code >> 8);
#endif
            }
        }
        alignas(32) double lanes[4];
        _mm256_store_pd(lanes, acc);
        tile_sum[t] = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        tile_cnt[t] = cnt;
    };
    if (threaded) Pool::get().run(ntiles, std::function<void(std::size_t)>(do_tile));
    else for (std::size_t t = 0; t < ntiles; ++t) do_tile(t);
    double sel_sum = 0.0;
    std::size_t sel_cnt = 0;
    for (std::size_t t = 0; t < ntiles; ++t) { sel_sum += tile_sum[t]; sel_cnt += tile_cnt[t]; }
    double scale = sel_cnt ? sel_sum / static_cast<double>(sel_cnt) : mean_abs;
    if (!(scale > 0.0)) scale = 1.0;
    *scale_out = static_cast<float>(scale);
    *threshold_out = static_cast<float>(threshold);
    return 0;
#else
    std::vector<std::uint8_t> lin((din * dout + 3) / 4);
    const int st = smaulbrain_pack_ternary(w, lin.data(), din * dout, scale_out, threshold_out);
    if (st != 0) return st;
    return smaulbrain_retile_ternary(lin.data(), out, din, dout);
#endif
}
