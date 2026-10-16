// Fused SmaulOPT update for factored (2-D) parameters.
//
// Mirrors smaulopt.SmaulOPT.step() for a matrix W (rows x cols):
//   m <- b1*m + (1-b1)*g
//   R <- b2*R + (1-b2)*mean_j(g^2),  C <- b2*C + (1-b2)*mean_i(g^2)
//   upd = m / (sqrt(max(R_i*C_j/mean(R), 0)) + eps)
//   W  <- W - (lr*wd)*W ;  W <- W - lr*upd
// in two streaming passes over the matrix (the NumPy path makes ~10).
// All per-element arithmetic is IEEE float32 in the same operation order as
// the NumPy path (no FMA, no fast-math); only the row/column means differ,
// they accumulate in float64. Reductions use fixed 16-row blocks combined in
// block order, so results do not depend on the thread count.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace {
constexpr std::size_t kBlockRows = 16;

// Persistent worker pool (same pattern as _ternary_kernels.cpp). Creating
// threads per optimizer call costs more than the update itself at matrix
// sizes this project targets, so workers are started once per process. Worker
// count = SMAULBRAIN_THREADS if set (>=1), else hardware_concurrency capped at
// 16; SMAULBRAIN_THREADS=1 disables threading.
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
            // Spin briefly for the next job, then sleep on the condvar.
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

// Dynamic work distribution over [0, nblocks). Callers must make each block's
// output independent of which thread runs it, and combine per-block results in
// block order (as the passes below do), so results stay deterministic.
template <class F>
void parallel_blocks(std::size_t nblocks, bool allow_threads, F&& fn) {
    Pool& pool = Pool::get();
    if (!allow_threads || nblocks == 0 || pool.size() <= 1) {
        for (std::size_t b = 0; b < nblocks; ++b) fn(b);
        return;
    }
    std::function<void(std::size_t)> job = [&fn](std::size_t b) { fn(b); };
    pool.run(nblocks, job);
}
}  // namespace

extern "C" int smaulbrain_smaulopt_factored(
    float* w, float* m, const float* g, float* R, float* C,
    std::size_t rows, std::size_t cols,
    float beta1, float omb1, float beta2, float omb2, float eps, float lr,
    float decay, int want_norm, double* upd_sq_sum) noexcept {
    if (!w || !m || !g || !R || !C || rows == 0 || cols == 0) return 1;
    try {
        const std::size_t nblocks = (rows + kBlockRows - 1) / kBlockRows;
        const bool threaded = rows * cols >= (std::size_t{1} << 18);

        // ---- pass 1: moment update + g^2 row/column statistics -------------
        std::vector<double> row_sum(rows, 0.0);
        std::vector<float> col_part(nblocks * cols, 0.0f);
        parallel_blocks(nblocks, threaded, [&](std::size_t blk) {
            const std::size_t i0 = blk * kBlockRows, i1 = std::min(rows, i0 + kBlockRows);
            float* cp = col_part.data() + blk * cols;
            for (std::size_t i = i0; i < i1; ++i) {
                const float* gr = g + i * cols;
                float* mr = m + i * cols;
                double rs = 0.0;
                std::size_t j = 0;
#if defined(__AVX__)
                const __m256 b1 = _mm256_set1_ps(beta1), o1 = _mm256_set1_ps(omb1);
                __m256d acc = _mm256_setzero_pd();
                for (; j + 8 <= cols; j += 8) {
                    const __m256 gv = _mm256_loadu_ps(gr + j);
                    const __m256 mv = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(mr + j), b1),
                                                    _mm256_mul_ps(gv, o1));
                    _mm256_storeu_ps(mr + j, mv);
                    const __m256 g2 = _mm256_mul_ps(gv, gv);
                    _mm256_storeu_ps(cp + j, _mm256_add_ps(_mm256_loadu_ps(cp + j), g2));
                    acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_castps256_ps128(g2)));
                    acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_extractf128_ps(g2, 1)));
                }
                alignas(32) double lanes[4];
                _mm256_store_pd(lanes, acc);
                rs = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
#endif
                for (; j < cols; ++j) {
                    const float gv = gr[j];
                    mr[j] = mr[j] * beta1 + gv * omb1;
                    const float g2 = gv * gv;
                    cp[j] += g2;
                    rs += static_cast<double>(g2);
                }
                row_sum[i] = rs;
            }
        });

        // ---- statistics update (tiny) --------------------------------------
        std::vector<double> col_sum(cols, 0.0);
        for (std::size_t blk = 0; blk < nblocks; ++blk) {
            const float* cp = col_part.data() + blk * cols;
            for (std::size_t j = 0; j < cols; ++j) col_sum[j] += static_cast<double>(cp[j]);
        }
        double r_total = 0.0;
        for (std::size_t i = 0; i < rows; ++i) {
            const float row_mean = static_cast<float>(row_sum[i] / static_cast<double>(cols));
            R[i] = R[i] * beta2 + row_mean * omb2;
            r_total += static_cast<double>(R[i]);
        }
        for (std::size_t j = 0; j < cols; ++j) {
            const float col_mean = static_cast<float>(col_sum[j] / static_cast<double>(rows));
            C[j] = C[j] * beta2 + col_mean * omb2;
        }
        const double mean_r = r_total / static_cast<double>(rows);
        const float denom = static_cast<float>(mean_r > 1e-30 ? mean_r : 1e-30);

        // ---- pass 2: update + weight decay (+ optional update norm) --------
        std::vector<double> blk_sq(nblocks, 0.0);
        parallel_blocks(nblocks, threaded, [&](std::size_t blk) {
            const std::size_t i0 = blk * kBlockRows, i1 = std::min(rows, i0 + kBlockRows);
            double sq = 0.0;
            for (std::size_t i = i0; i < i1; ++i) {
                float* wr = w + i * cols;
                const float* mr = m + i * cols;
                const float ri = R[i];
                std::size_t j = 0;
#if defined(__AVX__)
                const __m256 rv = _mm256_set1_ps(ri), dv = _mm256_set1_ps(denom);
                const __m256 ev = _mm256_set1_ps(eps), lv = _mm256_set1_ps(lr);
                const __m256 cv = _mm256_set1_ps(decay), zero = _mm256_setzero_ps();
                __m256d acc = _mm256_setzero_pd();
                for (; j + 8 <= cols; j += 8) {
                    __m256 v = _mm256_mul_ps(rv, _mm256_loadu_ps(C + j));
                    v = _mm256_div_ps(v, dv);
                    v = _mm256_max_ps(zero, v);  // NaN-propagating like np.maximum
                    v = _mm256_add_ps(_mm256_sqrt_ps(v), ev);
                    const __m256 u = _mm256_div_ps(_mm256_loadu_ps(mr + j), v);
                    if (want_norm) {
                        const __m256 u2 = _mm256_mul_ps(u, u);
                        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_castps256_ps128(u2)));
                        acc = _mm256_add_pd(acc, _mm256_cvtps_pd(_mm256_extractf128_ps(u2, 1)));
                    }
                    __m256 wv = _mm256_loadu_ps(wr + j);
                    if (decay != 0.0f) wv = _mm256_sub_ps(wv, _mm256_mul_ps(cv, wv));
                    wv = _mm256_sub_ps(wv, _mm256_mul_ps(u, lv));
                    _mm256_storeu_ps(wr + j, wv);
                }
                alignas(32) double lanes[4];
                _mm256_store_pd(lanes, acc);
                sq += (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
#endif
                for (; j < cols; ++j) {
                    float v = ri * C[j];
                    v = v / denom;
                    v = (v != v) ? v : (v > 0.0f ? v : 0.0f);
                    v = std::sqrt(v) + eps;
                    const float u = mr[j] / v;
                    if (want_norm) sq += static_cast<double>(u * u);
                    float wv = wr[j];
                    if (decay != 0.0f) wv = wv - decay * wv;
                    wr[j] = wv - u * lr;
                }
            }
            blk_sq[blk] = sq;
        });
        double total = 0.0;
        for (double v : blk_sq) total += v;
        if (upd_sq_sum) *upd_sq_sum = total;
        return 0;
    } catch (...) {
        return 2;
    }
}
