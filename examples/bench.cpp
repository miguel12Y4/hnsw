#include "hnsw_index.h"

#include <chrono>
#include <iostream>
#include <random>

int main()
{
    using namespace HNSW;

    const int dim = 128;
    const int N = 10000;

    auto idx = HNSWIndex::create(dim, 16, 200, metric_type2metric_func(MetricType::EUCLIDEAN_DISTANCE));

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(0, 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        Tensor v(dim);
        for (int j = 0; j < dim; ++j) {
            v[j] = d(rng);
        }
        idx->add_point(v);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Insert " << N << " x " << dim << "D: " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    Tensor q(dim);
    for (int j = 0; j < dim; ++j) {
        q[j] = d(rng);
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        idx->query(q, 10, 100, nullptr, nullptr);
    }
    t1 = std::chrono::high_resolution_clock::now();
    std::cout << "1000 queries: " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

    return 0;
}
