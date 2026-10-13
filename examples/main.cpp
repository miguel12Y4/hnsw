#include "hnsw_index.h"

#include <iostream>
#include <random>

int main()
{
    using namespace HNSW;

    const int dim = 4;
    const int N = 1000;

    auto index = HNSWIndex::create(dim, 16, 200, metric_type2metric_func(MetricType::EUCLIDEAN_DISTANCE));

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(0, 1);

    for (int i = 0; i < N; ++i) {
        Tensor v(dim);
        for (int j = 0; j < dim; ++j) {
            v[j] = d(rng);
        }
        index->add_point(v, i);
    }

    std::cout << "Indexado " << index->size() << " puntos, layers=" << index->get_params().layers
              << " entry=" << index->get_params().entry_point_id << "\n";

    Tensor q(dim);
    for (int j = 0; j < dim; ++j) {
        q[j] = d(rng);
    }

    auto res = index->query(q, 15, 50, nullptr, nullptr);
    std::cout << "Query top 15:\n";
    auto ids = res.extract_n_min(15);
    for (auto& e : ids) {
        std::cout << "  id=" << e.node_id << " dist=" << e.distance
                  << " ext_id=" << index->get_node(e.node_id).external_id << "\n";
    }

    index->save("/tmp/hnsw_test");
    std::cout << "Guardado en /tmp/hnsw_test\n";

    auto loaded = HNSWIndex::load("/tmp/hnsw_test", metric_type2metric_func(MetricType::EUCLIDEAN_DISTANCE));
    std::cout << "Cargado: " << loaded->size() << " puntos\n";

    return 0;
}
