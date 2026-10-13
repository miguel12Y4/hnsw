#include "hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << ": " \
                      << #cond << "\n";                                 \
            ++failures;                                                 \
        }                                                               \
    } while (0)

HNSW::Tensor random_vec(std::mt19937& rng, int dim)
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    HNSW::Tensor v(dim);
    for (int i = 0; i < dim; ++i) {
        v[i] = dist(rng);
    }
    return v;
}

std::vector<HNSW::Entry> brute_force_top_k(
    const std::vector<HNSW::Tensor>& data,
    const HNSW::Tensor& query,
    std::size_t k
)
{
    std::vector<HNSW::Entry> all;
    all.reserve(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        all.emplace_back(HNSW::euclidean_distance(query, data[i]), static_cast<uint32_t>(i));
    }
    std::sort(all.begin(), all.end(), HNSW::EntryLess {});
    if (all.size() > k) {
        all.resize(k);
    }
    return all;
}

void test_metrics()
{
    const HNSW::Tensor a { 0.0f, 0.0f };
    const HNSW::Tensor b { 3.0f, 4.0f };
    CHECK(std::fabs(HNSW::euclidean_distance(a, b) - 5.0f) < 1e-5f);
    CHECK(std::fabs(HNSW::manhattan_distance(a, b) - 7.0f) < 1e-5f);
    const HNSW::Tensor c { 1.0f, 0.0f };
    const HNSW::Tensor d { 0.0f, 1.0f };
    CHECK(std::fabs(HNSW::cosine_distance(c, c)) < 1e-5f);
    CHECK(std::fabs(HNSW::cosine_distance(c, d) - 1.0f) < 1e-5f);
}

void test_default_external_id()
{
    auto index = HNSW::HNSWIndex::create(
        2, 4, 10, HNSW::metric_type2metric_func(HNSW::MetricType::EUCLIDEAN_DISTANCE)
    );
    const uint32_t id0 = index->add_point(HNSW::Tensor { 0.0f, 0.0f });
    const uint32_t id1 = index->add_point(HNSW::Tensor { 1.0f, 1.0f });
    CHECK(id0 == 0);
    CHECK(id1 == 1);
    CHECK(index->get_node(id0).external_id == 0);
    CHECK(index->get_node(id1).external_id == 1);
}

void test_dimension_mismatch()
{
    auto index = HNSW::HNSWIndex::create(
        4, 16, 200, HNSW::metric_type2metric_func(HNSW::MetricType::EUCLIDEAN_DISTANCE)
    );
    bool threw = false;
    try {
        index->add_point(HNSW::Tensor { 1.0f, 2.0f, 3.0f }, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

void test_empty_query()
{
    auto index = HNSW::HNSWIndex::create(
        4, 16, 200, HNSW::metric_type2metric_func(HNSW::MetricType::EUCLIDEAN_DISTANCE)
    );
    auto heap = index->query(HNSW::Tensor { 0.0f, 0.0f, 0.0f, 0.0f }, 5, 10, nullptr, nullptr);
    CHECK(heap.empty());
}

void test_insert_and_recall()
{
    const int dim = 8;
    const int n = 500;
    const std::size_t k = 10;
    const uint64_t ef = 100;

    std::mt19937 rng(1234);
    std::vector<HNSW::Tensor> data;
    data.reserve(n);

    auto index = HNSW::HNSWIndex::create(
        dim, 16, 200, HNSW::metric_type2metric_func(HNSW::MetricType::EUCLIDEAN_DISTANCE)
    );
    for (int i = 0; i < n; ++i) {
        data.push_back(random_vec(rng, dim));
        const uint32_t node_id = index->add_point(data.back(), i);
        CHECK(node_id == static_cast<uint32_t>(i));
        CHECK(index->get_node(node_id).external_id == static_cast<uint64_t>(i));
    }
    CHECK(index->size() == static_cast<std::size_t>(n));

    std::size_t hits = 0;
    std::size_t total = 0;
    for (int q = 0; q < 20; ++q) {
        const auto query = random_vec(rng, dim);
        const auto expected = brute_force_top_k(data, query, k);
        auto heap = index->query(query, k, ef, nullptr, nullptr);
        const auto got = heap.extract_n_min(k);
        CHECK(got.size() == k);
        for (const auto& e : expected) {
            for (const auto& g : got) {
                if (g.node_id == e.node_id) {
                    ++hits;
                    break;
                }
            }
        }
        total += expected.size();
    }
    const double recall = static_cast<double>(hits) / static_cast<double>(total);
    std::cout << "recall@" << k << " = " << recall << "\n";
    CHECK(recall >= 0.9);
}

void test_cosine_metric()
{
    auto index = HNSW::HNSWIndex::create(
        2, 4, 10, HNSW::metric_type2metric_func(HNSW::MetricType::COSINE_DISTANCE)
    );
    index->add_point(HNSW::Tensor { 1.0f, 0.0f }, 10);
    index->add_point(HNSW::Tensor { 0.0f, 1.0f }, 20);
    index->add_point(HNSW::Tensor { 0.9f, 0.1f }, 30);

    auto res = index->query(HNSW::Tensor { 1.0f, 0.0f }, 1, 10, nullptr, nullptr).extract_n_min(1);
    CHECK(res.size() == 1);
    CHECK(res[0].node_id == 0);
}

void test_save_load()
{
    const int dim = 4;
    const int n = 200;

    std::mt19937 rng(7);
    std::vector<HNSW::Tensor> data;
    auto index = HNSW::HNSWIndex::create(
        dim, 8, 100, HNSW::metric_type2metric_func(HNSW::MetricType::EUCLIDEAN_DISTANCE)
    );
    for (int i = 0; i < n; ++i) {
        data.push_back(random_vec(rng, dim));
        index->add_point(data.back(), 1000 + i);
    }

    const auto query = random_vec(rng, dim);
    const auto before = index->query(query, 5, 50, nullptr, nullptr).extract_n_min(5);

    const auto dir = std::filesystem::temp_directory_path() / "hnsw_test_index";
    std::filesystem::remove_all(dir);
    index->save(dir.string());

    auto loaded = HNSW::HNSWIndex::load(
        dir.string(), HNSW::metric_type2metric_func(HNSW::MetricType::EUCLIDEAN_DISTANCE)
    );

    CHECK(loaded->size() == index->size());
    CHECK(loaded->get_params().dimensions == index->get_params().dimensions);
    CHECK(loaded->get_params().M == index->get_params().M);
    CHECK(loaded->get_params().ef_construction == index->get_params().ef_construction);
    CHECK(loaded->get_params().entry_point_id == index->get_params().entry_point_id);
    CHECK(loaded->get_params().layers == index->get_params().layers);

    for (std::size_t i = 0; i < index->size(); ++i) {
        CHECK(loaded->get_node(i).external_id == index->get_node(i).external_id);
        CHECK(loaded->get_node(i).vec == index->get_node(i).vec);
    }

    const auto after = loaded->query(query, 5, 50, nullptr, nullptr).extract_n_min(5);
    CHECK(after.size() == before.size());
    for (std::size_t i = 0; i < before.size(); ++i) {
        CHECK(after[i].node_id == before[i].node_id);
        CHECK(std::fabs(after[i].distance - before[i].distance) < 1e-6f);
    }

    std::filesystem::remove_all(dir);
}

} // namespace

int main()
{
    test_metrics();
    test_default_external_id();
    test_dimension_mismatch();
    test_empty_query();
    test_insert_and_recall();
    test_cosine_metric();
    test_save_load();

    if (failures == 0) {
        std::cout << "All tests passed\n";
        return 0;
    }
    std::cerr << failures << " check(s) failed\n";
    return 1;
}
