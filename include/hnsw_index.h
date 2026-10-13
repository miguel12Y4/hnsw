#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hnsw_entry.h"
#include "hnsw_entry_flat_map.h"
#include "hnsw_heap.h"
#include "hnsw_metric.h"
#include "hnsw_node.h"
#include "hnsw_visited_set.h"

namespace HNSW {
class HNSWIndex {
public:
    struct HNSWIndexParams {
        uint64_t entry_point_id;  // entry id
        uint64_t dimensions;      // tensor dimensions
        uint64_t layers;          // current number of layers
        uint64_t M;               // max neighbors at construction
        uint64_t ef_construction; // max candidates at construction
    };

    const HNSWIndexParams& get_params() const
    {
        return params;
    };

    static constexpr char DATA_FILENAME[] = "hnsw.dat";

    // create a new index in memory
    // create a new index in memory
    static std::unique_ptr<HNSWIndex> create(
        uint64_t dimensions,
        uint64_t max_neighbors,
        uint64_t ef_construction,
        MetricFuncType metric_func
    );

    // load an index from disk
    static std::unique_ptr<HNSWIndex> load(const std::string& path, MetricFuncType metric_func);

    // save an index to disk
    void save(const std::string& path) const;

    // Create constructor
    HNSWIndex(HNSWIndexParams params, MetricFuncType metric_func);

    // Load constructor
    HNSWIndex(
        HNSWIndexParams params,
        MetricFuncType metric_func,
        std::vector<HNSWNode>&& node_storage,
        std::vector<std::vector<HNSWEntryFlatMap>>&& node_neighbors_at_layer
    );

    // insert a point in the index
    uint32_t add_point(const Tensor& vec, uint64_t external_id = UINT64_MAX);

    // find the the top num_neighbors from a pool of num_candidates
    HNSWHeap query(
        const Tensor& query_tensor,
        uint64_t num_neighbors,
        uint64_t num_candidates,
        HNSWVisitedSet* visited_set,
        HNSWHeap* discarded_heap_ptr
    );

    inline const HNSWNode& get_node(uint32_t node_id) const
    {
        assert(node_id < node_storage.size());
        return node_storage[node_id];
    }

    inline std::size_t size() const
    {
        return node_storage.size();
    }

private:
    MetricFuncType metric_func;

    HNSWIndexParams params;

    // useful constants
    uint64_t M0;                // max neighbors per node at layer 0
    uint64_t Mi;                // max neighbors per node at layer i > 0
    double random_layer_factor; // normalization factor for random layer selection

    // used for generating levels
    std::random_device device;
    std::default_random_engine level_generator { device() };
    std::uniform_real_distribution<double> distribution { 0.0l, 1.0l };

    // main node storage
    std::vector<HNSWNode> node_storage;

    // node_neighbors_at_layer[i][j] is the map of neighbors for the node i at layer j
    // NOTE: This could be replaced with an array of node_ids, but the set_neighbors_at_layer we
    // would need to calculate the current-worst distance for each node in new_neighbors parameter.
    // This trade-off could be evaluated in the future.
    std::vector<std::vector<HNSWEntryFlatMap>> node_neighbors_at_layer;

    // inverse of node_storage for external_id lookup
    std::unordered_map<HNSWNode, uint32_t, HNSWNode::Hasher> hnsw_node2node_id;

    // initialize constants for the index
    inline void init_constants()
    {
        M0 = 2 * params.M;
        Mi = params.M;
        random_layer_factor = 1.0l / std::log(static_cast<double>(params.M));
    }

    inline uint64_t create_new_node(uint64_t external_id, const Tensor& vec, uint64_t top_layer)
    {
        const uint32_t node_id = node_storage.size();
        // create node
        node_storage.emplace_back(external_id == UINT64_MAX ? node_id : external_id, vec);

        // create node's neighbors
        auto neighbors = std::vector<HNSWEntryFlatMap>(top_layer + 1);
        node_neighbors_at_layer.emplace_back(std::move(neighbors));

        // add to inverse lookup
        hnsw_node2node_id.emplace(node_storage.back(), node_id);

        return node_id;
    }

    // search the top num_neighbors at a layer given a query and a group of entry points
    HNSWHeap search_at_layer(
        const Tensor& query_tensor,
        const std::vector<Entry>& entry_points,
        uint64_t num_neighbors,
        uint64_t layer,
        /* Arguments for resuming the iterative querying */
        // previously visited node_ids (iterative querying)
        HNSWVisitedSet* visited_set_ptr,
        // storage for discarded candidates (iterative querying)
        HNSWHeap* discarded_heap_ptr
    );

    // connect the node to the new neighbors (both ways) keeping the max edges limit
    void set_neighbors_at_layer(uint32_t node_id, uint64_t layer, const std::vector<Entry>& new_neighbors);

    // generate a random layer number
    inline uint64_t get_random_layer()
    {
        const double random_layer = -std::log(distribution(level_generator)) * random_layer_factor;
        return static_cast<uint64_t>(random_layer);
    }
};
} // namespace HNSW
