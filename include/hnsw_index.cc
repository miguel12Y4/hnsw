#include "hnsw_index.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "hnsw_heap.h"
#include "hnsw_visited_set.h"

namespace fs = std::filesystem;

namespace HNSW {
std::unique_ptr<HNSWIndex> HNSWIndex::create(
    uint64_t dimensions,
    uint64_t max_neighbors,
    uint64_t n_candidates_insertion,
    MetricFuncType metric_func
)
{
    HNSWIndexParams params {};
    params.entry_point_id = 0;
    params.dimensions = dimensions;
    params.layers = 1;
    params.M = max_neighbors;
    params.ef_construction = n_candidates_insertion;

    return std::make_unique<HNSWIndex>(params, metric_func);
}

std::unique_ptr<HNSWIndex> HNSWIndex::load(const std::string& path, MetricFuncType metric_func)
{
    const auto absolute_data_file_path = fs::path(path) / HNSWIndex::DATA_FILENAME;

    std::fstream ifs(absolute_data_file_path, std::ios::in | std::ios::binary);

    // load params
    HNSWIndex::HNSWIndexParams params;
    ifs.read(reinterpret_cast<char*>(&params), sizeof(HNSWIndex::HNSWIndexParams));
    if (!ifs.good()) {
        throw std::runtime_error("Could not read index file");
    }

    uint64_t num_nodes;
    ifs.read(reinterpret_cast<char*>(&num_nodes), sizeof(uint64_t));
    if (!ifs.good()) {
        throw std::runtime_error("Could not read index file");
    }

    std::vector<HNSWNode> node_storage;
    std::vector<std::vector<HNSWEntryFlatMap>> node_neighbors_at_layer;

    node_storage.reserve(num_nodes);
    node_neighbors_at_layer.reserve(num_nodes);

    for (uint32_t node_id = 0; node_id < num_nodes; ++node_id) {
        // read node
        uint64_t external_id;
        ifs.read(reinterpret_cast<char*>(&external_id), sizeof(uint64_t));
        if (!ifs.good()) {
            throw std::runtime_error("Could not read index file");
        }

        Tensor node_vec(params.dimensions);
        ifs.read(reinterpret_cast<char*>(node_vec.data()), params.dimensions * sizeof(float));
        if (!ifs.good()) {
            throw std::runtime_error("Could not read index file");
        }
        node_storage.emplace_back(external_id, std::move(node_vec));

        // read neighbors
        uint64_t num_layers;
        ifs.read(reinterpret_cast<char*>(&num_layers), sizeof(uint64_t));
        if (!ifs.good()) {
            throw std::runtime_error("Could not read index file");
        }

        node_neighbors_at_layer.emplace_back(std::vector<HNSWEntryFlatMap>(num_layers));

        for (uint64_t layer_i = 0; layer_i < num_layers; ++layer_i) {
            uint64_t num_neighbors_at_layer_i;
            ifs.read(reinterpret_cast<char*>(&num_neighbors_at_layer_i), sizeof(uint64_t));
            if (!ifs.good()) {
                throw std::runtime_error("Could not read index file");
            }

            Entry entry_buf;
            for (uint64_t neighbor_i = 0; neighbor_i < num_neighbors_at_layer_i; ++neighbor_i) {
                ifs.read(reinterpret_cast<char*>(&entry_buf), sizeof(entry_buf));
                if (!ifs.good()) {
                    throw std::runtime_error("Could not read index file");
                }
                node_neighbors_at_layer[node_id][layer_i].emplace(entry_buf);
            }
        }
    }

    return std::make_unique<HNSWIndex>(
        params,
        metric_func,
        std::move(node_storage),
        std::move(node_neighbors_at_layer)
    );
}

void HNSWIndex::save(const std::string& path) const
{
    fs::create_directories(path);
    const auto absolute_data_file_path = fs::path(path) / HNSWIndex::DATA_FILENAME;

    std::fstream ofs(absolute_data_file_path, std::ios::out | std::ios::trunc | std::ios::binary);
    // write params
    ofs.write(reinterpret_cast<const char*>(&params), sizeof(HNSWIndexParams));
    if (!ofs.good()) {
        throw std::runtime_error("Could not write index file");
    }

    const uint64_t num_nodes = node_storage.size();
    ofs.write(reinterpret_cast<const char*>(&num_nodes), sizeof(uint64_t));
    if (!ofs.good()) {
        throw std::runtime_error("Could not write index file");
    }

    for (uint32_t node_id = 0; node_id < num_nodes; ++node_id) {
        // write node
        const HNSWNode& node = node_storage[node_id];
        ofs.write(reinterpret_cast<const char*>(&node.external_id), sizeof(uint64_t));
        ofs.write(reinterpret_cast<const char*>(node.vec.data()), params.dimensions * sizeof(float));
        if (!ofs.good()) {
            throw std::runtime_error("Could not write index file");
        }

        // write neighbors
        const auto& node_neighbors_at_layers = node_neighbors_at_layer[node_id];
        const uint64_t num_layers = node_neighbors_at_layers.size();
        ofs.write(reinterpret_cast<const char*>(&num_layers), sizeof(uint64_t));
        if (!ofs.good()) {
            throw std::runtime_error("Could not write index file");
        }

        for (const auto& neighbors_at_layer_i : node_neighbors_at_layers) {
            // layer i size
            const uint64_t num_neighbors_at_layer_i = neighbors_at_layer_i.size();
            ofs.write(reinterpret_cast<const char*>(&num_neighbors_at_layer_i), sizeof(uint64_t));
            if (!ofs.good()) {
                throw std::runtime_error("Could not write index file");
            }

            // layer i data
            for (const auto& neighbor_entry : neighbors_at_layer_i) {
                ofs.write(reinterpret_cast<const char*>(&neighbor_entry), sizeof(neighbor_entry));
                if (!ofs.good()) {
                    throw std::runtime_error("Could not write index file");
                }
            }
        }
    }
}

HNSWHeap HNSWIndex::query(
    const Tensor& query_tensor,
    uint64_t num_neighbors,
    uint64_t num_candidates,
    HNSWVisitedSet* visited_set,
    HNSWHeap* discarded_heap_ptr
)
{
    if (node_storage.empty()) {
        return HNSWHeap(num_neighbors);
    }

    const uint64_t top_layer = params.layers - 1;

    // Initialize with entry point
    const auto& entry_tensor = node_storage[params.entry_point_id].vec;
    const float entry_dist = metric_func(query_tensor, entry_tensor);
    std::vector<Entry> entry_points;
    entry_points.emplace_back(entry_dist, params.entry_point_id);

    // find the best set entry points for layer 0
    for (uint64_t current_layer = top_layer; current_layer > 0; --current_layer) {
        const auto current_layer_nns = search_at_layer(
            query_tensor,
            entry_points,
            1,
            current_layer,
            nullptr,
            nullptr
        );
        assert(current_layer_nns.size() == 1);
        entry_points[0] = current_layer_nns.get_min();
    }

    auto nearest_neighbors = search_at_layer(
        query_tensor,
        entry_points,
        num_candidates,
        0,
        visited_set,
        discarded_heap_ptr
    );

    if (discarded_heap_ptr) {
        // modify inplace and store discarded
        while (nearest_neighbors.size() > num_neighbors) {
            discarded_heap_ptr->push(nearest_neighbors.get_max());
            nearest_neighbors.pop_max();
        }
        return nearest_neighbors;
    }

    const bool is_inplace_cheaper = nearest_neighbors.size() < 2 * num_neighbors;
    if (is_inplace_cheaper) {
        // modify inplace
        while (nearest_neighbors.size() > num_neighbors) {
            nearest_neighbors.pop_max();
        }
        return nearest_neighbors;
    }

    // create a new heap
    HNSWHeap res(num_neighbors);
    while (res.size() < num_neighbors) {
        res.push(nearest_neighbors.get_min());
        nearest_neighbors.pop_min();
    }
    return res;
}

uint32_t HNSWIndex::add_point(const Tensor& query_tensor, uint64_t external_id)
{
    if (query_tensor.size() != params.dimensions) {
        // Tensor dimension does not match
        throw std::runtime_error("Tensor dimension does not match index dimensions");
    }

    // Create new node at a random layer
    const uint64_t node_top_layer = get_random_layer();

    if (node_storage.empty()) {
        return create_new_node(external_id, query_tensor, 0);
    }

    const uint64_t node_id = create_new_node(external_id, query_tensor, node_top_layer);

    // Initialize with entry point
    const auto& entry_tensor = node_storage[params.entry_point_id].vec;
    const auto entry_dist = metric_func(query_tensor, entry_tensor);
    std::vector<Entry> entry_points;
    entry_points.emplace_back(entry_dist, params.entry_point_id);

    // Update the best entry point for layers in range [top_layer, node_top_layer)
    const uint64_t top_layer = params.layers - 1;
    for (uint64_t current_layer = top_layer; current_layer > node_top_layer; --current_layer) {
        const auto current_layer_nns = search_at_layer(
            query_tensor,
            entry_points,
            1,
            current_layer,
            nullptr,
            nullptr
        );
        assert(current_layer_nns.size() == 1);
        entry_points[0] = current_layer_nns.get_min();
    }

    // Insert in node's layer and the layers below
    for (int64_t current_layer = std::min(node_top_layer, top_layer); current_layer >= 0; --current_layer) {
        auto current_layer_nns = search_at_layer(
            query_tensor,
            entry_points,
            params.ef_construction,
            current_layer,
            nullptr,
            nullptr
        );
        const uint64_t max_neighbors = current_layer == 0 ? M0 : Mi;

        auto current_layer_top_k = current_layer_nns.extract_n_min(max_neighbors);

        // Mutually connect node with its new neighbors
        set_neighbors_at_layer(node_id, current_layer, current_layer_top_k);

        // Update entry points
        entry_points = std::move(current_layer_top_k);
    }

    if (node_top_layer > top_layer) {
        params.entry_point_id = node_id;
        params.layers = node_top_layer + 1;
    };

    return node_id;
}

HNSWIndex::HNSWIndex(HNSWIndexParams params_, MetricFuncType metric_func_) :
    metric_func { metric_func_ },
    params { params_ }
{
    init_constants();
}

HNSWIndex::HNSWIndex(
    HNSWIndexParams params_,
    MetricFuncType metric_func_,
    std::vector<HNSWNode>&& node_storage_,
    std::vector<std::vector<HNSWEntryFlatMap>>&& node_neighbors_at_layer_
) :
    metric_func { metric_func_ },
    params { params_ },
    node_storage { std::move(node_storage_) },
    node_neighbors_at_layer { std::move(node_neighbors_at_layer_) }
{
    init_constants();

    for (std::size_t i = 0; i < node_storage.size(); ++i) {
        hnsw_node2node_id.emplace(node_storage[i], i);
    }
}

HNSWHeap HNSWIndex::search_at_layer(
    const Tensor& query_tensor,
    const std::vector<Entry>& entry_points,
    uint64_t num_neighbors,
    uint64_t layer,
    HNSWVisitedSet* visited_set_ptr,
    HNSWHeap* discarded_heap_ptr
)
{
    assert(!entry_points.empty());
    assert(num_neighbors > 0);
    assert(entry_points.size() <= num_neighbors);

    HNSWVisitedSet visited_set_;
    HNSWHeap candidates_heap(num_neighbors);
    HNSWHeap top_k_heap(num_neighbors);

    if (visited_set_ptr != nullptr) {
        for (std::size_t i = 0; i < entry_points.size(); ++i) {
            candidates_heap.emplace(entry_points[i]);
            top_k_heap.emplace(entry_points[i]);
            visited_set_ptr->emplace(entry_points[i].node_id);
        }
    } else {
        visited_set_.reserve(node_storage.size());
        // no visited set was provided, initialize a new one
        for (std::size_t i = 0; i < entry_points.size(); ++i) {
            visited_set_.emplace(entry_points[i].node_id);
            candidates_heap.emplace(entry_points[i]);
            top_k_heap.emplace(entry_points[i]);
        }

        visited_set_ptr = &visited_set_;
    }

    while (!candidates_heap.empty()) {
        // get best candidate and worst result
        const Entry candidate_entry = candidates_heap.get_min();
        if (candidate_entry.distance > top_k_heap.get_max().distance) {
            // no further result improvement
            break;
        }

        candidates_heap.pop_min();

        // explore candidate's neighborhood
        for (const Entry& candidate_neighbor_entry : node_neighbors_at_layer[candidate_entry.node_id][layer])
        {
            if (visited_set_ptr->find(candidate_neighbor_entry.node_id) != visited_set_ptr->end()) {
                continue;
            }
            visited_set_ptr->emplace(candidate_neighbor_entry.node_id);

            const auto& candidate_neighbor_tensor = node_storage[candidate_neighbor_entry.node_id].vec;
            const float candidate_neighbor_dist = metric_func(query_tensor, candidate_neighbor_tensor);

            if (top_k_heap.size() < num_neighbors) {
                // always add, not enough results
                candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                top_k_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                continue;
            }

            if (candidate_neighbor_dist < top_k_heap.get_max().distance) {
                // the candidate improves the result, furthest result must be removed
                if (discarded_heap_ptr != nullptr) {
                    discarded_heap_ptr->emplace(top_k_heap.get_max());
                }
                candidates_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                top_k_heap.pop_max();
                top_k_heap.emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
                continue;
            }

            if (discarded_heap_ptr != nullptr) {
                // the candidate does not improve the result
                discarded_heap_ptr->emplace(candidate_neighbor_dist, candidate_neighbor_entry.node_id);
            }
        }
    }

    return top_k_heap;
}

void HNSWIndex::set_neighbors_at_layer(
    uint32_t node_id,
    uint64_t layer,
    const std::vector<Entry>& new_neighbors
)
{
    const auto max_neighbors = layer == 0 ? M0 : Mi;
    assert(new_neighbors.size() <= max_neighbors);

    for (const Entry& new_neighbor_entry : new_neighbors) {
        // node -> neighbor
        node_neighbors_at_layer[node_id][layer].emplace(new_neighbor_entry);

        // neighbor -> node
        auto& new_neighbor_neighbors_at_layer = node_neighbors_at_layer[new_neighbor_entry.node_id][layer];
        if (new_neighbor_neighbors_at_layer.size() < max_neighbors) {
            // max_neighbors not reached, just add the connection
            new_neighbor_neighbors_at_layer.emplace(new_neighbor_entry.distance, node_id);
        } else {
            // replace if the connection is better than the furthest one
            const auto furthest_it = --new_neighbor_neighbors_at_layer.end();
            if (new_neighbor_entry.distance < furthest_it->distance) {
                new_neighbor_neighbors_at_layer.erase(furthest_it);
                new_neighbor_neighbors_at_layer.emplace(new_neighbor_entry.distance, node_id);
            }
        }
    }
}

} // namespace HNSW
