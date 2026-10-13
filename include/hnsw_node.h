#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace HNSW {
struct HNSWNode {
    uint64_t external_id;   // Object id itself
    std::vector<float> vec; // Tensor associated to the object

    HNSWNode() = default;

    HNSWNode(uint64_t external_id_, std::vector<float> vec_) :
        external_id { external_id_ },
        vec { std::move(vec_) }
    { }

    inline bool operator==(const HNSWNode& other) const
    {
        return external_id == other.external_id;
    }

    struct Hasher {
        size_t operator()(const HNSW::HNSWNode& node) const
        {
            return std::hash<uint64_t> {}(node.external_id);
        }
    };
};

} // namespace HNSW
