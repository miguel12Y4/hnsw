#pragma once

#include <cmath>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

namespace HNSW {

using Tensor = std::vector<float>;

enum class MetricType : uint8_t {
    EUCLIDEAN_DISTANCE,
    MANHATTAN_DISTANCE,
    COSINE_DISTANCE
};

inline std::string to_string(const MetricType& type)
{
    switch (type) {
    case MetricType::EUCLIDEAN_DISTANCE:
        return "EUCLIDEAN_DISTANCE";
    case MetricType::MANHATTAN_DISTANCE:
        return "MANHATTAN_DISTANCE";
    case MetricType::COSINE_DISTANCE:
        return "COSINE_DISTANCE";
    default:
        return "UNKNOWN_DISTANCE";
    }
}

inline std::ostream& operator<<(std::ostream& os, MetricType type)
{
    return os << to_string(type);
}

inline float euclidean_distance(const Tensor& a, const Tensor& b)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline float manhattan_distance(const Tensor& a, const Tensor& b)
{
    float sum = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        sum += std::fabs(a[i] - b[i]);
    }
    return sum;
}

inline float cosine_distance(const Tensor& a, const Tensor& b)
{
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    return 1.0f - dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9f);
}

using MetricFuncType = float (*)(const Tensor&, const Tensor&);

// Helper function that maps MetricType to metric functions. If an unknown metric name is given, it returns nullptr
inline MetricFuncType metric_type2metric_func(MetricType metric_type)
{
    switch (metric_type) {
    case MetricType::EUCLIDEAN_DISTANCE:
        return euclidean_distance;
    case MetricType::MANHATTAN_DISTANCE:
        return manhattan_distance;
    case MetricType::COSINE_DISTANCE:
        return cosine_distance;
    default:
        return nullptr;
    }
}

} // namespace HNSW
