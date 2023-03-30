#ifndef SPARSE_STATES_ENTROPY_H
#define SPARSE_STATES_ENTROPY_H
#include <span>
#include <cmath>
#include <numeric>


constexpr double plogp(double p) noexcept {
    if (p > 0.0) {
        return p * std::log2(p);
    }
    return 0.0;
}

constexpr double entropy(const std::span<const double> &v, bool normalize = false) noexcept {
    if (!normalize) {
        return -std::transform_reduce(v.begin(), v.end(), 0.0, std::plus{}, plogp);
    }
    auto sum = std::accumulate(v.begin(), v.end(), 0.0);
    return -std::transform_reduce(v.begin(), v.end(), 0.0, std::plus{},
                                  [sum](auto p) { return plogp(p / sum); });
}

#endif //SPARSE_STATES_ENTROPY_H
