#ifndef SPARSE_STATES_CLUSTERING_H
#define SPARSE_STATES_CLUSTERING_H

#include <numeric>
#include <queue>
#include <random>
#include <functional>
#include <span>

#include "entropy.h"

inline double js_distance(const std::span<const double> &p, const std::span<const double> &q) noexcept {
    std::vector<double> mix(p.size(), 0);

    double sum_p = 0.0;
    double sum_q = 0.0;
    double sum_mix = 0.0;

    auto equal = true;

    for (std::size_t i = 0; i < p.size(); ++i) {
        auto pi = p[i];
        auto qi = q[i];
        equal &= pi == qi;
        sum_p += pi;
        sum_q += qi;
        mix[i] += 0.5 * (pi + qi);
        sum_mix += mix[i];
    }

    if (equal) {
        return 0.0;
    }

    double kl_p = 0.0;
    double kl_q = 0.0;

    for (std::size_t i = 0; i < p.size(); ++i) {
        auto pi = p[i] / sum_p;
        auto qi = q[i] / sum_q;
        auto mi = mix[i] / sum_mix;

        if (pi > 0) {
            kl_p -= pi * std::log2(mi / pi);
        }
        if (qi > 0) {
            kl_q -= qi * std::log2(mi / qi);
        }
    }

    return std::sqrt(0.5 * (kl_p + kl_q));
}

inline double js_divergence(const std::vector<std::vector<double>> &X,
                            const std::span<const double> &weights,
                            const std::span<const int> &indices) noexcept {
    auto num_rows = X.size();

    if (num_rows <= 1) {
        return 0.0;
    }

    auto num_features = X[0].size();

    double jsd = 0.0;
    double sum_w = 0.0;
    std::vector<double> mix(num_features, 0.0);

#pragma clang loop vectorize(assume_safety)
    for (auto i: indices) {
        const auto &x = X[i];
        auto weight = weights[i];
        sum_w += weight;

        for (auto j = 0; auto xj: x) {
            mix[j] += xj * weight;
            ++j;
        }

        jsd -= weight * entropy(x, true);
    }

    jsd += sum_w * entropy(mix, true);

    return jsd < 0 ? 0 : jsd;
}


std::mt19937 gen(std::random_device{}());
std::uniform_int_distribution<> rand_bit(0, 1);

template <typename T>
inline decltype(auto) get_random_element(const std::vector<T> &v) {
    std::uniform_int_distribution<int> dist(0, v.size() - 1);
    return v[dist(gen)];
}

struct Cluster {
    double jsd;
    std::vector<int> indices;

    friend bool operator<(const Cluster &lhs, const Cluster &rhs) {
        return lhs.jsd < rhs.jsd;
    }
};

class DivisiveClustering {
private:
    std::vector<std::vector<double>> X;
    std::vector<double> weights;
    std::vector<int> labels;
    double jsd;

    std::priority_queue<Cluster> queue;

public:
    DivisiveClustering(auto X, auto weights)
            : X(X), weights(weights), labels(X.size(), 0) {
        std::vector<int> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);
        jsd = js_divergence(X, weights, indices);
        queue.emplace(Cluster{jsd, indices});
    }

    [[nodiscard]] auto divide(double threshold) {
        if (threshold <= 0) {
            return labels;
        }

        while (jsd > threshold) {
            divide_largest();
        }

        return labels;
    }

    [[nodiscard]] auto get_jsd() const noexcept {
        return jsd;
    }

private:
    [[nodiscard]] auto get_distances(auto index_from) const noexcept {
        auto &indices = queue.top().indices;
        std::vector<double> distances(indices.size());
        std::transform(indices.cbegin(), indices.cend(), distances.begin(),
                       [&, this](auto index_to) { return js_distance(X[index_to], X[index_from]); });
        return distances;
    }

    [[nodiscard]] auto get_proportional_index(const std::span<const double> &distances) const noexcept {
        auto &indices = queue.top().indices;
        auto index = std::distance(distances.begin(),
                                   std::max_element(distances.begin(), distances.end()));
        return indices[index];
    }

    auto divide_largest() noexcept -> void {
        const auto &cluster = queue.top();
        auto cluster_jsd = cluster.jsd;
        auto &cluster_indices = cluster.indices;

        if (cluster_indices.size() <= 1) {
            // should not happen!
            return;
        } else if (cluster_indices.size() == 2) {
            // trivial case when only two indices
            queue.pop();
            queue.push({0.0, {cluster_indices[0]}});
            queue.push({0.0, {cluster_indices[1]}});
            jsd -= cluster_jsd; // ok?
            return;
        }

        std::vector<int> best_first_indices;
        std::vector<int> best_second_indices;
        double best_first_jsd = 0.0;
        double best_second_jsd = 0.0;

        const int attempts = 5;
        for (int attempt = 0; attempt < attempts; ++attempt) {
            auto rand_index = get_random_element(cluster_indices);
            auto rand_distances = get_distances(rand_index);

            auto first_index = get_proportional_index(rand_distances);
            auto first_distances = get_distances(first_index);

            auto second_index = get_proportional_index(first_distances);
            auto second_distances = get_distances(second_index);

            std::vector<int> first_indices;
            std::vector<int> second_indices;

            const double min_distance = 1e-10;

            for (int i = 0; auto index: cluster_indices) {
                auto dist_to_first = first_distances[i];
                auto dist_to_second = second_distances[i];
                auto delta_dist = std::abs(dist_to_first - dist_to_second);

                if (delta_dist < min_distance) {
                    // put in random cluster
                    //auto &rand_cluster = rand_bit(gen) ? second_indices : first_indices;
                    //rand_cluster.push_back(index);
                    // or, put in smallest?
                    if (second_indices.size() < first_indices.size()) {
                        second_indices.push_back(index);
                    } else if (second_indices.size() == first_indices.size()) {
                        auto &rand_cluster = rand_bit(gen) ? second_indices : first_indices;
                        rand_cluster.push_back(index);
                    } else {
                        first_indices.push_back(index);
                    }
                } else if (dist_to_second < dist_to_first) {
                    second_indices.push_back(index);
                } else {
                    first_indices.push_back(index);
                }
                ++i;
            }

            // try to balance clusters if needed
            if (first_indices.empty() || second_indices.empty()) {
                if (first_indices.size() < second_indices.size()) {
                    auto last = second_indices.back();
                    second_indices.pop_back();
                    first_indices.push_back(last);
                } else if (second_indices.size() < first_indices.size()) {
                    auto last = first_indices.back();
                    first_indices.pop_back();
                    second_indices.push_back(last);
                }
                continue;
            }

            auto first_jsd = js_divergence(X, weights, first_indices);
            auto second_jsd = js_divergence(X, weights, second_indices);

            auto attempt_jsd = first_jsd + second_jsd;
            auto best_attempt_jsd = best_first_jsd + best_second_jsd;

            if (attempt == 0 || attempt_jsd < best_attempt_jsd) {
                best_first_indices = first_indices;
                best_second_indices = second_indices;
                best_first_jsd = first_jsd;
                best_second_jsd = second_jsd;
            }
        }

        if (best_first_indices.empty() || best_second_indices.empty()) {
            return;
        }

        auto new_label = static_cast<int>(queue.size());
        for (auto index: best_second_indices) {
            labels[index] = new_label;
        }

        queue.pop();
        queue.push({best_first_jsd, best_first_indices});
        queue.push({best_second_jsd, best_second_indices});

        jsd -= cluster_jsd;
        jsd += best_first_jsd + best_second_jsd;
    }
};

#endif //SPARSE_STATES_CLUSTERING_H
