#ifndef SPARSE_STATES_SPARSE_STATES_H
#define SPARSE_STATES_SPARSE_STATES_H

#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "network.h"
#include "clustering.h"

inline auto bias(const Node &prev, const Node &target, double p_inv, double q_inv) {
    if (target.node_id == prev.node_id) {
        return p_inv;
    } else if (!prev.is_neighbor(target.node_id)) {
        return q_inv;
    }
    return 1.0;
}

constexpr StateId create_state_id(NodeId prev_id, NodeId current_id) {
    return prev_id << 16 | current_id;
}

inline auto create_states(Node &node, const auto &nodes, double p_inv, double q_inv, double thresh, bool weighted) noexcept {
    // if only one in-link, the JSD is 0
    if (node.in_degree() <= 1) {
        node.expanded = false;
        node.states = {{node.node_id, node.node_id}};
        return;
    }

    std::vector<std::vector<double>> X{node.in_degree(), std::vector<double>(node.out_degree())};
    std::vector<double> weights(node.in_degree(), 1.0 / static_cast<double>(node.in_degree()));
    std::vector<NodeId> prev_ids(node.in_degree());
    auto sum_in_weight = 0.0;

    for (auto i = 0; const auto &in_link: node.in_links) {
        auto &prev = nodes.at(in_link.source);
        prev_ids[i] = in_link.source;
        sum_in_weight += in_link.weight;

        for (auto j = 0; const auto &out_link: node.out_links) {
            auto &target = nodes.at(out_link.target);
            X[i][j] = out_link.weight * bias(prev, target, p_inv, q_inv);
            j++;
        }
        i++;
    }

    if (weighted) {
        auto &in_links = node.in_links;
        std::transform(in_links.cbegin(), in_links.cend(), weights.begin(),
                       [sum_in_weight](auto &link) { return link.weight / sum_in_weight; });
    }

    DivisiveClustering clustering{X, weights};
    auto jsd = clustering.get_jsd();

    node.jsd_initial = jsd;
    node.expanded = jsd > thresh;

    if (!node.expanded) {
        node.states = {{node.node_id, node.node_id}};
        return;
    }

    auto labels = clustering.divide(thresh);

    std::vector<int> unique_labels{labels};
    std::sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());

    std::unordered_map<int, std::vector<int>> lumped_prev_ids;
    for (auto i = 0; auto prev_id: prev_ids) {
        auto label = labels[i];
        lumped_prev_ids[label].push_back(prev_id);
        ++i;
    }

    for (const auto &[_, lumped_ids]: lumped_prev_ids) {
        // choose arbitrary id that represents all lumped ids in cluster
        //auto lumped_prev_id = lumped_ids[0];
        // or choose smallest?
        auto lumped_prev_id = *std::min_element(lumped_ids.begin(), lumped_ids.end());
        auto state_id = create_state_id(lumped_prev_id, node.node_id);
        node.states[lumped_prev_id] = state_id;

        for (auto prev_id: lumped_ids) {
            node.lumped_prev_ids[prev_id] = lumped_prev_id;
        }
    }
}

std::tuple<unsigned int, unsigned int>
expand_nodes(auto &nodes, double p, double q, double thresh, bool weighted) {
    auto num_expanded = 0;
    auto num_states = 0;
    const auto p_inv = 1.0 / p;
    const auto q_inv = 1.0 / q;

#pragma omp parallel for reduction(+:num_expanded, num_states) default(none) shared(nodes, p_inv, q_inv, thresh, weighted) schedule(dynamic)
    //for (auto &[node_id, node]: nodes) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto it = nodes.begin();
        std::advance(it, i);
        auto &[node_id, node] = *it;

        create_states(node, nodes, p_inv, q_inv, thresh, weighted);

        if (node.expanded) {
#pragma omp critical
            ++num_expanded;
            num_states += node.states.size();
        } else {
#pragma omp critical
            ++num_states;
        }
    }

    return {num_expanded, num_states};
}

std::tuple<std::vector<Link>, double>
create_links_for_node(const auto &node, const auto &nodes, double p_inv, double q_inv, bool self_links) noexcept {
    std::vector<Link> links;
    std::unordered_map<NodeId, std::unordered_map<NodeId, double>> aggregated_links;

    // TODO for directed and in degree = 0, loop only out links

    /*
    if (method == Bias::Unbiased) {
        double tot_weight = 0.0;

        if (!node.expanded) {
            auto source_id = node.node_id;
            for (const auto &out_link: node.out_links) {
                if (!self_links && out_link.target == node.node_id) {
                    continue;
                }

                const auto &target = nodes.at(out_link.target);
                auto weight = out_link.weight;
                tot_weight += weight;
                auto target_id = target.get_state_id(node.node_id);
                links.push_back({source_id, target_id, weight});
            }
        } else {
            for (const auto &in_link: node.in_links) {
                //const auto &prev = nodes.at(in_link.source);
                auto source_id = node.get_state_id(in_link.source);

                for (const auto &out_link: node.out_links) {
                    if (!self_links && out_link.target == node.node_id) {
                        continue;
                    }

                    const auto &target = nodes.at(out_link.target);
                    auto weight = out_link.weight / node.in_degree();
                    tot_weight += weight;
                    auto target_id = target.get_state_id(node.node_id);
                    aggregated_links[source_id][target_id] += weight;
                }
            }
            for (auto &[source_id, targets] : aggregated_links) {
                for (auto [target_id, weight] : targets) {
                    links.push_back({source_id, target_id, weight});
                }
            }
        }
        return {links, tot_weight};
    }
    */
    auto sum_weight = 0.0;
    auto sum_biased_weight = 0.0;

    //auto in_degree = node.in_degree();
    //auto log_norm = std::log2(in_degree > 1 ? in_degree : 2);
    //auto strength = bias_to_strength(method, node.jsd_initial / log_norm);

    for (const auto &in_link: node.in_links) {
        const auto &prev = nodes.at(in_link.source);
        auto source_id = node.get_state_id(in_link.source);

        for (const auto &out_link: node.out_links) {
            if (!self_links && out_link.target == node.node_id) {
                continue;
            }

            const auto &target = nodes.at(out_link.target);
            auto weight = out_link.weight * bias(prev, target, p_inv, q_inv);

            sum_weight += out_link.weight;
            sum_biased_weight += weight;

            auto target_id = target.get_state_id(node.node_id);
            aggregated_links[source_id][target_id] += weight;
        }
    }

    sum_weight /= node.in_degree();
    auto normalizing_factor = sum_weight / sum_biased_weight;

    for (auto &[source_id, targets] : aggregated_links) {
        for (auto [target_id, weight] : targets) {
            auto w = weight * normalizing_factor;
            links.push_back({source_id, target_id, w});
        }
    }

    return {links, sum_biased_weight * normalizing_factor};
}

std::tuple<std::vector<Link>, double>
create_links(const auto &nodes, double p, double q, bool self_links = false) {
    std::vector<Link> links;
    const auto p_inv = 1.0 / p;
    const auto q_inv = 1.0 / q;
    auto tot_weight = 0.0;

#pragma omp parallel for reduction(+:tot_weight) default(none) shared(nodes, links, p_inv, q_inv, self_links) schedule(dynamic)
    //for (const auto &[node_id, node] : nodes) {
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        auto it = nodes.begin();
        std::advance(it, i);
        auto &[_, node] = *it;

        auto [private_links, private_weight] = create_links_for_node(node, nodes, p_inv, q_inv, self_links);
        tot_weight += private_weight;

#pragma omp critical
        links.insert(links.end(), private_links.begin(), private_links.end());
    }

    return {links, tot_weight};
}

auto write(std::ofstream &out, const auto &nodes, const auto &links) {
    out << "*States\n";

    for (const auto &[node_id, node]: nodes) {
        for (const auto &[_, state_id]: node.states) {
            out << state_id << ' ' << node.node_id << '\n';
        }
    }

    out << "*Links\n";

    for (const auto &link: links) {
        out << link.source << ' ' << link.target << ' ' << link.weight << '\n';
    }
}

#endif //SPARSE_STATES_SPARSE_STATES_H
