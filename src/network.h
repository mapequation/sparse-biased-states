#ifndef SPARSE_STATES_NETWORK_H
#define SPARSE_STATES_NETWORK_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>

using NodeId = unsigned int;
using StateId = NodeId;

struct Link {
    NodeId source;
    NodeId target;
    double weight;
};

struct Node {
    NodeId node_id;

    std::vector<Link> in_links{};
    std::vector<Link> out_links{};

    std::unordered_set<NodeId> neighbors{};

    std::unordered_map<NodeId, StateId> states{};

    std::unordered_map<StateId, StateId> lumped_prev_ids{};

    bool expanded = false;
    double jsd_initial{};

    auto add_in_link(const Link &link) noexcept {
        in_links.push_back(link);
        neighbors.insert(link.source);
    }

    auto add_out_link(const Link &link) noexcept {
        out_links.push_back(link);
        neighbors.insert(link.target);
    }

    [[nodiscard]] auto is_neighbor(NodeId other) const noexcept {
        return neighbors.contains(other);
    }

    [[nodiscard]] auto get_state_id(NodeId prev_id) const {
        if (!expanded) {
            return node_id;
        }
        if (lumped_prev_ids.contains(prev_id)) {
            prev_id = lumped_prev_ids.at(prev_id);
        }
        return states.at(prev_id);
    }

    [[nodiscard]] auto in_degree() const noexcept {
        return in_links.size();
    }

    [[nodiscard]] auto out_degree() const noexcept {
        return out_links.size();
    }
};

std::tuple<std::unordered_map<NodeId, Node>, double>
read_links(std::ifstream &in, bool weighted = false, bool directed = false) {
    NodeId source_id, target_id;

    std::unordered_map<NodeId, Node> nodes;
    double weight = 1.0;
    auto tot_weight = 0.0;

    while (in >> source_id >> target_id) {
        if (weighted) {
            in >> weight;
        }

        tot_weight += weight;
        Link link{source_id, target_id, weight};

        if (!nodes.contains(source_id)) {
            nodes.emplace(source_id, Node{source_id});
        }
        if (!nodes.contains(target_id)) {
            nodes.emplace(target_id, Node{target_id});
        }

        auto &source = nodes.at(source_id);
        auto &target = nodes.at(target_id);

        source.add_out_link(link);
        target.add_in_link(link);

        if (!directed) {
            tot_weight += weight;
            Link rev{target_id, source_id, weight};
            target.add_out_link(rev);
            source.add_in_link(rev);
        }
    }

    return {nodes, tot_weight};
}

#endif //SPARSE_STATES_NETWORK_H
