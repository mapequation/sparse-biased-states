#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <fmt/format.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cxxopts.hpp"
#include "network.h"
#include "sparse-states.h"

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using duration = std::chrono::duration<double>;

    Timer() : start(clock::now()) {}

    [[nodiscard]] duration elapsed() const {
        return clock::now() - start;
    }

    void reset() {
        start = clock::now();
    }

    time_point start;
};

auto run(std::ifstream &in, std::ofstream &out, double p, double q, double thresh, bool weighted, bool directed) {
    Timer total;
    Timer timer;

    // 0. Read links
    fmt::print(stdout, "Reading links... ");
    std::cout << std::flush;
    auto [nodes, tot_weight] = read_links(in, weighted, directed);
    fmt::print("{} nodes, total link weight: {} ({} s)\n", nodes.size(), tot_weight, timer.elapsed().count());

    in.close();
    timer.reset();

    // 1. Expand nodes
    fmt::print(stdout,"Expanding nodes... ");
    std::cout << std::flush;
    auto [num_expanded, num_states] = expand_nodes(nodes, p, q, thresh, weighted);

    auto tot_jsd = std::transform_reduce(nodes.cbegin(), nodes.cend(), 0.0, std::plus{},
                                         [](auto &it) { return it.second.jsd_initial; });
    auto avg_jsd = tot_jsd / static_cast<double>(nodes.size());
    fmt::print("{} expanded, avg. JSD: {} ({} s)\n", num_expanded, avg_jsd, timer.elapsed().count());

    timer.reset();

    // 2. Create links
    fmt::print(stdout, "Creating links... ");
    std::cout << std::flush;
    auto [links, tot_created_weight] = create_links(nodes, p, q);
    fmt::print("{} weighted, directed links, total weight: {} ({} s)\n", links.size(), tot_created_weight, timer.elapsed().count());

    timer.reset();

    // 3. Write network
    fmt::print(stdout,"Writing state network... ");
    std::cout << std::flush;
    write(out, nodes, links);
    fmt::print(" done ({} s)\n", timer.elapsed().count());

    out.close();
    timer.reset();

    std::ofstream jsd_out("jsd.csv");
    jsd_out << "node_id,jsd\n";
    for (const auto &[_, node]: nodes) {
        jsd_out << fmt::format("{},{}\n", node.node_id, node.jsd_initial);
    }
    jsd_out.close();

    // 4. Write stats
    std::ofstream stats_out("stats.csv");
    if (!stats_out.is_open()) {
        return;
    }

    stats_out
            << "p,q,t,weighted,directed,num_expanded,num_states,num_links,elapsed\n"
            << fmt::format("{},{},{},{},{},{},{},{},{}\n",
                           p, q, thresh, weighted, directed,
                           num_expanded, num_states, links.size(), total.elapsed().count());

    stats_out.close();
}

int main(int argc, const char **argv) {
    auto program_name = "sparse_states";
    auto help_string = "";

    cxxopts::Options options(program_name, help_string);

    options.add_options()
            ("infile", "Input file", cxxopts::value<std::string>())
            ("outfile", "Output file", cxxopts::value<std::string>())
            ("p", "Return parameter", cxxopts::value<double>()->default_value("1.0"))
            ("q", "In-out parameter", cxxopts::value<double>()->default_value("1.0"))
            ("t", "Threshold", cxxopts::value<double>()->default_value("0.5"))
            ("d,directed", "Directed network", cxxopts::value<bool>()->default_value("false"))
            ("w,weighted", "Weighted network", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print help");

    options.parse_positional({"infile", "outfile"});

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << '\n';
        return 0;
    }

    if (argc < 3) {
        std::cout << options.help() << '\n';
        return 1;
    }

    if (result.count("infile") == 0 || result.count("outfile") == 0) {
        std::cerr << "Missing infile or outfile\n";
        return 1;
    }

    auto infile = result["infile"].as<std::string>();
    auto outfile = result["outfile"].as<std::string>();

    std::ifstream in(infile);
    if (!in.is_open()) {
        std::cerr << "Could not open file " << infile << '\n';
        return 1;
    }

    std::ofstream out(outfile);
    if (!out.is_open()) {
        std::cerr << "Could not open file " << infile << '\n';
        return 1;
    }

    auto p = result["p"].as<double>();
    auto q = result["q"].as<double>();

    if (p == 1.0 && q == 1.0) {
        std::cerr << "At least one of p or q must not be equal to 1\n";
        return 1;
    }

    if (p <= 0.0 || q <= 0.0) {
        std::cerr << "p and q must be positive\n";
        return 1;
    }

    auto thresh = result["t"].as<double>();

    if (thresh <= 0.0) {
        std::cerr << "Threshold must be positive\n";
        return 1;
    }

    auto directed = result["d"].as<bool>();
    auto weighted = result["w"].as<bool>();

    // "weighted" is not implemented
    if (weighted) {
        std::cerr << "Weighted networks are not implemented\n";
        return 1;
    }

#ifdef _OPENMP
#pragma omp parallel default(none)
#pragma omp master
    {
        std::cout << "OpenMP " << _OPENMP << " detected with " << omp_get_num_threads() << " threads.\n";
    }
#endif

    std::cout
            << "Input network: \"" << infile << "\"\n"
            << "Output network: \"" << outfile << "\"\n\n"
            << "Using parameters:\n"
            << '\t' << (directed ? "directed" : "undirected") << '\n'
            << '\t' << (weighted ? "weighted" : "unweighted") << '\n'
            << "\tp = " << p << '\n'
            << "\tq = " << q << '\n'
            << "\tthreshold = " << thresh << " bits\n\n";

    run(in, out, p, q, thresh, weighted, directed);
}
