#include <string_view>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <fmt/format.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

auto run(std::ifstream &in, std::ofstream &out, double p, double q, double thresh, Model model, bool directed, Bias method, Divergence div) {
    Timer total;
    Timer timer;

    // 0. Read links
    fmt::print(stdout, "Reading links... ");
    std::cout << std::flush;
    auto [nodes, tot_weight] = read_links(in, model == Model::Weighted, directed);
    fmt::print("{} nodes, total link weight: {} ({} s)\n", nodes.size(), tot_weight, timer.elapsed().count());

    in.close();
    timer.reset();

    // 1. Expand nodes
    fmt::print(stdout,"Expanding nodes... ");
    std::cout << std::flush;
    auto [num_expanded, num_states] = expand_nodes(nodes, p, q, thresh, model, div);

    auto tot_jsd = std::transform_reduce(nodes.cbegin(), nodes.cend(), 0.0, std::plus{},
                                         [](auto &it) { return it.second.jsd_initial; });
    auto avg_jsd = tot_jsd / static_cast<double>(nodes.size());
    fmt::print("{} expanded, avg. JSD: {} ({} s)\n", num_expanded, avg_jsd, timer.elapsed().count());

    timer.reset();

    // 2. Create links
    fmt::print(stdout, "Creating links... ");
    std::cout << std::flush;
    auto [links, tot_created_weight] = create_links(nodes, p, q, method);
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
            << "p,q,t,weighted,directed,bias_method,divergence,num_expanded,num_states,num_links,elapsed\n"
            << fmt::format("{},{},{},{},{},{},{},{},{},{},{}\n",
                           p, q, thresh, model_to_string(model), directed, bias_to_string(method), div_to_string(div),
                           num_expanded, num_states, links.size(), total.elapsed().count());

    stats_out.close();
}

auto usage(std::string_view program_name) {
    std::cout
            << "Usage:\n"
            << program_name << " infile outfile [-p N] [-q N] [-t N] [-d] [--directed] [-w] [--weighted]\n";
}

auto fail(std::string_view message, std::string_view program_name) {
    std::cerr << message;
    usage(program_name);
    return 1;
}

int main(int argc, const char **argv) {
    std::vector<std::string_view> args(
            argv, std::next(argv, static_cast<std::ptrdiff_t>(argc))
    );

    if (args.size() < 3) {
        usage(args[0]);
        return 1;
    }

    auto &infile = args[1];
    auto &outfile = args[2];

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

    double p = 1.0;
    double q = 1.0;
    double thresh = 0.5;
    bool directed = false;
    auto model = Model::Unweighted;
    auto method = Bias::Biased;
    auto div = Divergence::Unweighted;

    for (std::size_t i = 0; const auto &arg: args) {
        std::stringstream ss;
        if (arg == "-w" || arg == "--weighted") {
            model = Model::Weighted;
        } else if (arg == "-d" || arg == "--directed") {
            throw std::runtime_error("not implemented");
            directed = true;
        } else if (arg == "--unbiased") {
            method = Bias::Unbiased;
        } else if (arg == "--weighted-bias") {
            method = Bias::Weighted;
        } else if (arg == "--weighted-div") {
            div = Divergence::Weighted;
        } else if (arg == "-p" && args.size() != i + 1) {
            ss << args[i + 1];
            ss >> p;
            if (ss.fail() || p <= 0.0) {
                return fail("p must be positive!\n", args[0]);
            }
            ++i;
            continue;
        } else if (arg == "-q" && args.size() != i + 1) {
            ss << args[i + 1];
            ss >> q;
            if (ss.fail() || q <= 0.0) {
                return fail("q must be positive!\n", args[0]);
            }
            ++i;
            continue;
        } else if (arg == "-t" && args.size() != i + 1) {
            ss << args[i + 1];
            ss >> thresh;
            if (ss.fail() || thresh <= 0.0) {
                return fail("t must be positive!\n", args[0]);
            }
            ++i;
            continue;
        }
        ++i;
    }

    if (p == 1.0 && q == 1.0) {
        return fail("Not both p and q can be equal to 1!\n", args[0]);
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
            << '\t' << model_to_string(model) << "\n"
            << '\t' << "write " << bias_to_string(method) << " links" << "\n"
            << '\t' << "using " << div_to_string(div) << " divergence" << "\n"
            << "\tp = " << p << '\n'
            << "\tq = " << q << '\n'
            << "\tthreshold = " << thresh << " bits\n\n";

    run(in, out, p, q, thresh, model, directed, method, div);
}
