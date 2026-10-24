// Emit the reference's per-resolution network plan.
//
// `nr_runtime` does not feed the runner a fixed plan file: it calls
// `nr::make_native_plan(source_width, source_height)` and hands the parsed rows to the graph
// (nr_runtime.cpp, nr_int4_weights.cpp). The runner's `--plan` takes a file, and its default
// (`linux/src/core/nr_frame.txt`) is one frozen geometry, so running the offline runner at another
// resolution needs the plan text for that resolution. This prints it.
//
//   nr_plan 1920 1080 > plan-1080p.txt
//   nr_graph --plan plan-1080p.txt --source-width 1920 --source-height 1080 ...
//
// Build with the reference sources (no Vulkan needed):
//   g++ -std=c++17 -O2 -I<ref>/linux/src -I<ref>/linux/src/core \
//       tools/hwtest/nr_plan.cpp <ref>/linux/src/core/nr_native_plan.cpp -o nr_plan

#include "nr_native_plan.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: nr_plan <width> <height>\n");
        return 2;
    }
    try {
        const auto p = nr::make_native_plan(uint32_t(std::atoi(argv[1])), uint32_t(std::atoi(argv[2])));
        std::fputs(p.text.c_str(), stdout);
        std::fprintf(stderr, "working extent %ux%u, reductions %u/%u, model %s\n",
                     p.width, p.height, p.reductions_x, p.reductions_y, p.model.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nr_plan: %s\n", e.what());
        return 1;
    }
    return 0;
}
