// Per-launch dumps for the ROCm oracles (srref, rrref): the referee's record of what NeuralHub's own
// kernels read and wrote, so a Vulkan kernel can be checked alone (tools/verify/diffdump.py).
//
// Every device buffer of the engine is registered by name. For each dumped frame:
//   <dir>/frame<F>/start.<name>.bin       every registered buffer before the frame's first launch
//   <dir>/frame<F>/<NNN>.<name>.bin       a buffer whose bytes changed in launch NNN, after it
//   <dir>/frame<F>/launches.jsonl         one line per launch: kernel, grid, block, the argument block
//                                         in hex, which argument words point into which buffer, what
//                                         changed
//   <dir>/buffers.json                    name, size, row pitch (pitched allocations)
// The inputs of launch NNN are the newest copy of each buffer from start.* and the launches before it.
// Dumping synchronises after every launch; its frame times are not timings.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>

struct OracleDump {
    struct Buf {
        std::string name;
        uintptr_t d = 0;
        size_t n = 0, pitch = 0;
        std::vector<uint8_t> last;
    };
    std::vector<Buf> bufs;
    std::vector<std::pair<uint64_t, std::string>> handles;   // texture objects, by value
    std::string dir;
    std::set<int> frames;
    bool on = false;
    int frame = -1, launch_no = 0;
    FILE* log = nullptr;

    static void die(const char* what, const std::string& p) {
        std::fprintf(stderr, "oracle dump: %s %s\n", what, p.c_str());
        std::exit(1);
    }
    static void mkdirs(const std::string& p) {
        for (size_t i = 1; i <= p.size(); ++i)
            if (i == p.size() || p[i] == '/') mkdir(p.substr(0, i).c_str(), 0755);
    }
    static void spit(const std::string& p, const void* d, size_t n) {
        FILE* f = std::fopen(p.c_str(), "wb");
        if (!f || std::fwrite(d, 1, n, f) != n) die("cannot write", p);
        std::fclose(f);
    }
    // "--dump DIR" and "--dump-frames a,b,c" (default: every frame)
    void configure(const std::string& d, const std::string& list) {
        dir = d;
        mkdirs(dir);
        size_t i = 0;
        while (i < list.size()) {
            size_t j = list.find(',', i);
            if (j == std::string::npos) j = list.size();
            frames.insert(std::atoi(list.substr(i, j - i).c_str()));
            i = j + 1;
        }
    }
    bool enabled() const { return !dir.empty(); }
    void add(const std::string& name, const void* d, size_t n, size_t pitch = 0) {
        if (!d || !n) return;
        bufs.push_back({name, reinterpret_cast<uintptr_t>(d), n, pitch, {}});
    }
    void add_handle(const std::string& name, uint64_t h) { handles.push_back({h, name}); }
    void write_index() {
        if (!enabled()) return;
        std::string s = "[\n";
        for (size_t i = 0; i < bufs.size(); ++i) {
            char b[256];
            std::snprintf(b, sizeof b, "  {\"name\": \"%s\", \"bytes\": %zu, \"pitch\": %zu}%s\n", bufs[i].name.c_str(),
                          bufs[i].n, bufs[i].pitch, i + 1 < bufs.size() ? "," : "");
            s += b;
        }
        s += "]\n";
        spit(dir + "/buffers.json", s.data(), s.size());
    }
    std::vector<uint8_t> fetch(const Buf& b) {
        std::vector<uint8_t> h(b.n);
        if (hipMemcpy(h.data(), reinterpret_cast<void*>(b.d), b.n, hipMemcpyDeviceToHost) != hipSuccess)
            die("hipMemcpy failed for", b.name);
        return h;
    }
    void begin_frame(int f) {
        frame = f;
        on = enabled() && (frames.empty() || frames.count(f));
        launch_no = 0;
        if (!on) return;
        if (hipDeviceSynchronize() != hipSuccess) die("sync failed before frame", std::to_string(f));
        const std::string fd = dir + "/frame" + std::to_string(f);
        mkdirs(fd);
        for (auto& b : bufs) {
            b.last = fetch(b);
            spit(fd + "/start." + b.name + ".bin", b.last.data(), b.n);
        }
        log = std::fopen((fd + "/launches.jsonl").c_str(), "w");
        if (!log) die("cannot write", fd + "/launches.jsonl");
    }
    std::string where(uint64_t v) const {
        for (auto& b : bufs)
            if (v >= b.d && v < b.d + b.n) {
                char s[160];
                std::snprintf(s, sizeof s, "%s+%llu", b.name.c_str(), (unsigned long long)(v - b.d));
                return s;
            }
        for (auto& h : handles)
            if (v == h.first && v != 0) return "texture:" + h.second;
        return "";
    }
    void after_launch(const char* sym, dim3 g, dim3 blk, const std::vector<uint8_t>& args) {
        if (!on) return;
        hipError_t e = hipDeviceSynchronize();
        if (e != hipSuccess) die("kernel failed:", std::string(sym) + " " + hipGetErrorString(e));
        const std::string fd = dir + "/frame" + std::to_string(frame);
        char no[16];
        std::snprintf(no, sizeof no, "%03d", launch_no);
        std::string wrote, ptrs, hex;
        for (auto& b : bufs) {
            std::vector<uint8_t> now = fetch(b);
            if (now != b.last) {
                spit(fd + "/" + no + "." + b.name + ".bin", now.data(), b.n);
                wrote += std::string(wrote.empty() ? "" : ", ") + "\"" + b.name + "\"";
                b.last.swap(now);
            }
        }
        for (size_t o = 0; o + 8 <= args.size(); o += 8) {
            uint64_t v;
            std::memcpy(&v, args.data() + o, 8);
            std::string w = where(v);
            if (!w.empty()) ptrs += std::string(ptrs.empty() ? "" : ", ") + "\"" + std::to_string(o) + "\": \"" + w + "\"";
        }
        for (uint8_t c : args) {
            char x[3];
            std::snprintf(x, sizeof x, "%02x", c);
            hex += x;
        }
        std::fprintf(log, "{\"n\": %d, \"kernel\": \"%s\", \"grid\": [%u, %u, %u], \"block\": [%u, %u, %u], "
                          "\"args\": \"%s\", \"pointers\": {%s}, \"wrote\": [%s]}\n",
                     launch_no, sym, g.x, g.y, g.z, blk.x, blk.y, blk.z, hex.c_str(), ptrs.c_str(), wrote.c_str());
        std::fflush(log);
        ++launch_no;
    }
    void end_frame() {
        if (log) std::fclose(log);
        log = nullptr;
        on = false;
    }
};
