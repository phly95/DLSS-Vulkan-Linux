// rrref: run NeuralHub v0.8.0's own DLSS-RR (preset F) GPU code on Linux through ROCm, as the
// reference a Vulkan port is checked against. Host logic transcribed from neuralhub_host.dll
// (docs/rr/HOST-RE.md): the engine `E` (FUN_180126da0) is kept as NeuralHub's own int array, every
// buffer at the same index, and the frame (FUN_180127f10, FUN_180133980) fills every argument block
// byte for byte from it.
//
//   rrref <rr.gfx1201.co> <packed dir (rr_pack.py)> <frames dir> <out dir> [--dump DIR [--dump-frames a,b]]
//
// <frames dir>/frame.json: {"render":[W,H],"output":[oW,oH],"frames":N}; per frame i: params_i.json
// (jitter, mv_scale, reset, depth_inverted, hw_depth; optional normals_unorm (F+0x9a: xyz = 2v - 1)
// and format codes fmt_col, fmt_mv, fmt_dep, fmt_nrm, fmt_alb, fmt_spa, fmt_rgh) and the game's
// textures, tightly packed: colour col_i, motion vectors mv_i (render pixels), depth dep_i, normals
// nrm_i, diffuse / specular albedo alb_i / spa_i, roughness rgh_i. With the default code a file is
// <name>_i.f16 (RGBA16F: col, nrm, alb, spa; RG16F: mv) or <name>_i.f32 (R32F: dep, rgh); with another
// code it is <name>_i.c<code> (texel sizes: tools/verify/texfmt.py, RR_BPP). A missing or wrongly sized
// file is an error. They go through NeuralHub's own k_rr_gather exactly as the game's textures would
// (row pitch in bytes + its format code). Output: out_i.f32 (RGBA32F, oW*oH*16 B) and a line
// "GPU_FRAME_MS <i> <ms>" per frame; --dump writes every launch's outputs (tools/verify/oracle_dump.hpp).
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "../../verify/oracle_dump.hpp"

static OracleDump g_dump;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

static std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); std::exit(1); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}
static std::string slurps(const std::string& p) { auto v = slurp(p); return std::string(v.begin(), v.end()); }
static void spit(const std::string& p, const void* d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(d), std::streamsize(n));
}
// Tiny reader for flat JSON ("key": number, bool or [numbers]); `from` starts the search later.
static std::vector<double> jget(const std::string& txt, const std::string& key, size_t from = 0) {
    std::vector<double> v;
    size_t k = txt.find("\"" + key + "\"", from);
    if (k == std::string::npos) return v;
    size_t i = txt.find(':', k) + 1;
    while (i < txt.size() && (txt[i] == ' ' || txt[i] == '[' || txt[i] == '\n')) ++i;
    for (;;) {
        char* e = nullptr;
        double d = std::strtod(txt.c_str() + i, &e);
        if (e == txt.c_str() + i) {
            if (txt.compare(i, 4, "true") == 0) { v.push_back(1); i += 4; }
            else if (txt.compare(i, 5, "false") == 0) { v.push_back(0); i += 5; }
            else if (txt.compare(i, 4, "null") == 0) { i += 4; }
            else break;
        } else { v.push_back(d); i = size_t(e - txt.c_str()); }
        while (i < txt.size() && (txt[i] == ' ' || txt[i] == ',' || txt[i] == '\n')) ++i;
        if (i >= txt.size() || txt[i] == ']' || txt[i] == '}') break;
    }
    return v;
}

// A kernel argument block, filled by byte offset exactly as the host lays it out.
struct Args {
    std::vector<uint8_t> b;
    explicit Args(size_t n) : b(n, 0) {}
    template <class T> Args& put(size_t off, T v) { std::memcpy(b.data() + off, &v, sizeof v); return *this; }
    Args& raw(size_t off, const void* p, size_t n) { std::memcpy(b.data() + off, p, n); return *this; }
};

struct Mod {
    hipModule_t m{};
    std::map<std::string, hipFunction_t> fn;
    hipFunction_t get(const char* sym) {
        auto it = fn.find(sym);
        if (it != fn.end()) return it->second;
        hipFunction_t f; CK(hipModuleGetFunction(&f, m, sym));
        return fn[sym] = f;
    }
    void launch(const char* sym, dim3 g, dim3 b, Args& a) {
        size_t sz = a.b.size();
        void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, a.b.data(), HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
        CK(hipModuleLaunchKernel(get(sym), g.x, g.y, g.z, b.x, b.y, b.z, 0, nullptr, nullptr, cfg));
        g_dump.after_launch(sym, g, b, a.b);
        if (std::getenv("RRREF_SYNC")) {
            hipError_t e = hipDeviceSynchronize();
            if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", sym, hipGetErrorString(e)); std::exit(1); }
        }
    }
};

static float f16_to_f32(uint16_t h) {
    uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0x1f) u = s | 0x7f800000 | (m << 13);
    else if (e == 0) { float f = std::ldexp(float(m), -24); std::memcpy(&u, &f, 4); u |= s; }
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}
static uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

// ---- the engine: NeuralHub's 0x3448-byte object, as ints (rr_engine.h) ----
static int E[0x3448 / 4];
static uint64_t& Q(int i) { return *reinterpret_cast<uint64_t*>(&E[i]); }      // a 64-bit field at int index i
static std::map<int, size_t> g_size;
// FUN_180132f90: a device buffer {d, n} at int index i (zeroed, as hipMalloc + the first frame's reset)
static void alloc(int i, size_t n) {
    void* p; CK(hipMalloc(&p, n ? n : 16)); CK(hipMemset(p, 0, n ? n : 16));
    Q(i) = reinterpret_cast<uint64_t>(p); Q(i + 2) = n; g_size[i] = n;
}
static void* P(int i) { return reinterpret_cast<void*>(Q(i)); }

static const char* K_W16 = "_ZN3rrk8k_rr_w16EPKhPDF16_y";
static const char* K_POOLED2 = "_ZN4rrio9k_pooled2EPKjiiPjii";
static const char* K_AE_BLOCK = "_ZN4rrio10k_ae_blockENS_6AEArgsE";
static const char* K_AE_FINAL = "_ZN4rrio10k_ae_finalENS_6AEArgsE";
static const char* K_ENC0 = "_ZN4rrio11k_enc0_featENS_6E0ArgsE";
static const char* K_POST_PREP = "_ZN4rrio11k_post_prepEPKtiPKhS1_iiPNS_3TapE";
static const char* K_POST24 = "_ZN4rrio6k_postILi24EEEvNS_8PostArgsE";
static const char* K_POST26 = "_ZN4rrio6k_postILi26EEEvNS_8PostArgsE";
static const char* K_DS0 = "_ZN4rrio12k_downsampleILi0EEEvPKtS2_iiPKfP15HIP_vector_typeIfLj4EEii";
static const char* K_GATHER = "_ZN6rrunit11k_rr_gatherENS_10GatherArgsE";

// rrk::k_level<C, g, mode, Cdown, Cup, n>
static std::string klevel(int C, int g, int mode, int cd, int cu, int n) {
    char b[160];
    std::snprintf(b, sizeof b, "_ZN3rrk7k_levelILi%dELi%dELi%dELi%dELi%dELi%dEEEvNS_4ArgsE", C, g, mode, cd, cu, n);
    return b;
}

// rrk::Args (0xf8 bytes), the fields every unit shares (FUN_180133980)
static Args unit_args(int u, int w, int h, uint64_t shift) {
    Args a(0xf8);
    a.put(0, w).put(4, h).put(8, shift);
    a.put(0x20, Q(0x1c + 4 * u));                                     // blob
    a.raw(0x28, &E[0x5c + 0x14 * u], 80);                             // item offsets 0..19
    const uint16_t* s = reinterpret_cast<const uint16_t*>(&E[0x19c + 0x96 * u + 0x74]);   // W[u]+0x1d0
    for (int i = 0; i < 4; ++i) a.put(0xa8 + 4 * i, uint32_t(s[i]));
    a.put(0xb8, int(Q(0x1e + 4 * u)));                                // blob size
    uint64_t w16 = Q(0xb14 + 4 * u);
    a.put(0xf0, w16 ? w16 - 2 * Q(0x1c + 4 * u) : 0);
    return a;
}

static void launch_unit(Mod& M, Args& a, const std::string& sym, int n) {
    int w, h, sx, sy;
    std::memcpy(&w, &a.b[0], 4); std::memcpy(&h, &a.b[4], 4);
    std::memcpy(&sx, &a.b[8], 4); std::memcpy(&sy, &a.b[12], 4);
    dim3 grid((w + sx + 7) >> 3, (h + sy + 7) >> 3), block(32, n == 4 ? 1 : 4);
    M.launch(sym.c_str(), grid, block, a);
}

// FUN_180133980(E, p, stream, 0, 0, 0, 0): unit p + 1 of the U-Net
static void level(Mod& M, int p) {
    const int u = p + 1, parity = E[0x18];
    const uint64_t s6_2 = parity == 0 ? 0x600000002ull : 0x200000006ull;
    const uint64_t s4_0 = parity == 0 ? 0x4ull : 0x400000000ull;
    const uint64_t s0_4 = parity == 0 ? 0x400000000ull : 0x4ull;
    // the engine's 64-bit slots by qword index (the decompile's view of E in this function)
    auto q = [](int qi) { return Q(2 * qi); };
    if (p == 0) {
        Args a = unit_args(1, E[0xd], E[0x13], s6_2);
        a.put(0x10, q(0x5aa)).put(0x88, q(0x5e0));
        a.put(0x78, q(0x5ac)).put(0x80, q(0x5b8)).put(0x98, q(0x5d6)).put(0xa0, q(0x5e2));
        launch_unit(M, a, klevel(64, 2, 1, 96, 0, 1), 1);
    } else if (p >= 1 && p <= 4) {
        Args a = unit_args(u, E[0xc + u], E[0x12 + u], u == 5 ? s4_0 : s6_2);
        a.put(0x10, q(0x5b6 + 2 * p)).put(0x88, q(0x5e0 + 2 * p)).put(0x78, q(0x5aa + 2 * u));
        a.put(0x98, q(0x5d4 + 2 * u));
        if (p == 4) { launch_unit(M, a, klevel(160, 8, 0, 0, 0, 1), 1); return; }
        a.put(0x80, q(0x5b6 + 2 * u)).put(0xa0, q(0x5e0 + 2 * u));
        static const int cfg[4][3] = {{96, 4, 128}, {128, 4, 160}, {160, 8, 160}};
        launch_unit(M, a, klevel(cfg[p - 1][0], cfg[p - 1][1], 0, cfg[p - 1][2], 0, 1), 1);
    } else {
        const int k = (p - 5) >> 1, lv = 4 - k;          // decoder level 4 .. 0
        if (((p - 5) & 1) == 0) {                         // dec*_0: up-projection from the level below
            Args a = unit_args(u, E[0xc + lv], E[0x12 + lv], s4_0);
            a.put(0x10, p == 5 ? q(0x5b4) : q(0x5d4 - 2 * k));
            a.put(0x18, q(0x5aa + 2 * lv));
            a.put(0x78, q(0x5c0 + 2 * lv));
            static const int cfg[5][4] = {{160, 8, 160, 1}, {128, 4, 160, 1}, {96, 4, 128, 1}, {64, 2, 96, 1}, {32, 2, 64, 4}};
            launch_unit(M, a, klevel(cfg[k][0], cfg[k][1], 2, 0, cfg[k][2], cfg[k][3]), cfg[k][3]);
        } else if (lv == 0) {                             // dec0_1: the last unit, writes the output features
            Args a = unit_args(u, E[0xc], E[0x12], s0_4);
            a.put(0x10, q(0x5c0));
            a.put(0x88, q(0x5d4)).put(0x90, q(0x5ec)).put(0x98, q(0x5ea));
            a.put(0xc0, q(0x601)).put(0xc8, q(0x60b)).put(0xd0, q(0x60d)).put(0xd8, q(0x60f));
            a.put(0xe0, Q(0));
            a.put(0xe8, *reinterpret_cast<const uint64_t*>(&E[0x19c + 0x96 * u + 0x92]));   // W[u]+0x248
            launch_unit(M, a, klevel(32, 2, 4, 0, 0, 4), 4);
        } else {                                          // dec*_1
            Args a = unit_args(u, E[0xc + lv], E[0x12 + lv], s0_4);
            a.put(0x10, q(0x5c0 + 2 * lv)).put(0x78, q(0x5ca + 2 * lv)).put(0x88, q(0x5d4 + 2 * lv));
            a.put(0x90, p == 6 ? q(0x5de) : q(0x5f4 - 2 * k)).put(0x98, q(0x5ea + 2 * lv));
            static const int cfg[4][2] = {{160, 8}, {128, 4}, {96, 4}, {64, 2}};
            launch_unit(M, a, klevel(cfg[k][0], cfg[k][1], 3, 0, 0, 1), 1);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: rrref <rr.gfx1201.co> <packed dir> <frames dir> <out dir> [--dump DIR [--dump-frames a,b]]\n");
        return 2;
    }
    const std::string co = argv[1], wdir = argv[2], fdir = argv[3], odir = argv[4];
    std::string dump_dir, dump_frames;
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dump" && i + 1 < argc) dump_dir = argv[++i];
        else if (a == "--dump-frames" && i + 1 < argc) dump_frames = argv[++i];
        else { std::fprintf(stderr, "rrref: unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!dump_dir.empty()) g_dump.configure(dump_dir, dump_frames);
    const std::string fj = slurps(fdir + "/frame.json");
    const auto rd = jget(fj, "render"), od = jget(fj, "output");
    const int nframes = int(jget(fj, "frames").at(0));
    Mod M; CK(hipModuleLoad(&M.m, co.c_str()));

    // ---- FUN_180126da0: dimensions ----
    const int W = int(rd[0]), H = int(rd[1]), oW = int(od[0]), oH = int(od[1]);
    E[0] = W; E[1] = H; E[2] = oW; E[3] = oH;
    const int W0 = (W + 31) & ~31, H0 = (H + 31) & ~31;
    E[4] = W0; E[5] = H0;
    E[6] = ((oW + (oW & 1)) * 3) / 2; E[7] = ((oH + (oH & 1)) * 3) / 2;
    E[8] = (E[6] + 1) / 2; E[9] = (E[7] + 1) / 2;
    E[10] = (E[8] + 1) / 2; E[11] = (E[9] + 1) / 2;
    for (int l = 0; l < 6; ++l) { E[0xc + l] = W0 >> l; E[0x12 + l] = H0 >> l; }
    if (float(W) / float(E[6]) > 0.68f || float(H) / float(E[7]) > 0.68f) {
        std::fprintf(stderr, "RR: render / internal > 2/3 (render larger than the output): not a DLSS-RR configuration\n");
        return 1;
    }
    E[0x1a] = 1;                               // history reset pending (the engine is created with it)
    E[0xd0e] = -1; E[0xd0f] = -1;              // no debug probe pixel

    // ---- weights: 16 blobs, item offsets, W[u] scalars ----
    const std::string pj = slurps(wdir + "/params.json");
    size_t pos = 0;
    for (int u = 0; u < 16; ++u) {
        pos = pj.find("\"k\": " + std::to_string(u) + ",", pos);
        auto blob = slurp(wdir + "/blob" + std::to_string(u) + ".bin");
        alloc(0x1c + 4 * u, blob.size());
        CK(hipMemcpy(P(0x1c + 4 * u), blob.data(), blob.size(), hipMemcpyHostToDevice));
        auto meta = jget(pj, "meta", pos);
        for (int i = 0; i < 20; ++i) E[0x5c + 0x14 * u + i] = int(meta.at(i));
        if (u == 0) E[0xbf0] = int(meta.at(20));
        uint16_t* s = reinterpret_cast<uint16_t*>(&E[0x19c + 0x96 * u + 0x74]);
        s[0] = uint16_t(jget(pj, "s1d0", pos).at(0)); s[1] = uint16_t(jget(pj, "s1d2", pos).at(0));
        s[2] = uint16_t(jget(pj, "s1d4", pos).at(0)); s[3] = uint16_t(jget(pj, "s1d6", pos).at(0));
        auto d0 = jget(pj, "dec0_scalars", pos);
        if (d0.size() == 3)
            for (int i = 0; i < 3; ++i) { float f = float(d0[i]); std::memcpy(&E[0x19c + 0x96 * u + 0x92 + i], &f, 4); }
    }
    {
        uint32_t s2598 = uint32_t(jget(pj, "enc0_s2598").at(0));
        std::memcpy(&E[0xb02], &s2598, 4);
        size_t k = pj.find("\"enc0_s259c\"");
        std::string hex = pj.substr(pj.find('"', pj.find(':', k)) + 1, 16);
        uint8_t* b = reinterpret_cast<uint8_t*>(&E[0xb03]);
        for (int i = 0; i < 8; ++i) b[i] = uint8_t(std::stoi(hex.substr(2 * i, 2), nullptr, 16));
    }

    // ---- buffers, in FUN_180126da0's order ----
    const size_t A0 = size_t(H0) * W0;
    alloc(0xb54, A0 * 32); alloc(0xba8, A0 * 8); alloc(0xbc0, A0 * 2);
    alloc(0xb80, A0 * 32); alloc(0xb94, A0 * 32); alloc(0xbd4, A0 * 8);
    static const int Cl[5] = {64, 96, 128, 160, 160}, Cn[4] = {96, 128, 160, 160};   // DAT_1801bb464 / 468
    for (int i = 0; i < 5; ++i) {
        const size_t a = size_t(E[0x13 + i]) * E[0xd + i];
        alloc(0xb58 + 4 * i, size_t(Cl[i]) * a);
        alloc(0xbac + 4 * i, a * 8);
        if (i == 4) break;
        alloc(0xbc4 + 4 * i, a * 2);
        alloc(0xb84 + 4 * i, size_t(Cl[i]) * a);
        alloc(0xb98 + 4 * i, size_t(Cl[i]) * a);
        alloc(0xbd8 + 4 * i, a * 8);
        alloc(0xb70 + 4 * i, size_t(Cn[i]) * E[0x14 + i] * E[0xe + i]);
    }
    alloc(0xbe8, size_t(E[5]) * E[4] * 0x40); alloc(0xbec, size_t(E[5]) * E[4] * 8);
    const size_t R = size_t(W) * H;
    alloc(0xbf2, R * 8); alloc(0xbf6, R * 8); alloc(0xbfa, R * 4); alloc(0xbfe, R * 4); alloc(0xc02, R * 4);
    alloc(0xc16, R * 4); alloc(0xc1a, R * 4); alloc(0xc1e, R * 8);
    hipDeviceProp_t pr; CK(hipGetDeviceProperties(&pr, 0));
    {
        const size_t al = pr.texturePitchAlignment ? pr.texturePitchAlignment : 1;
        int n = E[6];
        while ((size_t(n) * 8) % al) ++n;
        E[0xbf1] = n;
    }
    alloc(0xc06, size_t(E[7]) * E[0xbf1] * 8); alloc(0xc0a, size_t(E[7]) * E[0xbf1] * 8);
    alloc(0xc0e, size_t(E[9]) * E[8] * 4); alloc(0xc12, size_t(E[0xb]) * E[0xa] * 4);
    alloc(0xc22, 0x10); alloc(0xc26, 2); alloc(0xc2a, 2);
    alloc(0xc6e, R * 8); alloc(0xc72, R * 4); alloc(0xc76, R * 4); alloc(0xc7a, R * 8);
    alloc(0xc7e, R * 16); alloc(0xc82, R * 16); alloc(0xc86, R * 4); alloc(0xc8a, R * 4);
    E[0xcd7] = E[2] + (E[2] & 1); E[0xcd8] = E[3] + (E[3] & 1);
    const size_t Oc = size_t(E[0xcd8]) * E[0xcd7], O = size_t(E[3]) * E[2];
    alloc(0xca6, R * 16); alloc(0xcaa, R * 16); alloc(0xcae, O * 8); alloc(0xcb2, R * 4);
    alloc(0xcb6, Oc * 8); alloc(0xcba, O * 16); alloc(0xcbe, Oc * 8); alloc(0xcc6, Oc * 8); alloc(0xcce, Oc * 2);
    alloc(0xcc2, Oc * 8); alloc(0xcca, Oc * 8); alloc(0xcd2, Oc * 2);
    alloc(0xce6, R * 16); alloc(0xcea, R * 8); alloc(0xcda, R * 8); alloc(0xcde, R * 4); alloc(0xce2, R * 4);
    alloc(0xc8e, R * 4); alloc(0xc92, R * 8); alloc(0xc96, R * 4); alloc(0xc9a, R * 16); alloc(0xc9e, R * 4); alloc(0xca2, R * 4);
    alloc(0xcee, 8); alloc(0xcf2, R * 0x20); alloc(0xcf6, size_t(E[7]) * E[6] * 4); alloc(0xcfa, size_t(E[7]) * E[6] * 2);
    alloc(0xcfe, O * 16);
    alloc(0xd02, 0x2010);
    {
        auto bic = slurp(wdir + "/bicubic.bin");
        if (bic.size() != 0x2010) { std::fprintf(stderr, "bicubic.bin: %zu bytes\n", bic.size()); return 1; }
        CK(hipMemcpy(P(0xd02), bic.data(), 0x2010, hipMemcpyHostToDevice));
    }
    // ih_tex[0/1]: the 1.5x history (RGBA16F, pitched), bilinear, clamp, normalized coordinates
    for (int i = 0; i < 2; ++i) {
        hipResourceDesc rdsc{}; rdsc.resType = hipResourceTypePitch2D;
        rdsc.res.pitch2D.devPtr = P(i ? 0xc0a : 0xc06);
        rdsc.res.pitch2D.desc = hipCreateChannelDesc(16, 16, 16, 16, hipChannelFormatKindFloat);
        rdsc.res.pitch2D.width = size_t(E[6]); rdsc.res.pitch2D.height = size_t(E[7]);
        rdsc.res.pitch2D.pitchInBytes = size_t(E[0xbf1]) * 8;
        hipTextureDesc td{}; td.addressMode[0] = hipAddressModeClamp; td.addressMode[1] = hipAddressModeClamp;
        td.addressMode[2] = hipAddressModeWrap; td.filterMode = hipFilterModeLinear; td.readMode = hipReadModeElementType;
        td.normalizedCoords = 1;
        hipTextureObject_t t; CK(hipCreateTextureObject(&t, &rdsc, &td, nullptr));
        Q(0xd08 + 2 * i) = uint64_t(t);
    }

    // ---- the game's inputs (19 slots, as NeuralHub's staging hands them to k_rr_gather) ----
    struct In { void* d = nullptr; int pitch = 0, fmt = -1; };
    In in[19];
    // the inputs this harness supplies; every other slot stays null, so F's "present" flags are 0
    struct Slot { int slot; const char* name; const char* key; int code0; const char* ext0; };
    static const Slot slots[] = {{0, "col", "fmt_col", 0, "f16"}, {1, "mv", "fmt_mv", 2, "f16"}, {2, "dep", "fmt_dep", 3, "f32"},
                                 {3, "nrm", "fmt_nrm", 0, "f16"}, {4, "alb", "fmt_alb", 0, "f16"}, {5, "spa", "fmt_spa", 0, "f16"},
                                 {8, "rgh", "fmt_rgh", 3, "f32"}};
    static const int rr_bpp[24] = {8, 4, 4, 4, 2, 4, 4, 4, 1, 2, 8, 16, 4, 4, 8, 8, 4, 4, 4, 4, 2, 2, 2, 1};   // texfmt.RR_BPP
    for (const Slot& sl : slots) CK(hipMalloc(&in[sl.slot].d, R * 16));
    if (g_dump.enabled()) {
        for (auto& kv : g_size) { char nm[16]; std::snprintf(nm, sizeof nm, "E%04x", kv.first); g_dump.add(nm, P(kv.first), kv.second); }
        for (const Slot& sl : slots) g_dump.add(std::string("in_") + sl.name, in[sl.slot].d, R * 16);
        g_dump.add_handle("E0c06", Q(0xd08)); g_dump.add_handle("E0c0a", Q(0xd0a));
        g_dump.write_index();
    }

    std::vector<float> host(O * 4);
    hipEvent_t t0, t1; CK(hipEventCreate(&t0)); CK(hipEventCreate(&t1));
    for (int fr = 0; fr < nframes; ++fr) {
        const std::string s = std::to_string(fr);
        const std::string pjf = slurps(fdir + "/params_" + s + ".json");
        for (const Slot& sl : slots) {
            const auto c = jget(pjf, sl.key);
            const int code = c.empty() ? sl.code0 : int(c[0]);
            if (code < 0 || code > 23) { std::fprintf(stderr, "params_%s.json: %s %d is not a format code\n", s.c_str(), sl.key, code); return 1; }
            const std::string path = fdir + "/" + sl.name + "_" + s + (code == sl.code0 ? std::string(".") + sl.ext0 : ".c" + std::to_string(code));
            auto v = slurp(path);
            if (v.size() != R * rr_bpp[code]) {
                std::fprintf(stderr, "%s: %zu bytes, expected %zu\n", path.c_str(), v.size(), R * rr_bpp[code]);
                return 1;
            }
            CK(hipMemcpy(in[sl.slot].d, v.data(), v.size(), hipMemcpyHostToDevice));
            in[sl.slot].pitch = W * rr_bpp[code];
            in[sl.slot].fmt = code;
        }
        const auto jit = jget(pjf, "jitter"), mvs = jget(pjf, "mv_scale");
        const bool reset = jget(pjf, "reset").at(0) != 0;
        const bool inverted = jget(pjf, "depth_inverted").at(0) != 0;
        const auto hwv = jget(pjf, "hw_depth");
        const bool hw = hwv.empty() ? true : hwv[0] != 0;
        const auto nuv = jget(pjf, "normals_unorm");
        const bool nrm_unorm = !nuv.empty() && nuv[0] != 0;
        g_dump.begin_frame(fr);

        // ---- k_rr_gather (FUN_180120200): GatherArgs, 0x1e8 bytes ----
        CK(hipEventRecord(t0, nullptr));
        {
            Args g(0x1e8);
            for (int i = 0; i < 19; ++i) {
                g.put(8 * i, reinterpret_cast<uint64_t>(in[i].d));
                g.put(0x98 + 4 * i, in[i].pitch);
                g.put(0xe4 + 4 * i, in[i].fmt);
            }
            static const int outs[] = {0xc6e, 0xc72, 0xc76, 0xc7a, 0xc7e, 0xc82, 0xc86, 0xc8a, 0xc8e, 0xc92, 0xc96, 0xc9a,
                                       0xc9e, 0xca2, 0xce6, 0xcea, 0xca6, 0xcaa, 0xcb2, 0xcae, 0xcda};
            for (int i = 0; i < 21; ++i) g.put(0x130 + 8 * i, Q(outs[i]));
            g.put(0x1d8, Q(0));
            g.put(0x1e0, 0);
            M.launch(K_GATHER, dim3((W + 15) >> 4, (H + 15) >> 4), dim3(16, 16), g);
        }

        // ---- F (the frame parameters FUN_180120200 builds), 0xd0 bytes ----
        Args F(0xd0);
        F.put(0x00, float(jit.at(0))).put(0x04, float(jit.at(1)));
        F.put<uint8_t>(0x08, reset);
        F.put(0x0c, float(mvs.at(0))).put(0x10, float(mvs.at(1)));
        F.put<uint8_t>(0x16, inverted);
        F.put<uint8_t>(0x98, in[3].fmt != 0);
        F.put<uint8_t>(0x99, hw);
        F.put<uint8_t>(0x9a, nrm_unorm);                             // job +0x58: UNORM normals (xyz = 2v - 1)
        F.put<uint8_t>(0x9b, in[8].d != nullptr);
        F.put<uint8_t>(0x9c, in[9].d != nullptr);
        F.put<uint8_t>(0x9d, in[10].d != nullptr);
        F.put<uint8_t>(0x9e, in[11].d != nullptr);
        F.put<uint8_t>(0x9f, in[12].d != nullptr);
        F.put<uint8_t>(0xa0, in[13].d != nullptr);
        F.put(0xa4, 1.0f).put(0xa8, 0);                               // job +0xb4 (1.0f, 0)
        F.put(0xac, 1.0f).put(0xb0, 1.0f).put(0xb4, 0.0f).put(0xb8, 1.0f);   // job +0xa0 defaults
        F.put<uint8_t>(0xc4, (in[1].fmt & ~2) != 0);
        F.put<uint8_t>(0xc5, in[17].d != nullptr);
        const uint8_t* f8 = F.b.data();
        auto fb = [&](int o) { return f8[o]; };
        auto fq = [&](int o) { uint64_t v; std::memcpy(&v, f8 + o, 8); return v; };
        auto fi = [&](int o) { int v; std::memcpy(&v, f8 + o, 4); return v; };

        // ---- FUN_180127f10 ----
        const int parity = E[0x18];
        const bool rst = ((E[0x1a] & 0xff) | fb(8)) & 1;
        if (rst) {
            const uint16_t one = 0x3c00;
            CK(hipMemcpy(P(0xc26), &one, 2, hipMemcpyHostToDevice));
            CK(hipMemcpy(P(0xc2a), &one, 2, hipMemcpyHostToDevice));
        } else {
            Args a(0x20);
            a.put(0, Q(0xc0e)).put(8, E[8]).put(12, E[9]).put(16, Q(0xc12)).put(24, E[10]).put(28, E[11]);
            M.launch(K_POOLED2, dim3((E[10] + 15) >> 4, (E[11] + 15) >> 4), dim3(16, 16), a);
        }
        {   // auto exposure
            Args a(0x38);
            a.put(0x00, Q(0xc6e)).put(0x08, W).put(0x0c, H).put(0x10, Q(0xcee));
            a.put(0x18, 2.3283064365386963e-06f / float(W * H)).put(0x1c, int(rst));
            a.put(0x20, Q(0xc26 + 4 * parity)).put(0x28, Q(0xc26 + 4 * (parity ^ 1))).put(0x30, Q(0xc22));
            M.launch(K_AE_BLOCK, dim3((W + 15) >> 4, (H + 15) >> 4), dim3(16, 16), a);
            M.launch(K_AE_FINAL, dim3(1), dim3(1), a);
        }
        {   // k_enc0_feat: rrio::E0Args, 0x248 bytes
            Args a(0x248);
            a.put(0x000, Q(4)).put(0x008, W).put(0x00c, H);
            a.put(0x010, E[0x19]).put(0x014, int(rst)).put(0x018, int(fb(0x16))).put(0x01c, int(fb(0x14)));
            a.put(0x020, Q(0xc86)).put(0x028, fb(0x15) ? Q(0xc8a) : 0);
            a.raw(0x030, f8 + 0x18, 0x80);
            a.put(0x0b0, int(fb(0x99))).put(0x0b4, 0);
            a.put(0x0b8, fb(0xc5) ? Q(0xcda) : 0);
            a.put(0x0c0, Q(0xcde + 4 * parity)).put(0x0c8, Q(0xcde + 4 * (parity ^ 1)));
            a.put(0x0d0, fb(0x98) ? Q(0xce6) : 0);
            a.put(0x0d8, int(fb(0x9a)));
            a.put(0x0dc, fq(0xac)).put(0x0e4, fq(0xb4)).put(0x0ec, 0);
            a.put(0x0f0, fb(0x9b) ? Q(0xc8e) : 0).put(0x0f8, fb(0x9c) ? Q(0xc92) : 0);
            a.put(0x100, fb(0xc4) ? Q(0xcea) : 0).put(0x108, fb(0x9d) ? Q(0xc96) : 0);
            a.put(0x110, fb(0x9e) ? Q(0xc9a) : 0).put(0x118, fb(0x9f) ? Q(0xc9e) : 0);
            a.put(0x120, fb(0xa0) ? Q(0xca2) : 0).put(0x128, fi(0xa4)).put(0x12c, 0);
            a.put(0x130, Q(0xc6e)).put(0x138, Q(0xc72)).put(0x140, Q(0xc76)).put(0x148, Q(0xc7a));
            a.put(0x150, Q(0xc7e)).put(0x158, Q(0xc82)).put(0x160, Q(0xbf2 + 4 * parity));
            a.put(0x168, Q(0xc0e)).put(0x170, Q(0xc12)).put(0x178, Q(0xc02));
            a.put(0x180, E[8]).put(0x184, E[9]).put(0x188, E[10]).put(0x18c, E[11]);
            a.put(0x190, Q(0xc22)).put(0x198, Q(0x1c) + uint64_t(E[0xbf0]));
            a.put(0x1a0, fq(0x00));
            a.put(0x1a8, E[0xd06]).put(0x1ac, E[0xd07]);                // previous jitter (raw floats)
            const float fW = float(W), fH = float(H), f6 = float(E[6]), f7 = float(E[7]);
            float pjx, pjy; std::memcpy(&pjx, &E[0xd06], 4); std::memcpy(&pjy, &E[0xd07], 4);
            a.put(0x1b0, fW / f6).put(0x1b4, fH / f7).put(0x1b8, f6 / fW).put(0x1bc, f7 / fH);
            a.put(0x1c0, 1.0f / fW).put(0x1c4, 1.0f / fH).put(0x1c8, 1.0f / f6).put(0x1cc, 1.0f / f7);
            a.put(0x1d0, (1.0f / fW) * pjx).put(0x1d4, (1.0f / fH) * pjy);
            a.put(0x1d8, 0.5f / float(E[8])).put(0x1dc, 0.5f / float(E[9]));
            a.put(0x1e0, 0.5f / float(E[10])).put(0x1e4, 0.5f / float(E[11]));
            a.put(0x1e8, float(E[8])).put(0x1ec, float(E[9]));
            a.put(0x1f0, f6).put(0x1f4, f7);
            a.put(0x1f8, Q(0xb03));
            const uint16_t* sc = reinterpret_cast<const uint16_t*>(&E[0xb02]);
            a.put(0x200, f16_to_f32(sc[0])).put(0x204, f16_to_f32(sc[1]));
            a.put(0x208, fi(0xa8)).put(0x20c, 1.0f);
            float mx, my; std::memcpy(&mx, f8 + 0x0c, 4); std::memcpy(&my, f8 + 0x10, 4);
            a.put(0x210, mx / (fW / f6)).put(0x214, my / (fH / f7));
            a.put(0x218, fq(0xbc));
            a.put(0x220, Q(0xbe8)).put(0x228, Q(0xbec)).put(0x230, Q(0xbf2 + 4 * (parity ^ 1)));
            a.put(0x238, Q(0xbfa)).put(0x240, Q(0xbfe));
            M.launch(K_ENC0, dim3((E[4] + 15) >> 4, (E[5] + 15) >> 4), dim3(16, 16), a);
        }
        {   // level 0 (FUN_1801346f0): enc0 unit
            Args a(0xf8);
            a.put(0x00, E[0xc]).put(0x04, E[0x12]);
            a.put(0x08, parity ? 0x200000006ull : 0x600000002ull);
            a.put(0x10, Q(0xbe8)).put(0x18, uint64_t(0)).put(0x20, Q(0x1c));
            a.raw(0x28, &E[0x5c], 80);
            a.put(0x78, Q(0xb54)).put(0x88, Q(0xbec)).put(0x98, Q(0xba8)).put(0xa0, Q(0xbc0));
            const uint16_t* s0 = reinterpret_cast<const uint16_t*>(&E[0x19c + 0x74]);
            for (int i = 0; i < 4; ++i) a.put(0xa8 + 4 * i, uint32_t(s0[i]));
            a.put(0xb8, E[0x1e]);
            a.put(0xf0, Q(0xb14) ? Q(0xb14) - 2 * Q(0x1c) : 0);
            launch_unit(M, a, klevel(32, 2, 5, 0, 0, 4), 4);
        }
        for (int p = 0; p < 15; ++p) level(M, p);
        {   // k_post_prep
            Args a(0x30);
            a.put(0, Q(0xbd4)).put(8, E[4]).put(16, Q(0xc16)).put(24, Q(0xc1e)).put(32, W).put(36, H).put(40, Q(0xcf2));
            M.launch(K_POST_PREP, dim3((W + 15) >> 4, (H + 15) >> 4), dim3(16, 16), a);
        }
        {   // k_post<24|26>: rrio::PostArgs, 0xb8 bytes
            Args a(0xb8);
            const float fW = float(W), fH = float(H), f6 = float(E[6]), f7 = float(E[7]);
            a.put(0x00, Q(0)).put(0x08, Q(6)).put(0x10, int(rst));
            a.put(0x14, fW / f6).put(0x18, fH / f7);
            a.put(0x1c, fq(0));
            a.put(0x24, 1.0f / fW).put(0x28, 1.0f / fH).put(0x2c, 1.0f / f6).put(0x30, 1.0f / f7);
            a.put(0x34, E[0xafa]).put(0x38, E[0xbf1]).put(0x3c, 0);
            a.put(0x40, Q(0xcf2)).put(0x48, Q(0xc22)).put(0x50, Q(0xc1a)).put(0x58, Q(0xbf2 + 4 * (parity ^ 1)));
            a.put(0x60, Q(0xbfa)).put(0x68, Q(0xbfe)).put(0x70, Q(0xc06 + 4 * parity)).put(0x78, Q(0xd02));
            a.put(0x80, Q(0xd08 + 2 * parity)).put(0x88, Q(0xc06 + 4 * (parity ^ 1)));
            a.put(0x90, Q(0xcf6)).put(0x98, Q(0xcfa)).put(0xa0, Q(0xc0e));
            a.put(0xa8, Q(0xd0c)).put(0xb0, E[0xd0e]).put(0xb4, E[0xd0f]);
            const bool k26 = 0.5f < fW / f6 || 0.5f < fH / f7;
            M.launch(k26 ? K_POST26 : K_POST24, dim3((E[6] + 15) >> 4, (((E[7] + 1) / 2) + 15) >> 4), dim3(16, 16), a);
        }
        {   // k_downsample<0>: the 1.5x result to the output size
            Args a(0x30);
            a.put(0, Q(0xcf6)).put(8, Q(0xcfa)).put(16, E[6]).put(20, E[7]).put(24, Q(0xc22)).put(32, Q(0xcfe));
            a.put(40, E[2]).put(44, E[3]);
            M.launch(K_DS0, dim3((E[2] + 15) >> 4, (E[3] + 15) >> 4), dim3(16, 16), a);
        }
        E[0xd06] = fi(0); E[0xd07] = fi(4);
        E[0x18] ^= 1;
        E[0x19] = (E[0x19] + 1) & 7;
        E[0x1a] = 0;
        CK(hipEventRecord(t1, nullptr));
        CK(hipDeviceSynchronize());
        g_dump.end_frame();
        float ms = 0; CK(hipEventElapsedTime(&ms, t0, t1));
        CK(hipMemcpy(host.data(), P(0xcfe), O * 16, hipMemcpyDeviceToHost));
        spit(odir + "/out_" + s + ".f32", host.data(), O * 16);
        float ex[4]; CK(hipMemcpy(ex, P(0xc22), 16, hipMemcpyDeviceToHost));
        double sum = 0; int nan = 0;
        for (size_t i = 0; i < O * 4; ++i) { if (std::isfinite(host[i])) sum += host[i]; else ++nan; }
        std::printf("frame %d: %.2f ms GPU, mean %.5f, non-finite %d, exposure (%g %g %g %g)\nGPU_FRAME_MS %d %.4f\n", fr, ms,
                    sum / double(O * 4), nan, ex[0], ex[1], ex[2], ex[3], fr, ms);
        std::fflush(stdout);
    }
    return 0;
}
