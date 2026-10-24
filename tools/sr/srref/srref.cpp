// srref: run NeuralHub v0.8.0's own DLSS-SR (preset M) GPU code on Linux through ROCm, as the
// reference the Vulkan port is checked against. Host logic transcribed from neuralhub_host.dll
// (docs/sr/HOST-RE.md): engine layout (FUN_18010db80), frame order (FUN_180101f60) and every
// launcher's argument struct, byte for byte.
//
//   srref <sr.gfx1201.co> <packed weight dir (sr_pack.py M)> <frames dir> <out dir>
//
// <frames dir>/frame.json: {"render":[W,H],"output":[oW,oH],"frames":N}; per frame i, files
// col_i.f16 (RGBA16F, W*H*8 B), dep_i.f32 (W*H*4 B), mv_i.f16 (RG16F, W*H*4 B) and params_i.json
// with jitter, jitter_prev, mv_scale, exposure, pre_exposure, reset, depth_inverted.
// By default those three buffers are written as the engine's g_col / g_dep / g_mv directly; with
// --gather they are uploaded as the game's textures (row pitch in bytes, format code 0 = RGBA16F /
// RG16F / R32F) and NeuralHub's srunit::k_sr_gather fills the engine buffers, as in a game.
// Output: out_i.f32 (RGBA32F, oW*oH*16 B), plus every stage tensor of the
// last frame with --dump.
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

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

static std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", p.c_str()); std::exit(1); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}
static void spit(const std::string& p, const void* d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(d), std::streamsize(n));
}
// Tiny reader for the flat JSON the frame generator writes ("key": number or [numbers]).
static std::vector<double> jget(const std::string& txt, const std::string& key) {
    std::vector<double> v;
    size_t k = txt.find("\"" + key + "\"");
    if (k == std::string::npos) return v;
    size_t i = txt.find(':', k) + 1;
    while (i < txt.size() && (txt[i] == ' ' || txt[i] == '[')) ++i;
    for (;;) {
        char* e = nullptr;
        double d = std::strtod(txt.c_str() + i, &e);
        if (e == txt.c_str() + i) {
            if (txt.compare(i, 4, "true") == 0) { v.push_back(1); i += 4; }
            else if (txt.compare(i, 5, "false") == 0) { v.push_back(0); i += 5; }
            else break;
        } else { v.push_back(d); i = size_t(e - txt.c_str()); }
        while (i < txt.size() && (txt[i] == ' ' || txt[i] == ',')) ++i;
        if (i >= txt.size() || txt[i] == ']' || txt[i] == '}' || txt[i] == '\n') break;
    }
    return v;
}

// A kernel argument block, filled by byte offset exactly as the host lays it out.
struct Args {
    std::vector<uint8_t> b;
    explicit Args(size_t n) : b(n, 0) {}
    template <class T> Args& put(size_t off, T v) { std::memcpy(b.data() + off, &v, sizeof v); return *this; }
    Args& f2(size_t off, float x, float y) { return put(off, x).put(off + 4, y); }
    Args& i2(size_t off, int x, int y) { return put(off, x).put(off + 4, y); }
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
        if (std::getenv("SRREF_SYNC")) {
            hipError_t e = hipDeviceSynchronize();
            if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", sym, hipGetErrorString(e)); std::exit(1); }
        }
    }
};

static void* dalloc(size_t n) { void* p; CK(hipMalloc(&p, n)); CK(hipMemset(p, 0, n)); return p; }

static uint16_t f32_to_f16_rne(float f) {
    // NeuralHub's host conversion (FUN_180115ee0's tail): round to nearest even, NaN keeps its sign.
    uint32_t u; std::memcpy(&u, &f, 4);
    uint16_t s = uint16_t((u >> 16) & 0x8000);
    uint32_t a = u & 0x7fffffff;
    if (a > 0x7f800000) return s | 0x7e00;
    if (a >= 0x477ff000) return s | 0x7c00;
    if (a < 0x38800000) {
        float af; std::memcpy(&af, &a, 4);
        return s | uint16_t(std::nearbyint(double(af) * 16777216.0));
    }
    return s | uint16_t(((a & 0x3fff) == 0x3000 || (a & 0x1fff) > 0x1000) + (a >> 13) + 0x4000);
}
static float f16_to_f32(uint16_t h) {
    uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (e == 0x1f) u = s | 0x7f800000 | (m << 13);
    else if (e == 0) { float f = std::ldexp(float(m), -24); std::memcpy(&u, &f, 4); u |= s; }
    else u = s | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4); return f;
}

struct Tensor { void* d; int C, W, H; };

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: srref <sr.gfx1201.co> <weights dir> <frames dir> <out dir> [--dump]\n"); return 2; }
    const std::string co = argv[1], wdir = argv[2], fdir = argv[3], odir = argv[4];
    bool dump = false, presetL = false, gather = false;
    for (int i = 5; i < argc; ++i) {
        if (std::string(argv[i]) == "--dump") dump = true;
        if (std::string(argv[i]) == "--L") presetL = true;
        if (std::string(argv[i]) == "--gather") gather = true;
    }
    const std::string fj(reinterpret_cast<const char*>(slurp(fdir + "/frame.json").data()), slurp(fdir + "/frame.json").size());
    const auto rd = jget(fj, "render"), od = jget(fj, "output");
    const int nframes = int(jget(fj, "frames").at(0));

    Mod M; CK(hipModuleLoad(&M.m, co.c_str()));

    // ---- engine dimensions (FUN_18010db80, FUN_18010fec0) ----
    const int W = int(rd[0]), H = int(rd[1]), oW = int(od[0]), oH = int(od[1]);
    const int W0 = (W + 31) & ~31, H0 = (H + 31) & ~31;
    const int L1w = W0 >> 1, L1h = H0 >> 1, L2w = W0 >> 2, L2h = H0 >> 2, L3w = W0 >> 3, L3h = H0 >> 3;
    const int W15 = (oW * 3 + 1) >> 1, H15 = (oH * 3 + 1) >> 1;
    const int hW15 = (W15 + 1) / 2, hH15 = (H15 + 1) / 2;
    const int mvpitch = (W + 63) & ~63;
    const bool post_3_2 = (float(W15) / float(W) < 3.0f) || (float(H15) / float(H) < 3.0f);
    const size_t rp = size_t(W) * H;

    // ---- weights: wbuf = stages 0..9, enc0, dec0, each 256-aligned ----
    std::vector<std::vector<uint8_t>> blob;
    for (int i = 0; i < 10; ++i) blob.push_back(slurp(wdir + "/stage" + std::to_string(i) + ".bin"));
    blob.push_back(slurp(wdir + "/enc0.bin"));
    blob.push_back(slurp(wdir + "/dec0.bin"));
    std::vector<size_t> woff;
    size_t tot = 0;
    for (auto& b : blob) { woff.push_back(tot); tot += (b.size() + 255) & ~size_t(255); }
    uint8_t* wbuf = static_cast<uint8_t*>(dalloc(tot));
    for (size_t i = 0; i < blob.size(); ++i) CK(hipMemcpy(wbuf + woff[i], blob[i].data(), blob[i].size(), hipMemcpyHostToDevice));
    const std::string pj(reinterpret_cast<const char*>(slurp(wdir + "/params.json").data()), slurp(wdir + "/params.json").size());
    const float dec0_s0 = float(jget(pj, "dec0_s0").at(0));
    const float dec0_f0 = float(jget(pj, "dec0_f0").at(0)), dec0_f1 = float(jget(pj, "dec0_f1").at(0));
    const bool enc0L_safe = pj.find("\"enc0L\"") == std::string::npos ||
        pj.compare(pj.find(':', pj.find("\"fp8_safe\"", pj.find("\"enc0L\"")) + 1) + 2, 4, "true") == 0;
    const bool dec0L_safe = pj.find("\"dec0L\"") == std::string::npos ||
        pj.compare(pj.find(':', pj.find("\"fp8_safe\"", pj.find("\"dec0L\"")) + 1) + 2, 4, "true") == 0;
    std::vector<bool> safe;
    for (size_t k = 0, pos = 0; k < 10; ++k) {
        pos = pj.find("\"fp8_safe\"", pos) + 1;
        safe.push_back(pj.compare(pj.find(':', pos) + 2, 4, "true") == 0);
    }

    // ---- device globals ----
    {
        // g_sr_lut: 1/sqrt(h) then 1/h for every f16 bit pattern h
        std::vector<uint16_t> lut(0x20000);
        for (uint32_t h = 0; h < 0x10000; ++h) {
            float f = f16_to_f32(uint16_t(h));
            double r = std::sqrt(double(f));          // negative -> NaN (sign set, as the CRT's sqrt)
            if (std::isnan(r) && f < 0) r = -std::numeric_limits<double>::quiet_NaN();
            float q = float(1.0 / r);
            lut[h] = f32_to_f16_rne(q);
            lut[0x10000 + h] = f32_to_f16_rne(1.0f / f);
        }
        void* lutbuf = dalloc(lut.size() * 2);
        CK(hipMemcpy(lutbuf, lut.data(), lut.size() * 2, hipMemcpyHostToDevice));
        hipDeviceptr_t g; size_t gs;
        CK(hipModuleGetGlobal(&g, &gs, M.m, "_ZN2sr8g_sr_lutE"));
        CK(hipMemcpy(g, &lutbuf, sizeof lutbuf, hipMemcpyHostToDevice));
        void* unorm8 = dalloc(512);
        Args a(8); a.put(0, unorm8);
        M.launch("_ZN2sr2io17k_sr_unorm8_tableEPt", dim3(1), dim3(256), a);
        CK(hipModuleGetGlobal(&g, &gs, M.m, "_ZN2sr2io10g_unorm8_pE"));
        CK(hipMemcpy(g, &unorm8, sizeof unorm8, hipMemcpyHostToDevice));
    }

    // ---- engine buffers ----
    const int lvC[9] = {64, 64, 96, 96, 128, 128, 128, 96, 64};
    const int lvW[9] = {L1w, L1w, L2w, L2w, L3w, L3w, L3w, L2w, L1w};
    const int lvH[9] = {L1h, L1h, L2h, L2h, L3h, L3h, L3h, L2h, L1h};
    Tensor t[9];
    for (int i = 0; i < 9; ++i) t[i] = {dalloc(size_t(lvC[i]) * lvW[i] * lvH[i]), lvC[i], lvW[i], lvH[i]};
    void* prev_mvec[2] = {dalloc(rp * 4), dalloc(rp * 4)};
    void* mv_ero = dalloc(rp * 4);
    void* mv_nondil = dalloc(size_t(H) * mvpitch * 4);
    void* hidden[2] = {dalloc(rp * 4), dalloc(rp * 4)};
    void* gates = dalloc(rp * 4);
    void* factors = dalloc(rp * 4);
    void* aniso = dalloc(rp * 8);
    void* dhist = dalloc(size_t(hW15) * hH15 * 4);
    void* full15 = dalloc(size_t(W15) * H15 * 4);          // E q0x45
    void* out = dalloc(size_t(oW) * oH * 16);
    void* expo = dalloc(4);
    { uint32_t one = 0x3c003c00; CK(hipMemcpy(expo, &one, 4, hipMemcpyHostToDevice)); }
    void* ihist[2]; size_t ihpitch = 0;
    hipTextureObject_t ihtex[2], mvntex;
    for (int i = 0; i < 2; ++i) {
        CK(hipMallocPitch(&ihist[i], &ihpitch, size_t(W15) * 8, H15));
        CK(hipMemset2D(ihist[i], ihpitch, 0, size_t(W15) * 8, H15));
    }
    auto mktex = [](void* p, size_t pitch, int w, int h, hipChannelFormatDesc cd) {
        hipResourceDesc r{}; r.resType = hipResourceTypePitch2D;
        r.res.pitch2D.devPtr = p; r.res.pitch2D.desc = cd;
        r.res.pitch2D.width = w; r.res.pitch2D.height = h; r.res.pitch2D.pitchInBytes = pitch;
        hipTextureDesc d{};
        d.addressMode[0] = hipAddressModeClamp; d.addressMode[1] = hipAddressModeClamp; d.addressMode[2] = hipAddressModeWrap;
        d.filterMode = hipFilterModeLinear; d.readMode = hipReadModeElementType; d.normalizedCoords = 1;
        hipTextureObject_t o; CK(hipCreateTextureObject(&o, &r, &d, nullptr)); return o;
    };
    for (int i = 0; i < 2; ++i)
        ihtex[i] = mktex(ihist[i], ihpitch, W15, H15, hipCreateChannelDesc(16, 16, 16, 16, hipChannelFormatKindFloat));
    mvntex = mktex(mv_nondil, size_t(mvpitch) * 4, W, H, hipCreateChannelDesc(16, 16, 0, 0, hipChannelFormatKindFloat));

    void* l0_skip = presetL ? dalloc(size_t(W0) * H0 * 32) : nullptr;
    void* l0_in = presetL ? dalloc(size_t(W0 + 8) * (H0 + 8) * 32) : nullptr;
    void* g_col = dalloc(rp * 8);
    void* g_dep = dalloc(rp * 4);
    void* g_mv = dalloc(rp * 4);
    void* in_col = gather ? dalloc(rp * 8) : nullptr;     // the game's textures (--gather)
    void* in_dep = gather ? dalloc(rp * 4) : nullptr;
    void* in_mv = gather ? dalloc(rp * 4) : nullptr;
    void* enc0_w = wbuf + woff[10];
    void* dec0_w = wbuf + woff[11];

    int p = 0, fc = 0;   // E[14] parity, E[15] frame counter (mod 8)
    std::vector<uint8_t> hostout(size_t(oW) * oH * 16);
    for (int fi = 0; fi < nframes; ++fi) {
        const std::string sfx = "_" + std::to_string(fi);
        auto col = slurp(fdir + "/col" + sfx + ".f16"), dep = slurp(fdir + "/dep" + sfx + ".f32"), mv = slurp(fdir + "/mv" + sfx + ".f16");
        CK(hipMemcpy(gather ? in_col : g_col, col.data(), rp * 8, hipMemcpyHostToDevice));
        CK(hipMemcpy(gather ? in_dep : g_dep, dep.data(), rp * 4, hipMemcpyHostToDevice));
        CK(hipMemcpy(gather ? in_mv : g_mv, mv.data(), rp * 4, hipMemcpyHostToDevice));
        if (gather) {   // FUN_180101f60: srunit::GatherArgs, 88 bytes
            Args a(0x58);
            a.put(0x00, in_col).put(0x08, in_mv).put(0x10, in_dep);
            a.put(0x18, W * 8).put(0x1c, W * 4).put(0x20, W * 4);          // row pitches in bytes
            a.put(0x24, 0).put(0x28, 0).put(0x2c, 0);                       // format codes
            a.put(0x30, g_col).put(0x38, g_mv).put(0x40, g_dep).put(0x48, W).put(0x4c, H).put(0x50, 0);
            M.launch("_ZN6srunit11k_sr_gatherENS_10GatherArgsE", dim3((W + 15) >> 4, (H + 15) >> 4), dim3(16, 16), a);
        }
        const std::string pf(reinterpret_cast<const char*>(slurp(fdir + "/params" + sfx + ".json").data()), slurp(fdir + "/params" + sfx + ".json").size());
        auto jit = jget(pf, "jitter"), jitp = jget(pf, "jitter_prev"), mvs = jget(pf, "mv_scale");
        const float exposure = float(jget(pf, "exposure").at(0)), pre = float(jget(pf, "pre_exposure").at(0));
        const uint8_t reset = uint8_t(jget(pf, "reset").at(0)), depinv = uint8_t(jget(pf, "depth_inverted").at(0));
        const float mvoffx = 0, mvoffy = 0;   // mv_offset_kernel (FUN_18010ff70): 0 for unjittered MVs
        // F, the per-frame parameter block (the launchers read it by offset)
        // +0x48 auto_exposure=0, +0x49 ae_exact_sum=0, +0x4a depth_inverted, +0x4b reset, +0x4c ldr=0, +0x4d mvhi=0
        const float F20x = float(jit[0]), F20y = float(jit[1]), F28x = float(jitp[0]), F28y = float(jitp[1]);
        const float F30 = exposure, F34x = float(mvs[0]), F34y = float(mvs[1]), F44 = pre;

        // exposure (FUN_180116c90): no texture, fixed value
        { Args a(28); a.put(0, (void*)nullptr).put(8, F30).put(12, F44).put(16, expo).put(24, p);
          M.launch("_ZN2sr2io13k_sr_exposureEPKfffPti", dim3(1), dim3(32), a); }

        // enc0 (FUN_180116e70, preset M): Enc0Args, 280 B
        {
            const float fW = float(W), fH = float(H), f15w = float(W15), f15h = float(H15);
            Args a(280);
            a.put(0x00, g_col).put(0x08, g_dep).put(0x10, g_mv).put(0x18, prev_mvec[p]).put(0x20, dhist)
             .put(0x28, hidden[presetL ? 0 : p]).put(0x30, prev_mvec[p ^ 1]).put(0x38, mv_ero).put(0x40, mv_nondil)
             .put(0x48, t[0].d).put(0x50, enc0_w)
             .i2(0x58, W, H).i2(0x60, W, H).i2(0x68, W0, H0).i2(0x70, W15, H15).i2(0x78, hW15, hH15)
             .put(0x80, expo).put(0x88, p).f2(0x8c, F20x, F20y).f2(0x94, F28x, F28y)
             .f2(0x9c, fW / f15w, fH / f15h).put(0xa4, f15w / fW).put(0xa8, f15h / fH).put(0xac, 1.0f / fW)
             // (1, mv_scale.x) / (H, W/W15); mv_scale.y / (H/H15)
             .f2(0xb0, 1.0f / fH, F34x / (fW / f15w))
             .put(0xb8, F34y / (fH / f15h))
             .f2(0xbc, mvoffx, mvoffy)
             .f2(0xc4, 0.5f / float(hW15), 0.5f / float(hH15))
             .f2(0xcc, presetL ? 5.137702941894531f : 9.045065879821777f, presetL ? 0.08225896209478378f : 0.22356846928596497f)
             .put(0xd4, 0x40549a78u)
             .put(0xd8, uint32_t(reset)).put(0xdc, fc).put(0xe0, uint32_t(depinv)).put(0xe4, mvpitch)
             // (4884.335, 1, 1, jitter_prev.x) / (H, W15, H15, W); jitter_prev.y / H
             .f2(0xe8, (presetL ? 2774.359619140625f : 4884.33544921875f) / fH, 1.0f / f15w).f2(0xf0, 1.0f / f15h, F28x / fW)
             .put(0xf8, F28y / fH).put(0xfc, 0u).put(0x100, 0u).put(0x104, W).put(0x108, H)
             .f2(0x10c, float(oW) / fW, float(oH) / fH).put(0x114, 0u);
            if (!presetL) {
                M.launch("_ZN2sr2io9k_sr_enc0ENS0_8Enc0ArgsE", dim3(W0 / 8, H0 / 8), dim3(32), a);
            } else {
                Args b(288);
                std::memcpy(b.b.data(), a.b.data(), 280);
                b.put(280, l0_in);
                M.launch("_ZN2sr2io13k_sr_enc0L_inENS0_8Enc0ArgsEPh", dim3(((W0 + 8) * (H0 + 8) + 63) / 64), dim3(64), b);
                // L0Args, 104 B (FUN_180116e70's tail, FUN_1801189d0)
                Args l(104);
                l.put(0x00, enc0_w).put(0x08, l0_in).put(0x18, l0_skip).put(0x20, t[0].d)
                 .i2(0x48, W0, H0).i2(0x50, (p == 0) * 2, 2).i2(0x58, W, H);
                M.launch(enc0L_safe ? "_ZN2sr2io7k_sr_l0ILi0ELi2ELb0EEEvNS0_6L0ArgsE" : "_ZN2sr2io7k_sr_l0ILi0ELi2ELb1EEEvNS0_6L0ArgsE",
                         dim3((W0 + (p == 0) * 2 + 7) >> 3, (H0 + 2 + 7) >> 3), dim3(64), l);
            }
        }

        // ten stages (FUN_180117970 -> FUN_180118ca0), gfx12: the n = 2 instantiations, 64 threads
        for (int st = 0; st < 10; ++st) {
            Args a(56);
            a.put(0x00, wbuf + woff[st]);
            int dimW, dimH, s0, s1;
            const int pe = (p == 0) * 2, po = (p != 0) * 2;
            const char* sym = nullptr;
            if (st == 0) { a.put(0x08, t[0].d).put(0x18, t[1].d).put(0x20, t[2].d); dimW = L1w; dimH = L1h; s0 = pe; s1 = 2;
                sym = safe[0] ? "_ZN2sr10k_sr_stageILi64ELi2ELi0ELi96ELi2ELb0EEEvNS_9StageArgsE" : "_ZN2sr10k_sr_stageILi64ELi2ELi0ELi96ELi2ELb1EEEvNS_9StageArgsE"; }
            else if (st == 1) { a.put(0x08, t[2].d).put(0x18, t[3].d).put(0x20, t[4].d); dimW = L2w; dimH = L2h; s0 = pe; s1 = 2;
                sym = safe[1] ? "_ZN2sr10k_sr_stageILi96ELi4ELi0ELi128ELi2ELb0EEEvNS_9StageArgsE" : "_ZN2sr10k_sr_stageILi96ELi4ELi0ELi128ELi2ELb1EEEvNS_9StageArgsE"; }
            else if (st <= 7) {
                void* in = st == 2 ? t[4].d : (st & 1) ? t[5].d : t[6].d;
                void* o = (st & 1) ? t[6].d : t[5].d;
                a.put(0x08, in).put(0x18, o); dimW = L3w; dimH = L3h;
                if ((st & 1) == 0) { s0 = pe; s1 = 2; } else { s0 = po; s1 = 0; }
                sym = safe[st] ? "_ZN2sr15k_sr_tube_splitILi128ELi4ELb0EEEvNS_9StageArgsE" : "_ZN2sr15k_sr_tube_splitILi128ELi4ELb1EEEvNS_9StageArgsE";
            } else if (st == 8) { a.put(0x08, t[6].d).put(0x10, t[3].d).put(0x18, t[7].d); dimW = L2w; dimH = L2h; s0 = po; s1 = 0;
                sym = safe[8] ? "_ZN2sr10k_sr_stageILi96ELi4ELi2ELi128ELi2ELb0EEEvNS_9StageArgsE" : "_ZN2sr10k_sr_stageILi96ELi4ELi2ELi128ELi2ELb1EEEvNS_9StageArgsE"; }
            else { a.put(0x08, t[7].d).put(0x10, t[1].d).put(0x18, t[8].d); dimW = L1w; dimH = L1h; s0 = po; s1 = 0;
                sym = safe[9] ? "_ZN2sr10k_sr_stageILi64ELi2ELi2ELi96ELi2ELb0EEEvNS_9StageArgsE" : "_ZN2sr10k_sr_stageILi64ELi2ELi2ELi96ELi2ELb1EEEvNS_9StageArgsE"; }
            a.i2(0x28, dimW, dimH).i2(0x30, s0, s1);
            dim3 g = (st >= 2 && st <= 7) ? dim3((dimW + s0 + 3) >> 2, (dimH + s1 + 3) >> 2) : dim3((dimW + s0 + 7) >> 3, (dimH + s1 + 7) >> 3);
            M.launch(sym, g, dim3(64), a);
        }

        if (presetL) {
            // FUN_180117ab0's preset L path: L0Args into k_sr_l0<2,2,b> (FUN_18011a7b0)
            Args l(104);
            l.put(0x00, dec0_w).put(0x08, t[8].d).put(0x10, l0_skip).put(0x28, gates).put(0x30, factors)
             .put(0x38, aniso).put(0x40, hidden[0]).i2(0x48, W0, H0).i2(0x50, (p != 0) * 2, 0).i2(0x58, W, H)
             .put(0x60, dec0_s0).put(0x64, dec0_f0);
            M.launch(dec0L_safe ? "_ZN2sr2io7k_sr_l0ILi2ELi2ELb0EEEvNS0_6L0ArgsE" : "_ZN2sr2io7k_sr_l0ILi2ELi2ELb1EEEvNS0_6L0ArgsE",
                     dim3((W0 + (p != 0) * 2 + 7) >> 3, (H0 + 0 + 7) >> 3), dim3(64), l);
        } else
        // dec0 (FUN_180117ab0, preset M): Dec0Args, 160 B
        {
            const float fW = float(W), fH = float(H), f15w = float(W15), f15h = float(H15);
            Args a(160);
            a.put(0x00, t[8].d).put(0x08, prev_mvec[p ^ 1]).put(0x10, hidden[p]).put(0x18, gates).put(0x20, factors)
             .put(0x28, aniso).put(0x30, hidden[p ^ 1]).put(0x38, dec0_w)
             .i2(0x40, W, H).i2(0x48, W, H).i2(0x50, W0, H0).i2(0x58, W15, H15)
             .put(0x60, dec0_s0).put(0x64, dec0_f0).f2(0x68, F20x, F20y).f2(0x70, F28x, F28y)
             .f2(0x78, f15w / fW, f15h / fH).f2(0x80, 1.0f / fW, 1.0f / fH)
             .put(0x88, uint32_t(depinv))
             .f2(0x8c, 1.0f / f15w, 1.0f / f15h).f2(0x94, F28x / fW, F28y / fH).put(0x9c, 0u);
            M.launch("_ZN2sr2io9k_sr_dec0ENS0_8Dec0ArgsE", dim3(W0 / 8, H0 / 8), dim3(32), a);
        }

        // post (FUN_180117ee0): PostArgs, 256 B; <isL=false, 12/16 or 15/27, ldr=0, mvhi=0>
        {
            const float fW = float(W), fH = float(H), f15w = float(W15), f15h = float(H15);
            Args a(256);
            a.put(0x00, g_col).put(0x08, expo).put(0x10, uint64_t(p)).put(0x18, gates).put(0x20, factors)
             .put(0x28, aniso).put(0x30, ihist[p]).put(0x38, ihtex[p]).put(0x40, uint64_t(ihpitch >> 1))
             .put(0x48, prev_mvec[p ^ 1]).put(0x50, mv_ero).put(0x58, mv_nondil).put(0x60, mvntex)
             .put(0x68, uint64_t(mvpitch)).put(0x70, ihist[p ^ 1]).put(0x78, full15).put(0x80, dhist)
             .i2(0x88, W, H).i2(0x90, W, H).i2(0x98, W15, H15).i2(0xa0, hW15, hH15).put(0xa8, dec0_f1)
             .f2(0xac, fW / f15w, fH / f15h).f2(0xb4, F20x, F20y)
             .f2(0xbc, 1.0f / fW, 1.0f / fH).f2(0xc4, 1.0f / f15w, 1.0f / f15h).put(0xcc, 0x40549a78u)
             .put(0xd0, uint32_t(depinv)).put(0xd4, uint32_t(0)).put(0xd8, uint64_t(0)).put(0xe0, (void*)nullptr)
             .i2(0xe8, oW, oH).f2(0xf0, F34x / (float(oW) / f15w), F34y / (float(oH) / f15h)).f2(0xf8, mvoffx, mvoffy);
            const char* sym = presetL ? (post_3_2 ? "_ZN2sr2io9k_sr_postILb1ELi15ELi27ELb0ELb0EEEvNS0_8PostArgsE"
                                                  : "_ZN2sr2io9k_sr_postILb1ELi12ELi16ELb0ELb0EEEvNS0_8PostArgsE")
                                      : (post_3_2 ? "_ZN2sr2io9k_sr_postILb0ELi15ELi27ELb0ELb0EEEvNS0_8PostArgsE"
                                                  : "_ZN2sr2io9k_sr_postILb0ELi12ELi16ELb0ELb0EEEvNS0_8PostArgsE");
            M.launch(sym, dim3((W15 + 15) >> 4, (hH15 + 15) >> 4), dim3(256), a);
        }

        // downsample (FUN_180118240): DownArgs, 88 B
        {
            Args a(88);
            a.put(0x00, full15).put(0x08, g_col).put(0x10, expo).put(0x18, p).put(0x1c, 0u).put(0x20, out)
             .i2(0x28, W, H).i2(0x30, W15, H15).i2(0x38, oW, oH)
             .f2(0x40, float(W) / float((oW + 1) & ~1), float(H) / float((oH + 1) & ~1))
             .f2(0x48, F20x, F20y).put(0x50, 0x40549a78u).put(0x54, 0u);
            M.launch("_ZN2sr2io15k_sr_downsampleENS0_8DownArgsE", dim3((oW + 15) >> 4, (oH + 15) >> 4), dim3(256), a);
        }
        CK(hipDeviceSynchronize());
        CK(hipMemcpy(hostout.data(), out, hostout.size(), hipMemcpyDeviceToHost));
        spit(odir + "/out" + sfx + ".f32", hostout.data(), hostout.size());
        if (dump) {
            std::vector<uint8_t> h(size_t(W15) * H15 * 8);
            CK(hipMemcpy2D(h.data(), size_t(W15) * 8, ihist[p ^ 1], ihpitch, size_t(W15) * 8, H15, hipMemcpyDeviceToHost));
            spit(odir + "/hist" + sfx + ".f16", h.data(), h.size());
            std::vector<uint8_t> g(size_t(W15) * H15 * 4);
            CK(hipMemcpy(g.data(), full15, g.size(), hipMemcpyDeviceToHost));
            spit(odir + "/full15" + sfx + ".bin", g.data(), g.size());
        }
        std::printf("frame %d done (parity %d)\n", fi, p);
        p ^= 1; fc = (fc + 1) & 7;
    }
    if (dump) {
        auto dumpbuf = [&](const char* n, void* d, size_t bytes) {
            std::vector<uint8_t> h(bytes); CK(hipMemcpy(h.data(), d, bytes, hipMemcpyDeviceToHost));
            spit(odir + "/" + n, h.data(), bytes);
        };
        dumpbuf("gates.bin", gates, rp * 4); dumpbuf("factors.bin", factors, rp * 4); dumpbuf("aniso.bin", aniso, rp * 8);
        dumpbuf("hidden0.bin", hidden[0], rp * 4); dumpbuf("hidden1.bin", hidden[1], rp * 4);
        dumpbuf("mv_ero.bin", mv_ero, rp * 4); dumpbuf("prev_mvec0.bin", prev_mvec[0], rp * 4); dumpbuf("prev_mvec1.bin", prev_mvec[1], rp * 4);
        dumpbuf("dhist.bin", dhist, size_t(hW15) * hH15 * 4); dumpbuf("expo.bin", expo, 4);
        for (int i = 0; i < 9; ++i) {
            std::vector<uint8_t> h(size_t(t[i].C) * t[i].W * t[i].H);
            CK(hipMemcpy(h.data(), t[i].d, h.size(), hipMemcpyDeviceToHost));
            spit(odir + "/t" + std::to_string(i) + ".fp8", h.data(), h.size());
        }
    }
    return 0;
}
