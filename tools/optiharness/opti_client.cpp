// opti_client: a Windows program (mingw, run under Proton) that drives frames THROUGH OptiScaler
// (the game's dxgi.dll middle layer, v0.9.4) into this repo's _nvngx.dll Vulkan DLSS port, times
// the evaluate, and reads its output back to prove the upscaled frame is real (not black).
//
//   opti_client.exe [--opti | --direct] [--nvngx NAME] [--frames N] [--out DIR] [--csv FILE]
//                   [--w W] [--h H] [--ow W] [--oh H] [--mv-fmt rg32|rg16]
//                   [--dep-fmt d32s8|r32f] [--corpus DIR] [--expect-fail] [--sl] [--sl-rr]
//
// --opti (default) resolves the NGX exports from the OptiScaler proxy dxgi.dll; --direct skips
// OptiScaler and GetProcAddress-es the same exports straight from _nvngx.dll.  Identical inputs
// and identical timing grammar, so (direct vs opti) isolates OptiScaler's overhead.
//
// Line grammar this program prints (consumers: tools/optiharness/run_harness.sh):
//   EVAL_MS <i> <ms>              QPC wall time around the EvaluateFeature call (whole stack)
//   GPU_EVAL_MS <i> <ms>         D3D12 timestamp queries bracketing that same call on our own
//                                 command list (or GPU_EVAL_MS <i> n/a when unsupported)
//   OUT_NZ <i> nz=<f> min= max=   per-frame non-black statistics of the read-back output
//   SL-RR-ENABLE rc=0x<x> ctx=... (--sl-rr ONLY) the RR-enable marker run_harness.sh greps as its
//                                 optional g5 subcheck c6 (implies --sl; no other line moves)
//   HARNESS SUMMARY               medians/p95 over frames 1..N, PASS/FAIL per gate, RESULT line
// This program does NOT print GPU_FRAME_MS: that name belongs to the port's runner (runner.cpp:340,
// brackets chain->record() alone; consumer eval_verify.py:115 reads frames 0..N-1 from it), and
// nothing observable from here brackets that scope - emitting a look-alike would poison the frozen
// pipeline's consumers.  The port's own host-side "-recorded in N ms" line is only PARSED at exit.
//
// Provenance: the D3D12 idioms (device/queue/fence/allocator/list boilerplate, upload + transition,
// readback + widen f16->f32) follow tools/verify/ngx_client/main.cpp's D3D12 path, which is included
// READ-ONLY through ngx_abi.h (-I tools/verify/ngx_client; the frozen tree is never edited).  This
// file lives in tools/optiharness/, a new directory outside the "verify: freeze v2" scope, and it
// carries its own headers: the timing pairs and the synthetic/corpus inputs below are net-new.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ngx_abi.h"
// Vendored Streamline v2.14.1 (the game's host SDK), quote-included so it resolves next to this
// source file; sl-include/VENDORED.txt records url/tag/commit/date/licence for all nine files.
#include "sl-include/sl.h"

// NVIDIA Streamline, vendored from the EXACT release the game boots: its archive says "Streamline
// v2.14.1.98614dad6 ... host SDK v2.14.1" (runs/OptiScaler.slayer3-noeval.log-20261030:277), and
// sl-include/ holds tag v2.14.1 verbatim (sl-include/VENDORED.txt = url/commit/date + license
// notes).  sl_abi_check.cpp proves the layouts by hand; these build-time asserts re-prove them on
// every compile so a swapped header cannot silently hand slInit a wrong-offset Preferences:
//   sl_struct.h: BaseStructure{next(8), structType(16), structVersion(8)} = 32
//   sl_core_types.h: Preferences = base 32 + fields ... RenderAPI @136 -> sizeof 144
//   sl_version.h: kSDKVersion = (2<<48)|(14<<32)|(1<<16)|0xfedc = host SDK v2.14.1
static_assert(sizeof(sl::BaseStructure) == 32, "sl_struct.h layout");
static_assert(sizeof(sl::Preferences) == 144, "slInit parameter block (sl_core_types.h)");
static_assert(offsetof(sl::Preferences, logMessageCallback) == 80, "the [sl-cb] slot");
static_assert(sl::kSDKVersion == ((uint64_t(2) << 48) | (uint64_t(14) << 32) | (uint64_t(1) << 16) | 0xfedc),
              "the game's host SDK 2.14.1");

// M2 of ngx_client's contract: 1 = Output RGBA16F (matches DLSS5_AMD_OUT_F16=1 in the package's
// _nvngx.dll), 0 = old RGBA32F.  build.sh passes -DFMT16=${FMT16:-1} like ngx_client/build.sh:17.
#ifndef FMT16
#define FMT16 1
#endif

// Message plumbing: under `proton run` a PE's stdout/stderr do NOT reach the shell's redirect
// (measured 2026-10-30: the frozen run_all.sh V9 logs, build/verify/run_*/v9_*.log, carry only
// Proton's own lines - and rg over them finds no [ngx_client] text), so every line this program
// prints is ALSO mirrored to <out>/client.log, which is where run_harness.sh parses from.  On a
// terminal both fire; nothing is ever hidden.  g_msg is opened as soon as --out is known.
static FILE* g_msg = nullptr;
static void vlog_to(FILE* f, const char* pre, const char* fmt, va_list ap) {
    if (!f) return;
    char buf[4096];
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    size_t len = std::strlen(buf);                 // call sites keep printf-style trailing "\n"s
    while (len && buf[len - 1] == '\n') buf[--len] = 0;
    std::fprintf(f, "%s%s\n", pre, buf);
    std::fflush(f);
}
static void say(const char* fmt, ...) {           // stdout + client.log
    va_list ap; va_start(ap, fmt);
    vlog_to(stdout, "", fmt, ap);
    va_end(ap);
    va_start(ap, fmt);
    vlog_to(g_msg, "", fmt, ap);
    va_end(ap);
}
static void sayerr(const char* fmt, ...) {        // stderr (rarely visible) + client.log
    va_list ap; va_start(ap, fmt);
    vlog_to(stderr, "", fmt, ap);
    va_end(ap);
    va_start(ap, fmt);
    vlog_to(g_msg, "", fmt, ap);
    va_end(ap);
}

#define FAIL(...) do { sayerr("[opti_client] FAIL: " __VA_ARGS__); return 1; } while (0)
#define HR(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { sayerr("[opti_client] FAIL: %s:%d %s = 0x%08lx", __FILE__, __LINE__, #x, (unsigned long)hr_); return 1; } } while (0)

// The in-game resource enums this harness reproduces end to end (nvngx.cpp:959/963 pins the same
// numbers as the port prints them): MotionVectors R32G32_FLOAT = 16, Depth R32G8X24_TYPELESS = 19.
static_assert(DXGI_FORMAT_R32G32_FLOAT == 16, "the in-game MotionVectors DXGI format");
static_assert(DXGI_FORMAT_R32G8X24_TYPELESS == 19, "the in-game Depth DXGI format (issue 1)");
static_assert(DXGI_FORMAT_R16G16_FLOAT == 34, "the harness/ngx_client MotionVectors format");
static_assert(DXGI_FORMAT_R32_FLOAT == 41, "the harness/ngx_client Depth format");
// N4 (2026-10-31): the port's RR-only slots: GBuffer.Normals / DLSS.Input.DiffuseAlbedo /
// DLSS.Input.SpecularAlbedo are RGBA16F, GBuffer.Roughness is R32F - the DXGI enums the port
// decodes into VK_FORMAT_R16G16B16A16_SFLOAT / VK_FORMAT_R32_SFLOAT (src/pe/nvngx.cpp:1132-1136,
// want_format at :1074; the frozen referee lists the same DXGI pair, ngx_client/main.cpp:95-98).
static_assert(DXGI_FORMAT_R16G16B16A16_FLOAT == 10, "the RR GBuffer/albedo DXGI format (N4)");

// ------------------------------------------------------------------------------------------------
// small helpers
// ------------------------------------------------------------------------------------------------
static inline float h2f_(uint16_t h) {   // half -> float (same widening ngx_client/main.cpp:35 uses)
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 0x3FFu;
    float v;
    if (e == 0u) v = (float)m * 5.9604644775390625e-08f;
    else if (e == 31u) { uint32_t b = s | 0x7F800000u | (m << 13); memcpy(&v, &b, 4); return v; }
    else { uint32_t b = s | ((e + 112u) << 23) | (m << 13); memcpy(&v, &b, 4); }
    return (h & 0x8000u) ? -v : v;
}

static inline uint16_t f2h_(float f) {   // float -> half, round-to-nearest-even; inputs are 0..~4
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int be = int((x >> 23) & 0xFF);
    uint32_t m = x & 0x7FFFFFu;
    if (be == 0xFF) return uint16_t(sign | 0x7C00u | (m ? 0x200u : 0u));   // inf/nan
    const int e = be - 127 + 15;
    if (e >= 0x1F) return uint16_t(sign | 0x7C00u);                          // overflow -> inf
    if (e <= 0) {                                                            // subnormal / underflow
        if (e < -10) return uint16_t(sign);
        m |= 0x800000u;
        const uint32_t sh = uint32_t(14 - e);
        uint32_t r = m >> sh;
        if ((m >> (sh - 1)) & 1u) ++r;
        return uint16_t(sign | r);
    }
    uint32_t r = uint32_t(e << 10) | (m >> 13);
    if (m & 0x1000u) {                                                        // round to nearest even
        if ((m & 0x0FFFu) != 0x0800u || (r & 1u)) ++r;
    }
    return uint16_t(sign | r);
}

static double median_(std::vector<double> v) {
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double p95_(std::vector<double> v) {   // nearest-rank p95
    if (v.empty()) return -1.0;
    std::sort(v.begin(), v.end());
    size_t k = (size_t)std::ceil(0.95 * double(v.size()));
    if (k == 0) k = 1;
    if (k > v.size()) k = v.size();
    return v[k - 1];
}
static double min_of(const std::vector<double>& v) { return v.empty() ? -1.0 : *std::min_element(v.begin(), v.end()); }
static double max_of(const std::vector<double>& v) { return v.empty() ? -1.0 : *std::max_element(v.begin(), v.end()); }

// ------------------------------------------------------------------------------------------------
// options
// ------------------------------------------------------------------------------------------------
struct Opt {
    bool direct = false;
    std::string nvngx;                 // which port DLL; run_harness stages it as <harness>/_nvngx.dll
    int frames = 12;
    std::string odir, csv, corpus;
    unsigned w = 1920, h = 1080, ow = 3840, oh = 2160;
    std::string mv_fmt = "rg32", dep_fmt = "d32s8";
    bool expect_fail = false;
    bool sl = false;                   // --sl: boot the game's Streamline stack before the Opti flow
    bool sl_rr = false;                // --sl-rr (implies --sl): REQUEST kFeatureDLSS_RR after slInit +
                                       // slSetD3DDevice - the in-game RR toggle --sl never made (the
                                       // 0xbad00000 chain, dlss_dEntry.cpp:1237 -> "not loaded")
};

static int usage() {
    sayerr(
        "[opti_client] usage: opti_client.exe [--opti | --direct] [--nvngx NAME] [--frames N]\n"
        "    [--out DIR] [--csv FILE] [--w W] [--h H] [--ow W] [--oh H]\n"
        "    [--mv-fmt rg32|rg16] [--dep-fmt d32s8|r32f] [--corpus DIR] [--expect-fail]\n"
        "    [--sl] [--sl-rr]  (--sl-rr implies --sl: request kFeatureDLSS_RR and print\n"
        "                      [opti_client] SL-RR-ENABLE rc=0x<x> ctx=<state> for run_harness g5 c6)");
    return 1;
}

// ------------------------------------------------------------------------------------------------
// Streamline boot surface (only under --sl): the game's sl.interposer.dll + the 10 sl.* plugins,
// booted headless so the exact in-game SL log lines are reproducible in ~90 s runs.
// ------------------------------------------------------------------------------------------------
// The logMessageCallback SL calls back into us with (sl_core_types.h: "using
// PFun_LogMessageCallback = void(LogType type, const char* msg)"): every game-identical
// [streamline] line lands here as [sl-cb], even when OptiScaler's own StreamlineHooks echo is
// absent - whichever registers first is fine (brief), this one is ours and unconditional.
static void sl_log_cb(sl::LogType type, const char* msg) {
    const char* tn = type == sl::LogType::eWarn ? "warn" : type == sl::LogType::eError ? "error" : "info";
    say("[sl-cb][%s] %s", tn, msg ? msg : "");
}

static const char* sl_res_name(sl::Result r) {
    switch (r) {
        case sl::Result::eOk: return "eOk";
        case sl::Result::eErrorDriverOutOfDate: return "eErrorDriverOutOfDate";
        case sl::Result::eErrorNoPlugins: return "eErrorNoPlugins";
        case sl::Result::eErrorNVAPI: return "eErrorNVAPI";
        case sl::Result::eErrorNGXFailed: return "eErrorNGXFailed";
        case sl::Result::eErrorNotInitialized: return "eErrorNotInitialized";
        case sl::Result::eErrorFeatureNotSupported: return "eErrorFeatureNotSupported";
        case sl::Result::eErrorFeatureMissing: return "eErrorFeatureMissing";
        case sl::Result::eErrorFeatureFailedToLoad: return "eErrorFeatureFailedToLoad";
        case sl::Result::eErrorInvalidState: return "eErrorInvalidState";
        default: return "?";
    }
}

// Feature IDs the game asked for, mirrored from the archive what-Was-NOT-requested lines: its
// `Ignoring plugin 'sl.deepdvc'/'sl.directsr'/'sl.nis'/'sl.nvperf' since it was not requested`
// (slayer3 lines 370/377/404/407) + the Loaded lines (sl.dlss id0, sl.reflex id3, sl.pcl id4,
// sl.dlss_g id1000, sl.dlss_d-id1001=kFeatureDLSS_RR; sl.common id 4294967295 = kFeatureCommon
// is always loaded and never requested).
static constexpr sl::Feature kGameFeatures[] = {
    sl::kFeatureDLSS, sl::kFeatureReflex, sl::kFeaturePCL, sl::kFeatureDLSS_G, sl::kFeatureDLSS_RR};

// sl::Preferences holds raw pointers into THESE process-lifetime buffers (SL keeps a copy of the
// struct, not necessarily of everything it points at - keep both alive to slShutdown).
static std::wstring g_sl_plugin_dir;
static const wchar_t* g_sl_path_ptrs[1] = {nullptr};

// Resolved sl.interposer.dll exports (export table verified with objdump -p on the game's
// sl.interposer.dll: slInit/slShutdown/slSetD3DDevice/slGetFeatureRequirements/
// slIsFeatureSupported/slIsFeatureLoaded all present, ordinal-named).
struct SlApi {
    HMODULE mod = nullptr;
    sl::Result (*init)(const sl::Preferences&, const uint64_t) = nullptr;
    sl::Result (*shutdown)() = nullptr;
    sl::Result (*setDevice)(void*) = nullptr;
    sl::Result (*getReqs)(sl::Feature, sl::FeatureRequirements&) = nullptr;
    sl::Result (*isSupported)(sl::Feature, const sl::AdapterInfo&) = nullptr;
    sl::Result (*isLoaded)(sl::Feature, bool&) = nullptr;
    // --sl-rr ONLY, resolved lazily there so a plain --sl run still resolves exactly the six above
    // (and prints the same "6 exports resolved" line).  Types are sl_core_api.h's own PFun_*
    // aliases (:54/:55/:70) - the exact vendored signatures, so a swapped header cannot hand this
    // file an unnoticed ABI drift; the names are also all present in the game's sl.interposer.dll
    // export table (objdump -p: slSetFeatureLoaded/slGetNewFrameToken/slEvaluateFeature).
    PFun_slSetFeatureLoaded* setLoaded = nullptr;
    PFun_slEvaluateFeature* evalFeat = nullptr;
    PFun_slGetNewFrameToken* newFrameTok = nullptr;
    bool ok() const { return mod && init && shutdown && setDevice && getReqs && isSupported && isLoaded; }
};

static std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }   // Wine Z:\ paths are ASCII

// ------------------------------------------------------------------------------------------------
// tiny JSON number reader for the corpus mode (frame.json / params_<i>.json of the V9 contract,
// tools/verify/README.md:61-75) - the keys ngx_client reads there, numbers and booleans only
// ------------------------------------------------------------------------------------------------
static bool jnum_i(const std::string& txt, const char* key, int idx, double* out) {
    size_t k = txt.find(std::string("\"") + key + "\"");
    if (k == std::string::npos) return false;
    size_t i = txt.find(':', k);
    if (i == std::string::npos) return false;
    ++i;
    while (i < txt.size() && (txt[i] == ' ' || txt[i] == '[' || txt[i] == '\n' || txt[i] == '\t')) ++i;
    if (txt.compare(i, 4, "true") == 0 || txt.compare(i, 5, "false") == 0) {
        if (idx != 0) return false;
        *out = (txt[i] == 't') ? 1.0 : 0.0;
        return true;
    }
    for (int e = 0; e < idx; ++e) {
        while (i < txt.size() && txt[i] != ',' && txt[i] != ']') ++i;
        if (i >= txt.size() || txt[i] == ']') return false;
        ++i;
        while (i < txt.size() && (txt[i] == ' ' || txt[i] == '\n')) ++i;
    }
    char* end = nullptr;
    double d = std::strtod(txt.c_str() + i, &end);
    if (end == txt.c_str() + i) return false;
    *out = d;
    return true;
}
static double jnum_d(const std::string& txt, const char* key, double dflt) {
    double d;
    return jnum_i(txt, key, 0, &d) ? d : dflt;
}
static bool read_file(const std::string& p, std::vector<uint8_t>& out) {
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(n > 0 ? size_t(n) : 0);
    bool ok = out.empty() || std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    return ok;
}
static bool write_file(const std::string& p, const void* d, size_t n) {
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) return false;
    bool ok = std::fwrite(d, 1, n, f) == n;
    std::fclose(f);
    return ok;
}

// ------------------------------------------------------------------------------------------------
// D3D12 helpers - idioms from ngx_client/main.cpp:192-219 (frozen, included read-only)
// ------------------------------------------------------------------------------------------------
static ID3D12Resource* d3d_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE t, D3D12_RESOURCE_STATES st) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size; d.Height = 1; d.DepthOrArraySize = 1; d.MipLevels = 1; d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = t;
    ID3D12Resource* r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}
static void d3d_barrier(ID3D12GraphicsCommandList* c, ID3D12Resource* r, D3D12_RESOURCE_STATES a, D3D12_RESOURCE_STATES b) {
    D3D12_RESOURCE_BARRIER br{};
    br.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    br.Transition.pResource = r; br.Transition.StateBefore = a; br.Transition.StateAfter = b;
    br.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    c->ResourceBarrier(1, &br);
}
static UINT pitch256(UINT bytes) { return (bytes + 255u) & ~255u; }

// One timestamp write in the D3D12 style: ID3D12GraphicsCommandList::EndQuery(heap, Type, Index)
// - confirmed against the MIDL-generated SDK header (agilitysdk/include/d3d12.h:6461: EndQuery
// takes the QUERY INDEX, not a byte offset; the tempting reading "End(pHeap, alignedByteOffset)"
// from armchair recall is what produced a first draft that wrote index 0 twice and left every
// resolved pair unwritten - T_IRAW garbage of about 2^63/freq per frame) - so the pair written
// here is exactly the pair ResolveQueryData reads back as [2i, 2i+2).
static inline void ts_end(ID3D12GraphicsCommandList* cl, ID3D12QueryHeap* qh, UINT query_index) {
    cl->EndQuery(qh, D3D12_QUERY_TYPE_TIMESTAMP, query_index);
}

// ------------------------------------------------------------------------------------------------
// synthetic inputs: deterministic, generated in-process (no file dependencies unless --corpus)
// ------------------------------------------------------------------------------------------------
// Colour: a moving gradient + a moving 16 px checker, every channel strictly non-zero, so a black
// output can only mean the evaluate produced nothing - this is the non-black gate's justification
// (the threshold is a conservative >= 0.10 while the healthy expectation here is ~1.00).
static void synth_color(void* dst, UINT pitch, unsigned w, unsigned h, int frame) {
    for (unsigned y = 0; y < h; ++y) {
        uint16_t* px = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(dst) + size_t(y) * pitch);
        for (unsigned x = 0; x < w; ++x) {
            const float gx = float((x + 6u * unsigned(frame)) % w) / float(w);
            const float gy = float(y) / float(h);
            const bool chk = (((x >> 4) ^ (y >> 4) ^ (unsigned(frame) >> 1)) & 1u) != 0;
            const float r = 0.25f + 0.50f * gx;
            const float g = 0.25f + 0.50f * gy;
            const float b = chk ? 0.875f : 0.375f;
            const float a = 1.0f;
            px[x * 4 + 0] = f2h_(r); px[x * 4 + 1] = f2h_(g);
            px[x * 4 + 2] = f2h_(b); px[x * 4 + 3] = f2h_(a);
        }
    }
}
// Motion vectors: constant pan +2 px in x, +0.25 px in y (NGX convention: previous = current + mv).
static void synth_mv(void* dst, UINT pitch, unsigned w, unsigned h, bool rg32) {
    for (unsigned y = 0; y < h; ++y) {
        uint8_t* row = static_cast<uint8_t*>(dst) + size_t(y) * pitch;
        for (unsigned x = 0; x < w; ++x) {
            if (rg32) { float* f = reinterpret_cast<float*>(row) + x * 2; f[0] = 2.0f; f[1] = 0.25f; }
            else { uint16_t* q = reinterpret_cast<uint16_t*>(row) + x * 2; q[0] = f2h_(2.0f); q[1] = f2h_(0.25f); }
        }
    }
}
// Depth (the R32F bisect fallback): a plane over [0,1] in y.  The d32s8 variant cannot be filled
// per-pixel: D3D12 offers no COPY_DEST for a depth-stencil resource, so it takes a constant
// ClearDepthStencilView(0.5) at creation - valid floats, all the non-black gate needs (README).
static void synth_depth(void* dst, UINT pitch, unsigned w, unsigned h) {
    for (unsigned y = 0; y < h; ++y) {
        float* row = reinterpret_cast<float*>(static_cast<uint8_t*>(dst) + size_t(y) * pitch);
        const float d = float(y) / float(h ? h : 1);
        for (unsigned x = 0; x < w; ++x) row[x] = d;
    }
}

// ---- N4 (2026-10-31): the RR (feature 13) inputs ------------------------------------------------
// OptiScaler's dlssd backend makes the port gather FOUR extra textures per evaluate: collect_frame
// fetches ALL of extra[] for a RayReconstruction feature (src/pe/nvngx.cpp:1154-1161) - the two the
// need names plus DLSS.Input.DiffuseAlbedo / DLSS.Input.SpecularAlbedo (nvngx.cpp:1132-1136).  Bind
// only the two and every eval just moves its refusal down the list, so the harness owns all four.
// Formats are what want_format pins (nvngx.cpp:1133-1136) and what the frozen referee feeds
// (ngx_client/main.cpp:91-98): Normals/Albedo/Specular RGBA16F (8 B/px), Roughness R32F (4 B/px) -
// exactly the chain's default gather codes fmt_nrm/fmt_alb/fmt_spa = 0, fmt_rgh = 3
// (src/vk/pipeline.h:36-39, kRrBpp sizes at pipeline.cpp:2250, rr_gather.comp:16).
// Content is synthetic and FRAME-INDEPENDENT (like synth_mv/synth_depth: no rand, zero run-to-run
// input variance; run_harness/staging never sets DLSS5_AMD_SKIP_INPUTS so nothing here is skipped):
static void synth_normals(void* dst, UINT pitch, unsigned w, unsigned h) {
    // GBuffer.Normals (RGBA16F): a benign hemisphere sweep of unit normals
    // n = normalize(2*uv - 1, 1) - centre faces +z, corners lean outward (|n| = 1 everywhere),
    // w = 0 (roughness-in-normals.w mode is off: roughness travels in GBuffer.Roughness).
    for (unsigned y = 0; y < h; ++y) {
        uint16_t* px = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(dst) + size_t(y) * pitch);
        const float gy = (h > 1) ? (2.0f * float(y) / float(h - 1) - 1.0f) : 0.0f;
        for (unsigned x = 0; x < w; ++x) {
            const float gx = (w > 1) ? (2.0f * float(x) / float(w - 1) - 1.0f) : 0.0f;
            const float inv = 1.0f / std::sqrt(gx * gx + gy * gy + 1.0f);
            px[x * 4 + 0] = f2h_(gx * inv); px[x * 4 + 1] = f2h_(gy * inv);
            px[x * 4 + 2] = f2h_(inv);       px[x * 4 + 3] = f2h_(0.0f);
        }
    }
}
// DiffuseAlbedo (RGBA16F): constant mid-grey g, a = 1 - used for BOTH albedo slots, only the grey
// differs: diffuse = 0.5, specular = 0.04 (the usual dielectric F0), both plain [0,1] reflectances.
static void synth_albedo(void* dst, UINT pitch, unsigned w, unsigned h, float g) {
    for (unsigned y = 0; y < h; ++y) {
        uint16_t* px = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(dst) + size_t(y) * pitch);
        for (unsigned x = 0; x < w; ++x) {
            px[x * 4 + 0] = f2h_(g); px[x * 4 + 1] = f2h_(g);
            px[x * 4 + 2] = f2h_(g); px[x * 4 + 3] = f2h_(1.0f);
        }
    }
}
// GBuffer.Roughness (R32F): constant mid-grey 0.5, one float per texel.
static void synth_rough(void* dst, UINT pitch, unsigned w, unsigned h) {
    for (unsigned y = 0; y < h; ++y) {
        float* row = reinterpret_cast<float*>(static_cast<uint8_t*>(dst) + size_t(y) * pitch);
        for (unsigned x = 0; x < w; ++x) row[x] = 0.5f;
    }
}
// UpSlot.kind -> filler: 3 normals, 4 diffuse albedo, 5 specular albedo, 6 roughness.
static void synth_rr(unsigned kind, void* dst, UINT pitch, unsigned w, unsigned h) {
    if (kind == 3) synth_normals(dst, pitch, w, h);
    else if (kind == 4) synth_albedo(dst, pitch, w, h, 0.5f);
    else if (kind == 5) synth_albedo(dst, pitch, w, h, 0.04f);
    else synth_rough(dst, pitch, w, h);
}
// Small label for the per-eval RR slot dump (the DXGI enum the port's GetDesc sees).
static const char* rr_slot_fmt(unsigned kind) {
    return kind == 6 ? "R32_SFLOAT(DXGI 41)" : "R16G16B16A16_SFLOAT(DXGI 10)";
}

// Is this run an RR run?  The client's CreateFeature always says NVSDK_NGX_Feature_SuperSampling;
// OptiScaler maps that onto the backend named by the staged [Upscalers] Dx12Upscaler - it reads the
// ini NEXT TO its own dxgi.dll (Config.cpp:45) = our exe dir - and, for dlssd, rewrites the
// port-side call to NVSDK_NGX_Feature_RayReconstruction = 13 (DLSSDFeature_Dx12.cpp:52), which is
// also why the port-side companion check wants nvngx_dlssd.dll (nvngx.cpp:187-189).  There is no
// query for that rewrite on the export surface this client resolves, and OptiScaler's `auto` never
// picks dlssd (Config.cpp:1561-1562), so the ini value the run actually loads IS the pre-evaluate
// RR signal.  Anything else (no file / --direct / --corpus input) = old behaviour, unchanged line
// for line.
static bool ini_declares_dlssd(const std::string& ini_path) {
    std::FILE* f = std::fopen(ini_path.c_str(), "rb");
    if (!f) return false;
    auto ci_eq = [](const char* a, const char* b) {          // case-insensitive, end-anchored
        while (*a && *b) {
            if (std::tolower((unsigned char)*a) != std::tolower((unsigned char)*b)) return false;
            ++a; ++b;
        }
        return !*a && !*b;
    };
    auto trim = [](char* s) {                                // trim trailing space/CR/LF in place
        char* e = s + std::strlen(s);
        while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r' || e[-1] == '\n')) *--e = 0;
        return s;
    };
    char line[512], section[128] = "";
    bool rr = false;
    while (std::fgets(line, sizeof line, f)) {
        char* s = line;
        while (*s == ' ' || *s == '\t') ++s;
        if (*s == ';' || *s == '#' || *s == '\r' || *s == '\n' || *s == '\0') continue;
        if (*s == '[') {
            char* e = std::strchr(s, ']');
            if (!e) continue;
            *e = 0;
            std::snprintf(section, sizeof section, "%s", trim(s + 1));
            continue;
        }
        char* eq = std::strchr(s, '=');
        if (!eq) continue;
        *eq = 0;
        if (ci_eq(trim(s), "Dx12Upscaler") && ci_eq(section, "Upscalers")) {
            char* v = eq + 1;
            while (*v == ' ' || *v == '\t') ++v;
            rr = ci_eq(trim(v), "dlssd");
            break;
        }
    }
    std::fclose(f);
    return rr;
}

// ------------------------------------------------------------------------------------------------
// the port log dlss5-amd-ngx.log (CWD) at exit: "recorded in N ms" median + eval OK/FAIL counts
// ------------------------------------------------------------------------------------------------
struct PortLog { bool present = false; double rec_median = -1.0; int rec_n = 0, ok = 0, fail = 0; };
static PortLog read_port_log() {
    PortLog pl;
    std::FILE* f = std::fopen("dlss5-amd-ngx.log", "r");
    if (!f) return pl;
    pl.present = true;
    char line[2048];
    std::vector<double> rec;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strstr(line, " OK: id=")) ++pl.ok;
        else if (std::strstr(line, "FAIL: id=")) ++pl.fail;
        if (const char* r = std::strstr(line, "recorded in ")) {
            // digit right after "in " - zeros count too: the port stamps with GetTickCount64
            // (webpack: ngx_ms), so a sub-millisecond record honestly reads "recorded in 0 ms".
            if (std::strchr("0123456789", r[12])) rec.push_back(std::atof(r + 12));
        }
    }
    std::fclose(f);
    pl.rec_n = int(rec.size());
    pl.rec_median = median_(rec);
    return pl;
}

struct FrameRec {
    int i = 0;
    double eval_ms = -1.0, gpu_ms = -1.0, nz = -1.0, mn = 0.0, mx = 0.0;
    bool eval_ok = false, gpu_valid = false;
};

int main(int argc, char** argv) {
    Opt o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--opti") o.direct = false;
        else if (a == "--direct") o.direct = true;
        else if (a == "--nvngx") o.nvngx = next();
        else if (a == "--frames") o.frames = std::atoi(next().c_str());
        else if (a == "--out") o.odir = next();
        else if (a == "--csv") o.csv = next();
        else if (a == "--w") o.w = unsigned(std::atoi(next().c_str()));
        else if (a == "--h") o.h = unsigned(std::atoi(next().c_str()));
        else if (a == "--ow") o.ow = unsigned(std::atoi(next().c_str()));
        else if (a == "--oh") o.oh = unsigned(std::atoi(next().c_str()));
        else if (a == "--mv-fmt") o.mv_fmt = next();
        else if (a == "--dep-fmt") o.dep_fmt = next();
        else if (a == "--corpus") o.corpus = next();
        else if (a == "--expect-fail") o.expect_fail = true;
        else if (a == "--sl") o.sl = true;
        else if (a == "--sl-rr") { o.sl_rr = true; o.sl = true; }   // implies --sl either way
        else { usage(); return 1; }   // an unknown arg exits 1: Proton probes rely on exactly that
    }
    if (o.frames < 1 || o.frames > 256) FAIL("--frames 1..256");
    if (o.odir.empty()) return usage();
    if (o.mv_fmt != "rg32" && o.mv_fmt != "rg16") FAIL("--mv-fmt rg32|rg16");
    if (o.dep_fmt != "d32s8" && o.dep_fmt != "r32f") FAIL("--dep-fmt d32s8|r32f");

    // Open the message mirror before the first line so client.log starts with the mode header
    // ("wb": binary, so the CRT's text mode cannot turn \n into \r\n and break the parsers).
    CreateDirectoryA(o.odir.c_str(), nullptr);
    g_msg = std::fopen((o.odir + "/client.log").c_str(), "wb");

    // ---- Streamline: interposer FIRST (before our dxgi/D3D12 activity, the ordering the brief
    // pins: the game loads it at startup and it injects dxgi hooks; Opti's staged dxgi.dll has
    // already installed its LdrLoadDll hooks from process start if the loader pulled it in).
    SlApi slapi;
    // (--sl-rr) the two upstream SL results, recorded so the RR-enable block below can honour
    // sl_core_api.h's "requires ... device to be created before calling it" (:117/:133); both stay
    // at eErrorNotInitialized when --sl is absent, and --sl-rr implies --sl by construction.
    sl::Result sl_init_r = sl::Result::eErrorNotInitialized;
    sl::Result sl_dev_r = sl::Result::eErrorNotInitialized;
    if (o.sl) {
        slapi.mod = LoadLibraryA("sl.interposer.dll");
        if (!slapi.mod) FAIL("cannot LoadLibrary sl.interposer.dll (run_harness.sh stages it next to the exe)");
        struct { const char* n; void** p; } want[] = {
            {"slInit", reinterpret_cast<void**>(&slapi.init)},
            {"slShutdown", reinterpret_cast<void**>(&slapi.shutdown)},
            {"slSetD3DDevice", reinterpret_cast<void**>(&slapi.setDevice)},
            {"slGetFeatureRequirements", reinterpret_cast<void**>(&slapi.getReqs)},
            {"slIsFeatureSupported", reinterpret_cast<void**>(&slapi.isSupported)},
            {"slIsFeatureLoaded", reinterpret_cast<void**>(&slapi.isLoaded)},
        };
        for (auto& w : want) {
            *w.p = reinterpret_cast<void*>(GetProcAddress(slapi.mod, w.n));
            if (!*w.p) FAIL("sl.interposer.dll lacks export %s", w.n);
        }
        say("[opti_client] sl.interposer loaded at %p, 6 exports resolved", (void*)slapi.mod);
        // --sl-rr: the three RR-request exports, resolved ONLY under that flag (the six-above
        // line stays byte-identical for every existing cell); missing = refuse before slInit.
        if (o.sl_rr) {
            struct { const char* n; void** p; } extra[] = {
                {"slSetFeatureLoaded", reinterpret_cast<void**>(&slapi.setLoaded)},
                {"slGetNewFrameToken", reinterpret_cast<void**>(&slapi.newFrameTok)},
                {"slEvaluateFeature", reinterpret_cast<void**>(&slapi.evalFeat)},
            };
            for (auto& w : extra) {
                *w.p = reinterpret_cast<void*>(GetProcAddress(slapi.mod, w.n));
                if (!*w.p) FAIL("sl.interposer.dll lacks export %s (needed by --sl-rr)", w.n);
            }
            say("[opti_client] sl.interposer RR surface: slSetFeatureLoaded + slGetNewFrameToken + "
                "slEvaluateFeature resolved");
        }
    }

    if (!o.corpus.empty()) {
        // Corpus mode: formats fixed to the V9 defaults (col RGBA16F, mv RG16F, dep R32F) to match
        // the on-disk files; geometry comes from frame.json and caps --frames.
        std::vector<uint8_t> fj;
        if (!read_file(o.corpus + "/frame.json", fj)) FAIL("no %s/frame.json", o.corpus.c_str());
        std::string s(fj.begin(), fj.end());
        double r0 = 0, r1 = 0, o0 = 0, o1 = 0, n = 0;
        if (!jnum_i(s, "render", 0, &r0) || !jnum_i(s, "render", 1, &r1) ||
            !jnum_i(s, "output", 0, &o0) || !jnum_i(s, "output", 1, &o1) ||
            !jnum_i(s, "frames", 0, &n))
            FAIL("frame.json lacks render / output / frames");
        o.w = unsigned(r0); o.h = unsigned(r1);
        o.ow = unsigned(o0); o.oh = unsigned(o1);
        if (o.frames > int(n)) o.frames = int(n);
        o.mv_fmt = "rg16"; o.dep_fmt = "r32f";
    }

    say("[opti_client] mode=%s frames=%d %ux%u -> %ux%u mv=%s dep=%s%s%s\n",
                o.direct ? "direct" : "opti", o.frames, o.w, o.h, o.ow, o.oh,
                o.mv_fmt.c_str(), o.dep_fmt.c_str(),
                o.corpus.empty() ? "" : " corpus=", o.corpus.empty() ? "" : o.corpus.c_str());
    

    // ---- module resolution: OptiScaler's dxgi.dll proxy (default) or the port directly ---------
    HMODULE ngx = nullptr;
    char self[MAX_PATH] = {0};
    GetModuleFileNameA(NULL, self, MAX_PATH);
    for (char* p = self + std::strlen(self); p > self && *p != '\\' && *p != '/'; --p) *p = 0;
    const std::string exe_dir(self);
    if (o.direct) {
        const std::string lib = o.nvngx.empty() ? exe_dir + "_nvngx.dll" : o.nvngx;
        ngx = LoadLibraryA(lib.c_str());
        if (!ngx) FAIL("cannot LoadLibrary %s", lib.c_str());
    } else {
        // OptiScaler reads its ini from its own module's directory (Config.cpp:45), and its logger
        // from the exe's (Logger.cpp Util::ExePath), so exe + dxgi.dll + OptiScaler.ini are
        // co-staged: load from the exe directory first, CWD second.
        const std::string dx = exe_dir + "dxgi.dll";
        ngx = LoadLibraryA(dx.c_str());
        if (!ngx) ngx = LoadLibraryA("dxgi.dll");
        if (!ngx) FAIL("cannot LoadLibrary dxgi.dll (OptiScaler proxy) from %s", dx.c_str());
    }
    struct {
        PFN_NGX_D3D12_Init Init;
        PFN_NGX_AllocateParameters Alloc;
        PFN_NGX_DestroyParameters Destroy;
        PFN_NGX_D3D12_CreateFeature Create;
        PFN_NGX_ReleaseFeature Release;
        PFN_NGX_D3D12_EvaluateFeature Evaluate;
        PFN_NGX_D3D12_Shutdown1 Shutdown1;
    } f{};
    f.Init = (PFN_NGX_D3D12_Init)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Init");
    f.Alloc = (PFN_NGX_AllocateParameters)GetProcAddress(ngx, "NVSDK_NGX_D3D12_AllocateParameters");
    f.Destroy = (PFN_NGX_DestroyParameters)GetProcAddress(ngx, "NVSDK_NGX_D3D12_DestroyParameters");
    f.Create = (PFN_NGX_D3D12_CreateFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_CreateFeature");
    // OptiScaler's dxgi.dll exports ONLY the API-specific spelling NVSDK_NGX_D3D12_ReleaseFeature
    // (verified on the shipped DLL), while the port exports that one AND the unprefixed alias that
    // ngx_client resolves - take D3D12 first, fall back to the alias for direct mode.
    f.Release = (PFN_NGX_ReleaseFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_ReleaseFeature");
    if (!f.Release) f.Release = (PFN_NGX_ReleaseFeature)GetProcAddress(ngx, "NVSDK_NGX_ReleaseFeature");
    f.Evaluate = (PFN_NGX_D3D12_EvaluateFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_EvaluateFeature");
    f.Shutdown1 = (PFN_NGX_D3D12_Shutdown1)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Shutdown1");
    if (!f.Init || !f.Alloc || !f.Destroy || !f.Create || !f.Release || !f.Evaluate)
        FAIL("the loaded module lacks a D3D12 NGX entry point (Init=%p Alloc=%p Destroy=%p Create=%p "
             "Release=%p Eval=%p) - the expected set is Init, Init_Ext, AllocateParameters, "
             "DestroyParameters, GetCapabilityParameters, Create, Evaluate, Release, Shutdown",
             (void*)f.Init, (void*)f.Alloc, (void*)f.Destroy, (void*)f.Create, (void*)f.Release,
             (void*)f.Evaluate);
    say("[opti_client] exports: Init_Ext=%d GetCapabilityParameters=%d Shutdown1=%d\n",
                GetProcAddress(ngx, "NVSDK_NGX_D3D12_Init_Ext") != nullptr,
                GetProcAddress(ngx, "NVSDK_NGX_D3D12_GetCapabilityParameters") != nullptr,
                f.Shutdown1 != nullptr);

    // ---- Streamline slInit (game-mirrored preferences; every field cites its archive evidence) --
    if (o.sl) {
        // pathsToPlugins[0] = this exe's directory: the archive's findPlugins line reads
        // "Looking for plugins in .../bin64" (slayer3:309) = the GAME's exe dir, analogously ours.
        g_sl_plugin_dir = widen(exe_dir);
        g_sl_path_ptrs[0] = g_sl_plugin_dir.c_str();
        sl::Preferences pref{};   // SL_STRUCT ctor + field defaults from the vendored v2.14.1 header
        pref.showConsole = false;
        pref.logLevel = sl::LogLevel::eVerbose;   // most [sl-cb] evidence; log-only effect
        pref.pathsToPlugins = g_sl_path_ptrs;
        pref.numPathsToPlugins = 1;
        pref.pathToLogsAndData = nullptr;         // no SL file log (header: nullptr disables it); the
                                                  // callback below captures every line instead
        pref.logMessageCallback = sl_log_cb;
        // flags mirror the archive: the OTA manifest WAS requested - "readServerManifest Failed to
        // open manifest ..." during slInit (slayer3:278) requires eAllowOTA - while the warn
        // "eLoadDownloadedPlugins flag not passed to preferences" (slayer3:333) pins the MISSING
        // bit, i.e. exactly eDisableCLStateTracking | eAllowOTA (both defined sl_core_types.h).
        pref.flags = sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eAllowOTA;
        pref.featuresToLoad = kGameFeatures;
        pref.numFeaturesToLoad = uint32_t(sizeof(kGameFeatures) / sizeof(kGameFeatures[0]));
        // "initializePlugins ... application ID 101640911" (slayer3:454) - the only preference the
        // SL log echoes; with applicationId set, sl_core_types.h says engine/engineVersion/projectId
        // are NOT required (they stay at defaults; see README: the archive's InProjectId GUID is
        // OptiScaler's own OPTI_GUID, SysUtils.h:33, not a game preference).
        pref.applicationId = 101640911u;
        pref.engine = sl::EngineType::eCustom;    // preserved default (sl_appidentity.h: eCustom=0)
        pref.renderAPI = sl::RenderAPI::eD3D12;
        sl_init_r = slapi.init(pref, sl::kSDKVersion);
        say("SL slInit result=%d (%s) sdk=0x%llx (v2.14.1) appid=%u feats=%u paths=%u\n",
            int(sl_init_r), sl_res_name(sl_init_r), (unsigned long long)sl::kSDKVersion, pref.applicationId,
            pref.numFeaturesToLoad, pref.numPathsToPlugins);
    }

    // ---- D3D12 device/queue/fence (idiom: ngx_client/main.cpp:234-252) --------------------------
    ID3D12Device* dev = nullptr;
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    // slSetD3DDevice IMMEDIATELY after the main device ("should be called IMMEDIATELY after main
    // device is created", sl_core_api.h slSetD3DDevice doc), then the two probes that predicted
    // the game's doom, printed as SLDLSS lines for gate g5 / human eyes.
    if (o.sl) {
        sl_dev_r = slapi.setDevice(dev);
        say("SL slSetD3DDevice result=%d (%s)\n", int(sl_dev_r), sl_res_name(sl_dev_r));
        // AdapterInfo with no LUID on purpose: ID3D12Device exposes no adapter LUID without a
        // DXGI factory, and the sl_core_api.h doc for slIsFeatureSupported says adapter info is
        // OPTIONAL - with fields empty SL answers the OS/driver-level question, which is exactly
        // where the game's "requires driver 512.15.0 or newer - detected 0.0.0" gate lives.
        sl::AdapterInfo ai{};
        const sl::Result sup = slapi.isSupported(sl::kFeatureDLSS, ai);
        say("SLDLSS slIsFeatureSupported(kFeatureDLSS) result=%d (%s) adapter=@(default)\n",
            int(sup), sl_res_name(sup));
        bool loaded = false;
        const sl::Result lr = slapi.isLoaded(sl::kFeatureDLSS, loaded);
        say("SLDLSS slIsFeatureLoaded(kFeatureDLSS) result=%d (%s) loaded=%d\n",
            int(lr), sl_res_name(lr), loaded ? 1 : 0);
        sl::FeatureRequirements fr{};
        const sl::Result rr = slapi.getReqs(sl::kFeatureDLSS, fr);
        say("SLDLSS slGetFeatureRequirements(kFeatureDLSS) result=%d (%s) flags=0x%08x threads=%u "
            "viewports=%u tags=%u driverDetected=%u.%u.%u driverRequired=%u.%u.%u "
            "osDetected=%u.%u.%u osRequired=%u.%u.%u\n",
            int(rr), sl_res_name(rr), unsigned(fr.flags), fr.maxNumCPUThreads, fr.maxNumViewports,
            fr.numRequiredTags, fr.driverVersionDetected.major, fr.driverVersionDetected.minor,
            fr.driverVersionDetected.build, fr.driverVersionRequired.major,
            fr.driverVersionRequired.minor, fr.driverVersionRequired.build,
            fr.osVersionDetected.major, fr.osVersionDetected.minor, fr.osVersionDetected.build,
            fr.osVersionRequired.major, fr.osVersionRequired.minor, fr.osVersionRequired.build);
    }

    // ---- --sl-rr: REQUEST Ray Reconstruction (the in-game RR toggle no harness run has ever
    // made) -------------------------------------------------------
    // --sl boots sl.dlss_d but nothing above asks SL for kFeatureDLSS_RR (1001), so the archived
    // failure chain (bin64/runs/live-capture-121208/OptiScaler.log:491 + every RR toggle after it)
    // could not fire headlessly:
    //   dlss_dEntry.cpp:1237[slOnPluginStartup] NGX parameter indicating DLSSDContext support
    //       cannot be found (0xbad00000) ...
    //   sl.cpp:179[slValidateFeatureContext] 'kFeatureDLSS_RR' context is missing.
    //   pluginManager.cpp:489[setFeatureEnabled] Feature 'kFeatureDLSS_RR' not loaded
    // Whichever of them SL emits arrives through sl_log_cb as [sl-cb] -> client.log -> g5's
    // evidence file, so run_harness.sh can gate them as its optional c6 subcheck.
    // API note (vendored v2.14.1 ground truth): there is NO slSetFeatureEnabled in sl-include/
    // nor in the game interposer's export table - the SDK's enable-with-`true` spelling is
    // slSetFeatureLoaded(sl::Feature, bool) (sl_core_api.h:134, PFun alias :54, "All requested
    // features are loaded by default and have to be unloaded explicitly if needed"), so this
    // block uses it and the gate's greps (not the function name) carry the game's wording.
    if (o.sl_rr) {
        if (sl_init_r != sl::Result::eOk || sl_dev_r != sl::Result::eOk) {
            // never poke SL's load/evaluate APIs on a half-booted stack: report the skip (printed
            // is the whole point), leave the run's own gates to judge everything else.
            say("[opti_client] SL-RR-ENABLE rc=0xSKIP ctx=skip(slInit=0x%08x slSetD3DDevice=0x%08x)",
                unsigned(sl_init_r), unsigned(sl_dev_r));
        } else {
            const sl::Result en = slapi.setLoaded(sl::kFeatureDLSS_RR, true);
            say("[opti_client] SL-RR: slSetFeatureLoaded(kFeatureDLSS_RR, true) -> 0x%08x (%s)",
                unsigned(en), sl_res_name(en));
            bool rr_loaded = false;
            const sl::Result rr_lr = slapi.isLoaded(sl::kFeatureDLSS_RR, rr_loaded);
            say("[opti_client] SL-RR: slIsFeatureLoaded(kFeatureDLSS_RR) -> 0x%08x (%s) loaded=%d",
                unsigned(rr_lr), sl_res_name(rr_lr), rr_loaded ? 1 : 0);
            // Deterministic marker for run_harness.sh (g5 c6 greps "SL-RR-ENABLE rc=").  ctx= is
            // the closest public readout of the game's "context is missing": SL exposes no context
            // handle at all, so loaded/not-loaded from slIsFeatureLoaded is the best that exists.
            say("[opti_client] SL-RR-ENABLE rc=0x%08x ctx=%s", unsigned(en),
                (en == sl::Result::eOk && rr_lr == sl::Result::eOk && rr_loaded) ? "loaded" : "missing");
            // Evaluate smoke, guarded: a fresh frame token FIRST (sl_core_api.h:316 - a valid
            // token is a documented precondition), and a non-OK or null token SKIPS the smoke
            // instead of being dereferenced.  A failing evaluate against a missing context IS the
            // desired evidence and is only ever logged - this block can never fail the run.
            sl::FrameToken* tok = nullptr;
            const sl::Result gt = slapi.newFrameTok(tok, nullptr);
            if (gt != sl::Result::eOk || !tok) {
                say("[opti_client] SL-RR: slEvaluateFeature SKIPPED (slGetNewFrameToken -> 0x%08x (%s) tok=%p)",
                    unsigned(gt), sl_res_name(gt), (void*)tok);
            } else {
                const sl::Result ev = slapi.evalFeat(sl::kFeatureDLSS_RR, *tok, nullptr, 0, nullptr);
                say("[opti_client] SL-RR: slEvaluateFeature(kFeatureDLSS_RR) -> 0x%08x (%s)",
                    unsigned(ev), sl_res_name(ev));
            }
        }
    }
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* q = nullptr; HR(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ID3D12CommandAllocator* ca = nullptr; HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ca)));
    ID3D12GraphicsCommandList* cl = nullptr;
    HR(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ca, nullptr, IID_PPV_ARGS(&cl)));
    ID3D12Fence* fence = nullptr; HR(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    UINT64 fv = 0;
    HANDLE evt = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    auto submit = [&]() -> bool {
        if (FAILED(cl->Close())) return false;
        ID3D12CommandList* l[] = {cl};
        q->ExecuteCommandLists(1, l);
        q->Signal(fence, ++fv);
        if (fence->GetCompletedValue() < fv) { fence->SetEventOnCompletion(fv, evt); WaitForSingleObject(evt, INFINITE); }
        if (dev->GetDeviceRemovedReason() != S_OK) return false;
        return SUCCEEDED(ca->Reset()) && SUCCEEDED(cl->Reset(ca, nullptr));
    };

    // ---- GPU timestamps around EvaluateFeature (net-new grammar, research contract B) ----------
    // One query pair per frame (never re-resolved), a ResolveQueryData into one readback blob at
    // the end of that frame's command list, converted with GetTimestampFrequency after the fence
    // wait.  Every creation-side HRESULT is inspected: an unsupported device degrades to one
    // "GPU_EVAL_MS n/a" line plus "GPU_EVAL_MS <i> n/a" per frame, never a failed run.
    ID3D12QueryHeap* qh = nullptr;
    ID3D12Resource* ts_rb = nullptr;
    UINT64 ts_freq = 0;
    bool ts_ok = false;
    {
        D3D12_QUERY_HEAP_DESC hd{};
        hd.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        hd.Count = 2 * (UINT(o.frames) + 1);
        if (SUCCEEDED(dev->CreateQueryHeap(&hd, IID_PPV_ARGS(&qh))) &&
            SUCCEEDED(q->GetTimestampFrequency(&ts_freq)) && ts_freq) {
            ts_rb = d3d_buffer(dev, UINT64(16) * (UINT64(o.frames) + 1), D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_STATE_COPY_DEST);
            ts_ok = (ts_rb != nullptr);
        }
    }
    if (!ts_ok) {
        say("GPU_EVAL_MS n/a\n");
        
    }

    // ---- geometry-dependent inputs ---------------------------------------------------------------
    const bool mv_rg32 = (o.mv_fmt == "rg32");
    const bool dep_ds8 = (o.dep_fmt == "d32s8");
    const DXGI_FORMAT mv_dxgi = mv_rg32 ? DXGI_FORMAT_R32G32_FLOAT : DXGI_FORMAT_R16G16_FLOAT;
    const unsigned mv_bpp = mv_rg32 ? 8u : 4u;

    struct UpSlot { const char* key; DXGI_FORMAT fmt; unsigned bpp; int kind;   // kind: 0 col 1 mv 2 dep
                    ID3D12Resource *tex = nullptr, *up = nullptr; };             // 3 nrm 4 alb 5 spa 6 rgh
    std::vector<UpSlot> ups;
    ups.push_back({"Color", DXGI_FORMAT_R16G16B16A16_FLOAT, 8u, 0, nullptr, nullptr});
    ups.push_back({"MotionVectors", mv_dxgi, mv_bpp, 1, nullptr, nullptr});
    if (!dep_ds8) ups.push_back({"Depth", DXGI_FORMAT_R32_FLOAT, 4u, 2, nullptr, nullptr});

    // N4: a dlssd cell (the staged OptiScaler.ini says [Upscalers] Dx12Upscaler=dlssd) runs Opti's
    // DLSSDFeatureDx12, i.e. port feature 13, whose evaluate gathers four extra slots (nvngx.cpp
    // 1154-1161 fetch ALL of extra[]) - bind them alongside the three SR inputs, same
    // CommittedResource/upload/barrier/`p->Set(key, res)` pattern as Color/MotionVectors/Depth.
    // --direct never (the client asks the port for feature 1 itself) and --corpus never (the V9
    // corpus contract is col/mv/dep only; an RR corpus would need nrm/alb/spa/rgh files too).
    const bool rr_run = !o.direct && o.corpus.empty() &&
                        ini_declares_dlssd(exe_dir + "OptiScaler.ini");
    if (rr_run) {
        ups.push_back({"GBuffer.Normals", DXGI_FORMAT_R16G16B16A16_FLOAT, 8u, 3, nullptr, nullptr});
        ups.push_back({"DLSS.Input.DiffuseAlbedo", DXGI_FORMAT_R16G16B16A16_FLOAT, 8u, 4, nullptr, nullptr});
        ups.push_back({"DLSS.Input.SpecularAlbedo", DXGI_FORMAT_R16G16B16A16_FLOAT, 8u, 5, nullptr, nullptr});
        ups.push_back({"GBuffer.Roughness", DXGI_FORMAT_R32_FLOAT, 4u, 6, nullptr, nullptr});
        say("[opti_client] RR=1: staged OptiScaler.ini [Upscalers] Dx12Upscaler=dlssd -> OptiScaler "
            "creates NVSDK_NGX_Feature_RayReconstruction (13); binding the port's four mandatory RR "
            "slots (GBuffer.Normals, DLSS.Input.DiffuseAlbedo, DLSS.Input.SpecularAlbedo, "
            "GBuffer.Roughness) every eval\n");
    }

    ID3D12Resource* depth = nullptr;
    ID3D12DescriptorHeap* dsv_heap = nullptr;
    if (dep_ds8) {
        // ISSUE 1 reproduction surface: the game hands the port a typeless D32+S8 depth resource
        // whose resource desc carries DXGI format 19 (nvngx.cpp:960-963); the fixed port decodes it,
        // the archived DLL refuses it.  D3D12 gives no COPY_DEST for a depth-stencil resource and
        // ClearDepthStencilView cannot paint a gradient, so the content is a constant 0.5 fill.
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = o.w; d.Height = o.h; d.DepthOrArraySize = 1; d.MipLevels = 1;
        d.Format = DXGI_FORMAT_R32G8X24_TYPELESS;      // == 19: what the port's GetDesc will see
        d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_CLEAR_VALUE cv{};
        cv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;   // the typed DSV this typeless family allows
        cv.DepthStencil = {0.5f, 0};
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                                 D3D12_RESOURCE_STATE_DEPTH_WRITE, &cv,
                                                 IID_PPV_ARGS(&depth))) || !depth)
            FAIL("cannot create the depth-stencil texture (DXGI 19)");
        // A real CPU DSV slot: a null D3D12_CPU_DESCRIPTOR_HANDLE makes vkd3d write a view into
        // descriptor address 0 (measured: the first run crashed there with rc=5 before frame 0's
        // evaluate, with no port log at all).
        D3D12_DESCRIPTOR_HEAP_DESC hh{};
        hh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;   // mingw's name for the SDK's ..._DEPTH_STENCIL = 3
        hh.NumDescriptors = 1;
        if (FAILED(dev->CreateDescriptorHeap(&hh, IID_PPV_ARGS(&dsv_heap))) || !dsv_heap)
            FAIL("cannot create the DSV descriptor heap");
    }

    for (auto& s : ups) {
        {
            D3D12_RESOURCE_DESC d{};
            d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            d.Width = o.w; d.Height = o.h; d.DepthOrArraySize = 1; d.MipLevels = 1;
            d.Format = s.fmt; d.SampleDesc.Count = 1;
            D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
            if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&s.tex))) || !s.tex)
                FAIL("cannot create the %s texture", s.key);
        }
        s.up = d3d_buffer(dev, UINT64(pitch256(o.w * s.bpp)) * o.h, D3D12_HEAP_TYPE_UPLOAD,
                          D3D12_RESOURCE_STATE_GENERIC_READ);
        if (!s.up) FAIL("cannot create the %s upload buffer", s.key);
    }

    ID3D12Resource* out = nullptr;
    {
        D3D12_RESOURCE_DESC d{};
        d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        d.Width = o.ow; d.Height = o.oh; d.DepthOrArraySize = 1; d.MipLevels = 1;
#if FMT16
        d.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
#else
        d.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
#endif
        d.SampleDesc.Count = 1;
        d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                 IID_PPV_ARGS(&out))) || !out)
            FAIL("cannot create the output texture");
    }
    const UINT opitch = pitch256(o.ow * (FMT16 ? 8u : 16u));
    ID3D12Resource* out_rb = d3d_buffer(dev, UINT64(opitch) * o.oh, D3D12_HEAP_TYPE_READBACK,
                                        D3D12_RESOURCE_STATE_COPY_DEST);
    if (!out_rb) FAIL("cannot create the output readback");

    // ---- NGX host sequence (the documented OptiScaler host surface, research contract A) --------
    const NVSDK_NGX_Result ir = f.Init(0x13371337, L".", dev, nullptr, NGX_VERSION_API);
    if (!ngx_ok(ir)) FAIL("NVSDK_NGX_D3D12_Init = 0x%08x", unsigned(ir));
    NVSDK_NGX_Parameter* p = nullptr;
    if (!f.Alloc(&p) || !p) FAIL("NVSDK_NGX_D3D12_AllocateParameters failed");

    // create params, mirroring ngx_client's create_params (ngx_client/main.cpp:150-168): sizes,
    // quality ratio, node masks, create flags, preset M everywhere.  OptiScaler's
    // ProcessInitParams (upscalers/dlss/DLSSFeature.cpp:49) may overwrite ONLY the create flags.
    p->Set("Width", o.w); p->Set("Height", o.h);
    p->Set("OutWidth", o.ow); p->Set("OutHeight", o.oh);
    {
        const double r = double(o.ow) / double(o.w);
        struct { double r; int q; } t[] = {{1.0, NGX_PQ_DLAA}, {1.3, NGX_PQ_ULTRA_QUALITY}, {1.5, NGX_PQ_MAX_QUALITY},
                                           {1.724, NGX_PQ_BALANCED}, {2.0, NGX_PQ_MAX_PERF}, {3.0, NGX_PQ_ULTRA_PERF}};
        int best = 0;
        for (int k = 1; k < 6; ++k) if (std::fabs(t[k].r - r) < std::fabs(t[best].r - r)) best = k;
        p->Set("PerfQualityValue", t[best].q);
    }
    p->Set("CreationNodeMask", 1u); p->Set("VisibilityNodeMask", 1u);
    p->Set("DLSS.Feature.Create.Flags", int(NGX_FLAG_IS_HDR | NGX_FLAG_MV_LOW_RES));
    for (const char* m : {"DLAA", "Quality", "Balanced", "Performance", "UltraPerformance", "UltraQuality"})
        p->Set((std::string("DLSS.Hint.Render.Preset.") + m).c_str(), int(NGX_PRESET_M));
    p->Set("Sharpness", 0.0f);   // research contract A: nothing sharpens or rewrites our evaluate

    std::vector<std::string> params_json;
    if (!o.corpus.empty())
        for (int i = 0; i < o.frames; ++i) {
            std::vector<uint8_t> b;
            if (!read_file(o.corpus + "/params_" + std::to_string(i) + ".json", b))
                FAIL("no params_%d.json in %s", i, o.corpus.c_str());
            params_json.emplace_back(b.begin(), b.end());
        }

    // CreateFeature: OptiScaler blocks ~500 ms first (DLSSFeature_Dx12.cpp:38) and the port then
    // builds the frame chain (~3.8 s at 4K, measured port log).  CreateFeature succeeds in both
    // normal and --expect-fail runs: issue 1's refusal happens per FRAME inside EvaluateFeature.
    NVSDK_NGX_Handle* h = nullptr;
    const NVSDK_NGX_Result cr = f.Create(cl, NVSDK_NGX_Feature_SuperSampling, p, &h);
    if (!submit()) FAIL("device lost after CreateFeature");
    if (!ngx_ok(cr) || !h) FAIL("CreateFeature = 0x%08x", unsigned(cr));
    say("[opti_client] CreateFeature = 0x%08x handle=%p\n", unsigned(cr), (void*)h);
    

    std::vector<FrameRec> recs(size_t(o.frames));
    std::vector<float> host(size_t(o.ow) * o.oh * 4);
    CreateDirectoryA(o.odir.c_str(), nullptr);

    LARGE_INTEGER qpc_fr, qpc_a, qpc_b;
    QueryPerformanceFrequency(&qpc_fr);

    for (int i = 0; i < o.frames; ++i) {
        FrameRec& fr = recs[size_t(i)];
        fr.i = i;

        // ---- upload this frame's inputs into our own command list ----
        for (auto& s : ups) {
            const UINT pt = pitch256(o.w * s.bpp);
            void* m = nullptr;
            HR(s.up->Map(0, nullptr, &m));
            if (!o.corpus.empty()) {
                const char* stem = s.kind == 0 ? "col" : s.kind == 1 ? "mv" : "dep";
                const char* ext = s.kind == 2 ? ".f32" : ".f16";
                const std::string path = o.corpus + "/" + stem + "_" + std::to_string(i) + ext;
                std::vector<uint8_t> b;
                if (!read_file(path, b) || b.size() != size_t(o.w) * o.h * s.bpp)
                    FAIL("corpus input %s: missing or wrong size (%zu bytes)", path.c_str(), b.size());
                for (unsigned y = 0; y < o.h; ++y)
                    std::memcpy(static_cast<uint8_t*>(m) + size_t(y) * pt,
                                b.data() + size_t(y) * o.w * s.bpp, size_t(o.w) * s.bpp);
            } else if (s.kind == 0) synth_color(m, pt, o.w, o.h, i);
            else if (s.kind == 1) synth_mv(m, pt, o.w, o.h, mv_rg32);
            else if (s.kind == 2) synth_depth(m, pt, o.w, o.h);
            else synth_rr(unsigned(s.kind), m, pt, o.w, o.h);   // kind 3..6, the RR slots (rr_run only)
            s.up->Unmap(0, nullptr);
            if (i > 0) d3d_barrier(cl, s.tex, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
            src.pResource = s.up; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint.Footprint = {s.fmt, o.w, o.h, 1, pt};
            dst.pResource = s.tex; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            d3d_barrier(cl, s.tex, D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            p->Set(s.key, s.tex);   // ID3D12Resource*, the D3D12 NGX convention (ngx_client:306)
        }
        if (depth) {
            if (i == 0) {
                D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
                dsv.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
                dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                D3D12_CPU_DESCRIPTOR_HANDLE cpu = dsv_heap->GetCPUDescriptorHandleForHeapStart();
                dev->CreateDepthStencilView(depth, &dsv, cpu);
                cl->ClearDepthStencilView(cpu, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
            }
            p->Set("Depth", depth);
        }
        p->Set("Output", out);

        // ---- frame params (mirror ngx_client frame_params; jitter is deterministic) ----
        if (!o.corpus.empty()) {
            const std::string& js = params_json.at(size_t(i));
            double jx = 0, jy = 0, sx = 1.0, sy = 1.0;
            jnum_i(js, "jitter", 0, &jx); jnum_i(js, "jitter", 1, &jy);
            jnum_i(js, "mv_scale", 0, &sx); jnum_i(js, "mv_scale", 1, &sy);
            p->Set("Jitter.Offset.X", float(jx));
            p->Set("Jitter.Offset.Y", float(jy));
            p->Set("MV.Scale.X", float(sx));
            p->Set("MV.Scale.Y", float(sy));
            p->Set("Reset", int(jnum_d(js, "reset", 0) != 0));
            p->Set("DLSS.Pre.Exposure", float(jnum_d(js, "pre_exposure", 1.0)));
        } else {
            p->Set("Jitter.Offset.X", float((i * 5) % 8) * 0.125f - 0.4375f);
            p->Set("Jitter.Offset.Y", float((i * 3) % 8) * 0.125f - 0.4375f);
            p->Set("MV.Scale.X", 1.0f);
            p->Set("MV.Scale.Y", 1.0f);
            p->Set("Reset", i == 0 ? 1 : 0);
            p->Set("DLSS.Pre.Exposure", 1.0f);
        }
        p->Set("DLSS.Exposure.Scale", 1.0f);
        p->Set("DLSS.Render.Subrect.Dimensions.Width", int(o.w));
        p->Set("DLSS.Render.Subrect.Dimensions.Height", int(o.h));

        // ---- the measured call ----
        if (ts_ok) ts_end(cl, qh, UINT(2 * i));
        QueryPerformanceCounter(&qpc_a);
        const NVSDK_NGX_Result er = f.Evaluate(cl, h, p, nullptr);
        QueryPerformanceCounter(&qpc_b);
        fr.eval_ms = double(qpc_b.QuadPart - qpc_a.QuadPart) * 1000.0 / double(qpc_fr.QuadPart);
        fr.eval_ok = ngx_ok(er);
        if (ts_ok) ts_end(cl, qh, UINT(2 * i + 1));
        say("EVAL_MS %d %.4f\n", i, fr.eval_ms);
        if (rr_run) {
            // One line per evaluate listing every RR slot THIS client bound for that eval - the
            // key, the exact DXGI format and the resource pointer - in `slot=` wording mirroring
            // the port's refusal line (`eval #N FAIL: ... slot=X why=...`, nvngx.cpp:1230), so a
            // refusal in dlss5-amd-ngx.log stays self-explaining against the bound state.
            char slots[4 * 128] = "";
            for (auto& s : ups) if (s.kind >= 3)
                std::snprintf(slots + std::strlen(slots), sizeof slots - std::strlen(slots),
                              " slot=%s res=%p fmt=%s", s.key, (void*)s.tex, rr_slot_fmt(unsigned(s.kind)));
            say("RR_SLOTS i=%d eval_ok=%d feat=13%s\n", i, fr.eval_ok ? 1 : 0, slots);
        }

        // ---- output readback ----
        d3d_barrier(cl, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        {
            D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
            src.pResource = out; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = out_rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Footprint = {
#if FMT16
                DXGI_FORMAT_R16G16B16A16_FLOAT,
#else
                DXGI_FORMAT_R32G32B32A32_FLOAT,
#endif
                o.ow, o.oh, 1, opitch};
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        d3d_barrier(cl, out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (ts_ok) cl->ResolveQueryData(qh, D3D12_QUERY_TYPE_TIMESTAMP, UINT(2 * i), 2, ts_rb,
                                        UINT64(16) * UINT64(i));
        if (!submit())
            FAIL("device lost in frame %d (0x%08lx)", i, (unsigned long)dev->GetDeviceRemovedReason());

        if (ts_ok) {
            void* tm = nullptr;
            D3D12_RANGE rr{UINT64(16) * UINT64(i), UINT64(16) * UINT64(i) + 16};
            if (SUCCEEDED(ts_rb->Map(0, &rr, &tm)) && tm) {
                const uint64_t* t = reinterpret_cast<const uint64_t*>(static_cast<uint8_t*>(tm) + 16 * i);
                if (t[1] >= t[0]) {
                    fr.gpu_ms = double(t[1] - t[0]) * 1000.0 / double(ts_freq);
                    fr.gpu_valid = true;
                    say("GPU_EVAL_MS %d %.4f\n", i, fr.gpu_ms);
                }
                ts_rb->Unmap(0, nullptr);
            }
        } else if (i > 0) {
            // degradation contract: n/a from frame 1 on instead of failing (frame 0's attempt is
            // where the run discovers timestamps are unsupported - see the creation block above)
            say("GPU_EVAL_MS %d n/a\n", i);
        }

        void* m = nullptr;
        D3D12_RANGE range{0, size_t(opitch) * o.oh};
        HR(out_rb->Map(0, &range, &m));
#if FMT16
        for (unsigned y = 0; y < o.oh; ++y) {
            const uint16_t* row = reinterpret_cast<const uint16_t*>(static_cast<uint8_t*>(m) + size_t(y) * opitch);
            float* dstrow = host.data() + size_t(y) * size_t(o.ow) * 4;
            for (unsigned x = 0; x < o.ow * 4; ++x) dstrow[x] = h2f_(row[x]);   // widen: out_<i>.f32 stays f32
        }
#else
        for (unsigned y = 0; y < o.oh; ++y)
            std::memcpy(host.data() + size_t(y) * size_t(o.ow) * 4,
                        static_cast<uint8_t*>(m) + size_t(y) * opitch, size_t(o.ow) * 16);
#endif
        out_rb->Unmap(0, nullptr);
        if (!write_file(o.odir + "/out_" + std::to_string(i) + ".f32", host.data(), host.size() * 4))
            FAIL("cannot write %s/out_%d.f32", o.odir.c_str(), i);

        // Non-black statistics: a pixel is non-zero when any of its 4 channels is non-zero.
        size_t nz = 0;
        float mn = 0, mx = 0;
        const size_t npx = size_t(o.ow) * o.oh;
        for (size_t k = 0; k < npx; ++k) {
            const float* px = &host[k * 4];
            if ((px[0] != 0.0f) || (px[1] != 0.0f) || (px[2] != 0.0f) || (px[3] != 0.0f)) ++nz;
            for (int cch = 0; cch < 4; ++cch) {
                if (px[cch] < mn) mn = px[cch];
                if (px[cch] > mx) mx = px[cch];
            }
        }
        fr.nz = double(nz) / double(npx);
        fr.mn = mn; fr.mx = mx;
        say("OUT_NZ %d nz=%.6f min=%.6g max=%.6g eval_ok=%d\n", i, fr.nz, fr.mn, fr.mx,
                    fr.eval_ok ? 1 : 0);
        
    }

    // ---- teardown ----
    if (h && f.Release) f.Release(h);
    if (f.Destroy) f.Destroy(p);
    if (f.Shutdown1) f.Shutdown1(dev);
    if (o.sl) {
        const sl::Result r = slapi.shutdown();
        say("SL slShutdown result=%d (%s)\n", int(r), sl_res_name(r));
    }

    // ---- summary ---------------------------------------------------------------------------
    std::vector<double> ev, gp, nzv;
    for (int i = 1; i < o.frames; ++i) {        // frame 0 excluded everywhere (the repo convention)
        ev.push_back(recs[size_t(i)].eval_ms);
        if (recs[size_t(i)].gpu_valid) gp.push_back(recs[size_t(i)].gpu_ms);
        nzv.push_back(recs[size_t(i)].nz);
    }
    const PortLog pl = read_port_log();
    int e_ok = 0, e_fail = 0;
    for (auto& r : recs) r.eval_ok ? ++e_ok : ++e_fail;

    // Gates, inverted wholesale by --expect-fail (the issue-1 reproduction demo):
    //   eval       every EvaluateFeature returned success  <-> at least one was refused
    //   nonblack   every frame nz >= 0.10                  <-> every frame nz < 0.10 (black)
    //   timing     frames 1..N have EVAL_MS data with median > 0
    //   port       the port's own eval OK/FAIL counts from dlss5-amd-ngx.log (run_harness.sh gates
    //              this strictly too; here it catches "OptiScaler quietly served someone else's
    //              upscaler" because a different provider logs zero port evals)
    const bool g_eval = o.expect_fail ? (e_fail >= 1) : (e_fail == 0 && e_ok == o.frames);
    bool g_nonblack = true;
    for (auto& r : recs) {
        if (o.expect_fail) { if (!(r.nz >= 0.0 && r.nz < 0.10)) g_nonblack = false; }
        else if (!(r.nz >= 0.10)) g_nonblack = false;
    }
    const bool g_timing = !ev.empty() && median_(ev) > 0.0;
    const bool g_port = o.expect_fail
        ? (pl.fail >= 1)
        : (pl.present && pl.fail == 0 && pl.ok >= o.frames - 2 && pl.ok > 0);

    char summary[4096];
    std::snprintf(summary, sizeof summary,
        "mode=%s\nframes=%d\nrender=%ux%u\noutput=%ux%u\nmv=%s\ndep=%s\ncorpus=%s\nexpect_fail=%d\n"
        "EVAL_MS n=%zu median=%.4f p95=%.4f min=%.4f max=%.4f  (frames 1..)\n"
        "GPU_EVAL_MS n=%zu median=%.4f p95=%.4f tags=%s\n"
        "PORT_RECORD_MS present=%d n=%d median=%.4f  (dlss5-amd-ngx.log 'recorded in N ms')\n"
        "PORT_EVAL ok=%d fail=%d\n"
        "OUT_NZ min=%.6f max=%.6f threshold=0.10  (frames 1..)\n"
        "GATE eval=%s\nGATE nonblack=%s\nGATE timing=%s\nGATE port=%s\nRESULT=%s\n",
        o.direct ? "direct" : "opti", o.frames, o.w, o.h, o.ow, o.oh,
        o.mv_fmt.c_str(), o.dep_fmt.c_str(), o.corpus.c_str(), o.expect_fail ? 1 : 0,
        ev.size(), median_(ev), p95_(ev), min_of(ev), max_of(ev),
        gp.size(), median_(gp), p95_(gp), ts_ok ? "d3d12-timestamp" : "n/a-timestamps-unsupported",
        pl.present ? 1 : 0, pl.rec_n, pl.rec_median,
        pl.ok, pl.fail,
        min_of(nzv), max_of(nzv),
        g_eval ? "PASS" : "FAIL", g_nonblack ? "PASS" : "FAIL",
        g_timing ? "PASS" : "FAIL", g_port ? "PASS" : "FAIL",
        (g_eval && g_nonblack && g_timing && g_port) ? "PASS" : "FAIL");

    say("HARNESS SUMMARY\n%s", summary);
    

    if (!o.csv.empty()) {
        std::FILE* cf = std::fopen(o.csv.c_str(), "w");
        if (cf) {
            std::fprintf(cf, "frame,eval_ms,gpu_ms,nz,min,max,eval_ok\n");
            for (auto& r : recs)
                std::fprintf(cf, "%d,%.4f,%.4f,%.6f,%.6g,%.6g,%d\n", r.i, r.eval_ms,
                             r.gpu_valid ? r.gpu_ms : -1.0, r.nz, r.mn, r.mx, r.eval_ok ? 1 : 0);
            std::fclose(cf);
        }
    }
    {
        const std::string sp = o.odir + "/summary.txt";
        std::FILE* sf = std::fopen(sp.c_str(), "w");
        if (sf) { std::fputs(summary, sf); std::fclose(sf); }
    }
    const bool pass = g_eval && g_nonblack && g_timing && g_port;
    return pass ? 0 : 1;
}