#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <fstream>
#include <algorithm>

#include "../../tools/verify/ngx_client/ngx_abi.h"
#include "../vk/pipeline.h"
#include "../vk/vkfuncs.h"

// The ffx surface (src/ffx/ffx_front.cpp) links this same TU inside dlss5-amd-ffx.dll and drives
// the bridge through the declarations below: one dxgi_to_vk map, one interop dance, one
// record_bridge - the two surfaces cannot be allowed to drift apart.  Everything the NGX surface
// keeps private (its own exports, the parameter object, the D3D12 hooks) stays `static` here.
#include "ngx_bridge_shared.h"

const GUID ngxd_IID_ID3D12GraphicsCommandListExt = 
    { 0x77a86b09, 0x2bea, 0x4801, { 0xb8, 0x9a, 0x37, 0x64, 0x8e, 0x10, 0x4a, 0xf1 } };

const GUID ngxd_IID_ID3D12DXVKInteropDevice = 
    { 0x39da4e09, 0xbd1c, 0x4198, { 0x9f, 0xae, 0x86, 0xbb, 0xe3, 0xbe, 0x41, 0xfd } };

// The interop interfaces/IIDs now live in d3d12_interop.h (included via ngx_bridge_shared.h);
// the aliases below keep every later `IID_*` call site here compiling untouched.
static const GUID& IID_ID3D12GraphicsCommandListExt = ngxd_IID_ID3D12GraphicsCommandListExt;
static const GUID& IID_ID3D12DXVKInteropDevice = ngxd_IID_ID3D12DXVKInteropDevice;

HINSTANCE ngxd_hinstance = NULL;   // DllMain's instance (also used below via g_hinstance alias)

// Barrier helper + format helpers the FFX surface needs for nothing directly (record_bridge uses
// them internally); they stay static here - the shared surface goes through ngxd_record_bridge.

// The frame chain is a plain C++ library (src/vk/pipeline.cpp); under the DLL it reaches Vulkan through
// the pointers below, filled from NVSDK_NGX_VULKAN_Init's gipa/gdpa or from winevulkan in the process.
#ifdef VK_NO_PROTOTYPES
#define NGXD_VK_DEFINE(f) PFN_##f f = nullptr;
NGXD_VK_FUNCS(NGXD_VK_DEFINE)
#undef NGXD_VK_DEFINE
#endif

static PFN_vkGetInstanceProcAddr g_gipa = nullptr;
static PFN_vkGetDeviceProcAddr g_gdpa = nullptr;
static HMODULE g_vk_module = nullptr;   // the module the entry points came from
static bool g_vk_loaded = false;         // entry points resolved once

// Parameter value representation with type cross-conversion
struct ParamValue {
    enum class Kind { None, ULL, F, D, UI, I, Ptr } kind = Kind::None;
    unsigned long long ull = 0;
    float f = 0.0f;
    double d = 0.0;
    unsigned int ui = 0;
    int i = 0;
    void* ptr = nullptr;

    bool is_pointer() const { return kind == Kind::Ptr; }

    template <typename T> bool number(T* out) const {
        switch (kind) {
            case Kind::ULL: *out = static_cast<T>(ull); return true;
            case Kind::F:   *out = static_cast<T>(f);   return true;
            case Kind::D:   *out = static_cast<T>(d);   return true;
            case Kind::UI:  *out = static_cast<T>(ui);  return true;
            case Kind::I:   *out = static_cast<T>(i);   return true;
            default: return false;
        }
    }
};

class ParametersImpl {
public:
    ParametersImpl();
    const NVSDK_NGX_ParameterVtbl* vtbl;

    NVSDK_NGX_Parameter* as_param() { return reinterpret_cast<NVSDK_NGX_Parameter*>(this); }

    void set_ull(const char* name, unsigned long long v) { auto& s = slot(name); s.kind = ParamValue::Kind::ULL; s.ull = v; }
    void set_float(const char* name, float v)            { auto& s = slot(name); s.kind = ParamValue::Kind::F;   s.f = v; }
    void set_double(const char* name, double v)          { auto& s = slot(name); s.kind = ParamValue::Kind::D;   s.d = v; }
    void set_uint(const char* name, unsigned int v)      { auto& s = slot(name); s.kind = ParamValue::Kind::UI;  s.ui = v; }
    void set_int(const char* name, int v)                { auto& s = slot(name); s.kind = ParamValue::Kind::I;   s.i = v; }
    void set_ptr(const char* name, void* v) {
        auto& s = slot(name);
        s.kind = ParamValue::Kind::Ptr;
        s.ptr = v;
        s.ull = reinterpret_cast<unsigned long long>(v);
    }

    NVSDK_NGX_Result get_ull(const char* name, unsigned long long* out) const {
        const ParamValue* v = find(name);
        if (!v || !out) return NVSDK_NGX_Result_Fail;
        if (v->is_pointer()) { *out = reinterpret_cast<unsigned long long>(v->ptr); return NVSDK_NGX_Result_Success; }
        return v->number(out) ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
    }
    NVSDK_NGX_Result get_float(const char* name, float* out) const   { return number(name, out); }
    NVSDK_NGX_Result get_double(const char* name, double* out) const { return number(name, out); }
    NVSDK_NGX_Result get_uint(const char* name, unsigned int* out) const { return number(name, out); }
    NVSDK_NGX_Result get_int(const char* name, int* out) const       { return number(name, out); }
    NVSDK_NGX_Result get_ptr(const char* name, void** out) const {
        const ParamValue* v = find(name);
        if (!v || !out) return NVSDK_NGX_Result_Fail;
        if (!v->is_pointer()) return NVSDK_NGX_Result_Fail;
        *out = v->ptr;
        return NVSDK_NGX_Result_Success;
    }

    void reset() {
        std::lock_guard<std::mutex> guard(lock_);
        values_.clear();
    }

private:
    ParamValue& slot(const char* name) {
        std::lock_guard<std::mutex> guard(lock_);
        return values_[name ? name : ""];
    }
    const ParamValue* find(const char* name) const {
        std::lock_guard<std::mutex> guard(lock_);
        auto it = values_.find(name ? name : "");
        return it == values_.end() ? nullptr : &it->second;
    }
    template <typename T> NVSDK_NGX_Result number(const char* name, T* out) const {
        const ParamValue* v = find(name);
        if (!v || !out) return NVSDK_NGX_Result_Fail;
        return v->number(out) ? NVSDK_NGX_Result_Success : NVSDK_NGX_Result_Fail;
    }

    mutable std::mutex lock_;
    std::map<std::string, ParamValue> values_;
};

static void NGX_MSABI p_set_ptr(NVSDK_NGX_Parameter* p, const char* k, void* v)              { reinterpret_cast<ParametersImpl*>(p)->set_ptr(k, v); }
static void NGX_MSABI p_set_d3d12(NVSDK_NGX_Parameter* p, const char* k, ID3D12Resource* v) { reinterpret_cast<ParametersImpl*>(p)->set_ptr(k, v); }
static void NGX_MSABI p_set_d3d11(NVSDK_NGX_Parameter* p, const char* k, ID3D11Resource* v) { reinterpret_cast<ParametersImpl*>(p)->set_ptr(k, v); }
static void NGX_MSABI p_set_int(NVSDK_NGX_Parameter* p, const char* k, int v)                { reinterpret_cast<ParametersImpl*>(p)->set_int(k, v); }
static void NGX_MSABI p_set_uint(NVSDK_NGX_Parameter* p, const char* k, unsigned int v)     { reinterpret_cast<ParametersImpl*>(p)->set_uint(k, v); }
static void NGX_MSABI p_set_double(NVSDK_NGX_Parameter* p, const char* k, double v)         { reinterpret_cast<ParametersImpl*>(p)->set_double(k, v); }
static void NGX_MSABI p_set_float(NVSDK_NGX_Parameter* p, const char* k, float v)           { reinterpret_cast<ParametersImpl*>(p)->set_float(k, v); }
static void NGX_MSABI p_set_ull(NVSDK_NGX_Parameter* p, const char* k, unsigned long long v) { reinterpret_cast<ParametersImpl*>(p)->set_ull(k, v); }

static NVSDK_NGX_Result NGX_MSABI p_get_ptr(const NVSDK_NGX_Parameter* p, const char* k, void** v)              { return reinterpret_cast<const ParametersImpl*>(p)->get_ptr(k, v); }
static NVSDK_NGX_Result NGX_MSABI p_get_d3d12(const NVSDK_NGX_Parameter* p, const char* k, ID3D12Resource** v) { return reinterpret_cast<const ParametersImpl*>(p)->get_ptr(k, reinterpret_cast<void**>(v)); }
static NVSDK_NGX_Result NGX_MSABI p_get_d3d11(const NVSDK_NGX_Parameter* p, const char* k, ID3D11Resource** v) { return reinterpret_cast<const ParametersImpl*>(p)->get_ptr(k, reinterpret_cast<void**>(v)); }
static NVSDK_NGX_Result NGX_MSABI p_get_int(const NVSDK_NGX_Parameter* p, const char* k, int* v)                { return reinterpret_cast<const ParametersImpl*>(p)->get_int(k, v); }
static NVSDK_NGX_Result NGX_MSABI p_get_uint(const NVSDK_NGX_Parameter* p, const char* k, unsigned int* v)     { return reinterpret_cast<const ParametersImpl*>(p)->get_uint(k, v); }
static NVSDK_NGX_Result NGX_MSABI p_get_double(const NVSDK_NGX_Parameter* p, const char* k, double* v)          { return reinterpret_cast<const ParametersImpl*>(p)->get_double(k, v); }
static NVSDK_NGX_Result NGX_MSABI p_get_float(const NVSDK_NGX_Parameter* p, const char* k, float* v)            { return reinterpret_cast<const ParametersImpl*>(p)->get_float(k, v); }
static NVSDK_NGX_Result NGX_MSABI p_get_ull(const NVSDK_NGX_Parameter* p, const char* k, unsigned long long* v) { return reinterpret_cast<const ParametersImpl*>(p)->get_ull(k, v); }
static void NGX_MSABI p_reset(NVSDK_NGX_Parameter* p) { reinterpret_cast<ParametersImpl*>(p)->reset(); }

static const NVSDK_NGX_ParameterVtbl g_param_vtbl = {
    p_set_ptr, p_set_d3d12, p_set_d3d11, p_set_int, p_set_uint, p_set_double, p_set_float, p_set_ull,
    p_get_ptr, p_get_d3d12, p_get_d3d11, p_get_int, p_get_uint, p_get_double, p_get_float, p_get_ull,
    p_reset
};

ParametersImpl::ParametersImpl() : vtbl(&g_param_vtbl) {}

// g_hinstance (the NGX surface's historical name) IS the shared bridge instance slot: one
// storage, two names, and DllMain below sets it once.
static HINSTANCE& g_hinstance = ngxd_hinstance;

// Check NVIDIA DLL presence and size
static bool check_nvidia_dll(NVSDK_NGX_Feature feat) {
    const char* filename = (feat == NVSDK_NGX_Feature_SuperSampling) ? "nvngx_dlss.dll" : "nvngx_dlssd.dll";
    uint64_t expected_size = (feat == NVSDK_NGX_Feature_SuperSampling) ? 58956912ULL : 48343664ULL;

    char dll_path[MAX_PATH] = {0};
    if (g_hinstance) {
        if (GetModuleFileNameA(g_hinstance, dll_path, MAX_PATH)) {
            char* last_slash = std::strrchr(dll_path, '\\');
            if (!last_slash) last_slash = std::strrchr(dll_path, '/');
            if (last_slash) {
                *(last_slash + 1) = '\0';
                std::strcat(dll_path, filename);
            }
        }
    }

    HANDLE hFile = CreateFileA(dll_path[0] ? dll_path : filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE && dll_path[0]) {
        hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    }
    if (hFile == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hFile, &sz)) {
        CloseHandle(hFile);
        return false;
    }
    CloseHandle(hFile);
    return (uint64_t)sz.QuadPart == expected_size;
}

// Global context
static uint32_t g_next_handle_id = 1;
static VkInstance g_vk_instance = VK_NULL_HANDLE;
static VkPhysicalDevice g_vk_phys_dev = VK_NULL_HANDLE;
static VkDevice g_vk_device = VK_NULL_HANDLE;
static ID3D12Device* g_d3d12_device = nullptr;

// ------------------------------------------------------------------------------------------------
// The bridge: what a feature is, and how an NGX resource reaches the frame chain
// ------------------------------------------------------------------------------------------------

static std::mutex g_bridge_lock;

// Diagnostics (build tag kPortTag below): one id per EvaluateFeature call, bumped under the same
// lock every Evaluate path already holds, so the ids are a total order over *this* DLL's calls and
// a fail/success pair in the log can be split back into the calls it came from.  A second or stale
// _nvngx.dll in the process has its own counter and its own tag, so interleaving two sequences (or
// lines without a tag) is exactly what these two values exist to expose.
static uint64_t g_eval_seq = 0;
static const char* kPortTag = "diag-v1";   // which build wrote a line: stamped on the init lines

static unsigned long long g_log_t0 = 0;
unsigned long long ngx_ms() { return GetTickCount64(); }

void ngx_log(const char* fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (!g_log_t0) g_log_t0 = ngx_ms();
    const unsigned long long at = ngx_ms() - g_log_t0;
    std::fprintf(stderr, "[dlss5-amd %8llu ms] %s\n", at, buf);
    std::fflush(stderr);
    // The game folder is off limits (uninstall must restore it byte for byte), so the log goes to the
    // working directory, which is where NGX itself writes its logs too.
    if (FILE* f = std::fopen("dlss5-amd-ngx.log", "a")) {
        std::fprintf(f, "[%8llu ms] %s\n", at, buf);
        std::fclose(f);
    }
}

// The directory this DLL sits in, with `tail` appended: the game's dlss5-amd tree.
static bool module_path_tail(const char* tail, char* out, size_t cap) {
    if (!g_hinstance || !out || cap < 2) return false;
    DWORD n = GetModuleFileNameA(g_hinstance, out, DWORD(cap));
    if (!n || n >= cap) return false;
    char* slash = std::strrchr(out, '\\');
    if (!slash) slash = std::strrchr(out, '/');
    if (!slash) return false;
    slash[1] = '\0';
    size_t used = std::strlen(out), add = std::strlen(tail);
    if (used + add + 1 >= cap) return false;
    std::memcpy(out + used, tail, add + 1);
    return true;
}

// DLSS5_AMD_WEIGHTS, spelled the way the Win32 CRT can open it.  The packed weights live outside the game
// folder (r2scan.py scans the game folder, build/vk, build/shaders and build/package), so the installer
// hands their path over in the launch environment instead of copying them anywhere a scan would see them.
std::string ngxd_weights_root() {
    const char* env = std::getenv("DLSS5_AMD_WEIGHTS");
    if (!env || !*env) return {};
    const std::string raw(env);
    std::vector<std::string> tries{raw};
    if (raw[0] == '/' || raw[0] == '\\') tries.push_back("Z:" + raw);
    std::string slashed = raw;
    for (char& ch : slashed) if (ch == '\\') ch = '/';
    if (slashed != raw) tries.push_back(slashed);
    for (const std::string& t : tries)
        if (GetFileAttributesA(t.c_str()) != INVALID_FILE_ATTRIBUTES) return t;
    return raw;
}

static bool load_vk_entry_points() {
    if (g_gipa && g_gdpa && vkCreateBuffer) return true;
    if (!g_gipa) {
        for (const char* m : {"winevulkan.dll", "vulkan-1.dll"}) {
            HMODULE h = GetModuleHandleA(m);
            if (!h) h = LoadLibraryA(m);
            if (!h) continue;
            auto p = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(h, "vkGetInstanceProcAddr"));
            if (p) { g_gipa = p; break; }
        }
    }
    if (!g_gipa) return false;
    if (!g_gdpa) g_gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(g_gipa(VK_NULL_HANDLE, "vkGetDeviceProcAddr"));
    return g_gdpa != nullptr;
}

// Per-symbol GetProcAddress from a module, the way reference-nr-vulkan's nr_vk_load() does it: the
// D3D12 path never gets a gipa/gdpa (the host only passes those to NVSDK_NGX_VULKAN_Init), and
// vkd3d-proton's instance/device/command buffers are winevulkan's own objects -- Proton's
// vulkan-1.dll only forwards to winevulkan.dll, so either module hands back the same function
// pointers.  winevulkan is taken only when it exports everything; the winevulkan.dll *forwarder*
// exports a few symbols and leaves the rest on vulkan-1.dll.
static bool load_from_module() {
    HMODULE m = GetModuleHandleA("winevulkan.dll");
    const char* src = "winevulkan.dll";
    bool complete = (m != nullptr);
    if (m) {
#define NGXD_VK_CHECK(f) if (!GetProcAddress(m, #f)) { complete = false; }
        NGXD_VK_FUNCS(NGXD_VK_CHECK)
#undef NGXD_VK_CHECK
    }
    if (!complete) {
        m = GetModuleHandleA("vulkan-1.dll");
        if (!m) m = LoadLibraryA("vulkan-1.dll");
        src = "vulkan-1.dll";
    }
    if (!m) {
        ngx_log("no winevulkan.dll and no vulkan-1.dll in this process, and NVSDK_NGX_VULKAN_Init "
                "passed no gipa/gdpa");
        return false;
    }
    const char* missing = nullptr;
#define NGXD_VK_GET(f)                                                                     \
    if (!f) {                                                                              \
        f = reinterpret_cast<PFN_##f>(GetProcAddress(m, #f));                              \
        if (!f && !missing) missing = #f;                                                  \
    }
    NGXD_VK_FUNCS(NGXD_VK_GET)
#undef NGXD_VK_GET
    if (missing) {
        ngx_log("%s does not export %s", src, missing);
        return false;
    }
    g_vk_module = m;
    g_vk_loaded = true;
    ngx_log("vulkan entry points from %s", src);
    return true;
}

static bool load_vk_for_device(VkInstance inst, VkPhysicalDevice phys, VkDevice dev) {
    if (g_vk_loaded) return true;   // NVSDK_NGX_VULKAN_Init and build_chain both ask
    if (!inst || !dev) {
        ngx_log("no Vulkan entry points: NVSDK_NGX_VULKAN_Init must pass gipa/gdpa, or winevulkan must be loaded");
        return false;
    }
    // Route 1: the loader the Vulkan host handed us -- correct for the objects that loader made.
    if (g_gipa && g_gdpa && !g_vk_module) {
        bool ok = true;
#define NGXD_VK_LOAD(f)                                                                       \
    f = reinterpret_cast<PFN_##f>(g_gdpa(dev, #f));                                           \
    if (!f) { f = reinterpret_cast<PFN_##f>(g_gipa(inst, #f)); if (!f) { ok = false; } }
        NGXD_VK_FUNCS(NGXD_VK_LOAD)
#undef NGXD_VK_LOAD
        if (ok) {
            g_vk_loaded = true;
            ngx_log("vulkan entry points from the host's gipa/gdpa");
            return true;
        }
        ngx_log("the host's gipa/gdpa did not resolve every entry point, falling back to the module");
    }
    (void)phys;
    // Route 2: per-symbol GetProcAddress, which is the only route the D3D12 path has.
    return load_from_module();
}

// FeatureState is FeatureState_ (ngx_bridge_shared.h): the ffx surface shares the layout.  The
// NGX surface's extra typing comes from the member functions/kinds below, not from fields.
struct FeatureState : FeatureState_ {
    NVSDK_NGX_Feature NGXFeature() const { return static_cast<NVSDK_NGX_Feature>(feature); }
    void set_feature(NVSDK_NGX_Feature f) { feature = int(f); }
    bool is_rr() const { return feature == NVSDK_NGX_Feature_RayReconstruction; }
};

// BridgeImage = BridgeImage_ (ngx_bridge_shared.h), the shared layout; the NGX TU's name stays.
struct BridgeImage : BridgeImage_ {
};

// The depth formats this bridge now decodes, named so both the map below and the aspect choice in
// fetch_image come from one place.  DXGI 19 (R32G8X24_TYPELESS) is the measured in-game Depth of
// Crimson Desert: its refusal cost 63,760 refused frames per session ("Depth: DXGI format 19 is not
// one this bridge decodes").  vkd3d-proton resolves it through its depth-stencil table
// (libs/vkd3d/utils.c:156-161: DXGI 19/20 -> VK_FORMAT_D32_SFLOAT_S8_UINT, plane_count 2, which
// vkd3d_get_format prefers over the general table regardless of the depth_stencil argument).
// The DEPTH aspect of that image is D32_SFLOAT = 4 B/texel of raw float32 - byte for byte what
// sr_gather.comp:125-129 reads at code 0 (uintBitsToFloat) and what record() strides at
// pipeline.cpp:1852 (pdep = W * 4).
static inline bool vk_is_depth_stencil(VkFormat f) {
    return f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT;
}

static const char* feature_name(const FeatureState* s) {
    return s->feature == NVSDK_NGX_Feature_RayReconstruction ? "RR" : "SR";
}

// Create the chain: packed weights from DLSS5_AMD_WEIGHTS, shaders from <game>/dlss5-amd/shaders.
static bool build_chain(FeatureState* s) {
    if (!g_vk_device || !g_vk_phys_dev) {
        ngx_log("%s %ux%u: no VkDevice (NVSDK_NGX_VULKAN_Init / the vkd3d interop device never gave us one)",
                feature_name(s), s->rw, s->rh);
        return false;
    }
    if (!load_vk_for_device(g_vk_instance, g_vk_phys_dev, g_vk_device)) return false;
    const std::string root = ngxd_weights_root();
    if (root.empty()) {
        ngx_log("%s %ux%u: DLSS5_AMD_WEIGHTS is not set; the packed weights are kept outside the game folder "
                "on purpose, so the installer has to say where they are", feature_name(s), s->rw, s->rh);
        return false;
    }
    const bool rr = s->feature == NVSDK_NGX_Feature_RayReconstruction;
    const std::string w = root + (rr ? "/rr" : (s->preset == 'L' ? "/L" : "/M"));
    char shaders[MAX_PATH] = {0};
    if (!module_path_tail("dlss5-amd\\shaders", shaders, sizeof shaders))
        ngx_log("cannot work out the DLL's own directory");
    std::string err;
    const unsigned long long t0 = ngx_ms();
    ngx_log("%s %ux%u -> %ux%u: creating the frame chain from %s", feature_name(s), s->rw, s->rh, s->ow, s->oh,
            w.c_str());
    s->chain = ngxd::FrameChain::create(g_vk_device, g_vk_phys_dev, rr ? "rr" : "sr", std::string(1, s->preset),
                                        int(s->rw), int(s->rh), int(s->ow), int(s->oh), w, shaders, &err);
    if (!s->chain) {
        ngx_log("%s %ux%u -> %ux%u: no frame chain (%s)", feature_name(s), s->rw, s->rh, s->ow, s->oh, err.c_str());
        return false;
    }
    ngx_log("%s preset %c %ux%u -> %ux%u: chain ready in %llu ms, %zu pipelines, weights %s", feature_name(s),
            s->preset, s->rw, s->rh, s->ow, s->oh, ngx_ms() - t0, s->chain->pipeline_count(), w.c_str());
    return true;
}

// The formats the port's gather shaders decode: anything else is refused rather than reinterpreted.
// `alt` is the second format this slot now accepts (VK_FORMAT_UNDEFINED when there is only one) and
// `dxgi` is what the D3D12 caller named - the same VkFormat can come from more than one DXGI enum
// (or from a view the bridge never sees the DXGI for, on the Vulkan path), so a refusal prints both:
// "Depth: VkFormat 100 (DXGI 45), this bridge only carries 98" already says which of the two is at
// fault without anyone decoding VkFormat numbers by hand.
static bool want_format(const char* key, VkFormat got, VkFormat want, VkFormat alt, unsigned dxgi) {
    if (got == want || (alt != VK_FORMAT_UNDEFINED && got == alt)) return true;
    if (alt)
        ngx_log("%s: VkFormat %d (DXGI %u), this bridge only carries %d or %d", key, int(got), dxgi ? dxgi : 0,
                int(want), int(alt));
    else if (dxgi)
        ngx_log("%s: VkFormat %d (DXGI %u), this bridge only carries %d", key, int(got), dxgi, int(want));
    else
        ngx_log("%s: VkFormat %d, this bridge only carries %d", key, int(got), int(want));
    return false;
}

static bool read_frame_params(NVSDK_NGX_Parameter* p, const FeatureState* s, ngxd::FrameParams& fp) {
    float f = 0.0f;
    if (ngx_ok(p->vtbl->GetFloat(p, "Jitter.Offset.X", &f))) fp.jx = f;
    f = 0.0f;
    if (ngx_ok(p->vtbl->GetFloat(p, "Jitter.Offset.Y", &f))) fp.jy = f;
    fp.jpx = s->prev_jx;
    fp.jpy = s->prev_jy;
    int i = 0;
    if (ngx_ok(p->vtbl->GetInt(p, "Reset", &i))) fp.reset = i ? 1 : 0;
    f = 1.0f;
    if (ngx_ok(p->vtbl->GetFloat(p, "DLSS.Pre.Exposure", &f))) fp.pre_exposure = f;
    f = 1.0f;
    if (ngx_ok(p->vtbl->GetFloat(p, "DLSS.Exposure.Scale", &f))) fp.exposure = f;
    // S5 (master-2 Q3#2, 2026-10-31 evening): the CANONICAL keys are MV.Scale.X / MV.Scale.Y - the
    // harness writes exactly those (opti_client.cpp:1048-1056, from the fixture's mv_scale pair) and
    // sl.dlss.dll carries the strings while the GAME exe has none (SL is the writer).  The line
    // below historically read a key nobody ever sets, which is why every d5 diag prints the primed
    // 1.0000 - plumbing bug proven by master-2.
    // S6 (in-game session 22:32, 145 MB log): SL writes X = 1280.0 (== the MV plane width) and
    // Y = -720.0 (== NEGATIVE plane height) = UV-normalized MVs, 1.0 = full plane, Y mirrored.
    // The port CONSUMES the pair from here on: the chain gathers mv *= scale (shaders/sr/
    // sr_gather*.comp), so pixel-unit apps see the unchanged identity and UV apps land on the
    // chain's plane-pixel convention (prev = cur + mv, tools/scenes/check_mv.py).  Per-axis reads,
    // legacy 'MV.Scale' spelling as the X fallback; each axis independent (an app may set only one
    // with the Y unit, measured: X only on some sessions).  Guard = DlssNr_Pipeline_Dx12.cpp:
    // 123-128: non-finite or exactly 0 -> 1.0 (a dead key must not scale MVs to 0).
    f = 1.0f;
    if (ngx_ok(p->vtbl->GetFloat(p, "MV.Scale.X", &f)) ||
        ngx_ok(p->vtbl->GetFloat(p, "MV.Scale", &f)))
        fp.mv_scale_x = f;
    f = 1.0f;
    if (ngx_ok(p->vtbl->GetFloat(p, "MV.Scale.Y", &f)))
        fp.mv_scale_y = f;
    if (!std::isfinite(fp.mv_scale_x) || fp.mv_scale_x == 0.0f) fp.mv_scale_x = 1.0f;
    if (!std::isfinite(fp.mv_scale_y) || fp.mv_scale_y == 0.0f) fp.mv_scale_y = 1.0f;
    // One observability bench-mark per feature: the first frame whose scale is not the identity
    // names the exact in-game pair (legacy: d5 diag was the only record, at 1 Hz partly lost in
    // the 145 MB stream).
    if (!s->s6_logged && (fp.mv_scale_x != 1.0f || fp.mv_scale_y != 1.0f)) {
        s->s6_logged = 1;
        ngx_log("[S6] mv scale=(%g,%g) (MV.Scale.X/.Y, UV-normalized when X==planeW && Y==-planeH)",
                fp.mv_scale_x, fp.mv_scale_y);
    }
    float sharp = 0.0f;
    int sharp_set = 0;
    if (ngx_ok(p->vtbl->GetFloat(p, "Sharpness", &sharp))) sharp_set = 1;
    // D5 (H2/H3 triage, 2026-10-31, 1 Hz): jitter pair + MV.Scale + reset/exposure on the record.
    // H2 asks whether the app ever delivers jitter (SL's "cameraPinholeOffset invalid" would leave
    // jx/jy at 0 forever); H3 asks what MV.Scale actually is (S5 read it, S6 consumes it).
    static unsigned long long d5_last_ms = 0;
    const unsigned long long d5_now = ngx_ms();
    if (d5_now - d5_last_ms >= 1000u) {
        d5_last_ms = d5_now;
        ngx_log("d5 diag: jx=%.6f jy=%.6f prev=(%.6f,%.6f) mv_scale=%.4f mvY=%.4f sharp=%.3f(+%d) "
                "reset=%d pre_exp=%.4f [diag-d5]",
                fp.jx, fp.jy, fp.jpx, fp.jpy, fp.mv_scale_x, fp.mv_scale_y, sharp, sharp_set,
                fp.reset, fp.pre_exposure);
    }
    // The chain works on whole frames.  A subrect that is not the whole render target would need the
    // gather to read a window of the image, which the port does not have; refuse instead of guessing.
    int sw = int(s->rw), sh = int(s->rh);
    if (ngx_ok(p->vtbl->GetInt(p, "DLSS.Render.Subrect.Dimensions.Width", &i))) sw = i;
    if (ngx_ok(p->vtbl->GetInt(p, "DLSS.Render.Subrect.Dimensions.Height", &i))) sh = i;
    if (sw != int(s->rw) || sh != int(s->rh)) {
        ngx_log("subrect %dx%d is not the whole %ux%u render target: unsupported", sw, sh, s->rw, s->rh);
        return false;
    }
    // Code 0 is what the gather shaders mean by "the default texel": RGBA16F colour, RG16F motion
    // vectors (4 B/px), R32F depth (texfmt.py:6-11; sr_gather.comp:111/114/124), and it is the code
    // ngx_client's Input table feeds (ngx_client/main.cpp:86-90, code0 column), i.e. the measured
    // 57-59 dB path.  fmt_mv is then *re-derived from the MV resource actually fetched*, in
    // collect_frame below, because the in-game MotionVectors are not RG16F (see the Slot table);
    // fmt_col/fmt_dep stay 0 here - both are the only code this bridge carries for RGBA16F/R32F.
    // MV.Scale X/Y ARE now read (above) and consumed by the chain's gather decode (S6).
    fp.fmt_col = fp.fmt_mv = fp.fmt_dep = 0;
    return true;
}

static void image_barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout old_l, VkImageLayout new_l,
                          VkAccessFlags src, VkAccessFlags dst,
                          VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = old_l;
    b.newLayout = new_l;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b);
}

// Host-visible scratch for the DLSS5_AMD_DUMP image probe: two 40-px columns of the app image,
// left and right.  One small buffer for the life of the process, only ever used with DLSS5_AMD_DUMP
// set (an env-gated debug path, so nothing here runs in a game unless asked).
static VkBuffer g_diag_buf = VK_NULL_HANDLE;
static VkDeviceMemory g_diag_mem = VK_NULL_HANDLE;
static bool ensure_diag_slice_buffer() {
    if (g_diag_buf) return true;
    if (!g_vk_device || !g_vk_phys_dev) return false;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = 2ull * 40ull * 4096ull * 8ull + 4ull * 4096ull * 4ull * 8ull;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(g_vk_device, &bci, nullptr, &g_diag_buf) != VK_SUCCESS) {
        g_diag_buf = VK_NULL_HANDLE;
        return false;
    }
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_vk_device, g_diag_buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_vk_phys_dev, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (req.memoryTypeBits & (1u << i))) { type = i; break; }
    if (type == UINT32_MAX) { g_diag_buf = VK_NULL_HANDLE; return false; }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (vkAllocateMemory(g_vk_device, &mai, nullptr, &g_diag_mem) != VK_SUCCESS) {
        g_diag_buf = VK_NULL_HANDLE;
        return false;
    }
    return vkBindBufferMemory(g_vk_device, g_diag_buf, g_diag_mem, 0) == VK_SUCCESS;
}

// Report the two slices recorded by the probe above (called while recording the next frame, i.e.
// after the frame-0 command buffer that copied them has executed).
static void report_diag_slices(const BridgeImage& out, int frame) {
    if (!g_diag_buf) return;
    void* p = nullptr;
    if (vkMapMemory(g_vk_device, g_diag_mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) return;
    const uint16_t* h = static_cast<const uint16_t*>(p);
    const size_t n = size_t(40) * out.h * 4;    // 40 px x H rows x 4 channels (RGBA16F)
    long nz_l = 0, nz_r = 0;
    for (size_t i = 0; i < n; ++i) { if (h[i]) ++nz_l; if (h[n + i]) ++nz_r; }
    // the strip: last nonzero pixel column of image row 0 and row 1, at the image's own pitch
    const uint16_t* strip = h + 2 * n;
    int last4k[2] = {-1, -1};
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < int(out.w); ++x)
            if (strip[y * out.w * 4 + x * 4] || strip[y * out.w * 4 + x * 4 + 1] ||
                strip[y * out.w * 4 + x * 4 + 2] || strip[y * out.w * 4 + x * 4 + 3])
                last4k[y] = x;
    vkUnmapMemory(g_vk_device, g_diag_mem);
    ngx_log("frame %d: image after copy-back: left40 nz=%ld/%zu right40 nz=%ld/%zu row0 columns "
            "0..%d", frame, nz_l, n, nz_r, n, last4k[0]);
}

// Bytes per texel of the VkFormats the input slots accept.  Used only to reconcile row strides
// below; anything unlisted is 0 = "unknown", and an unknown stride is left exactly as it was copied.
static unsigned vk_format_bpp(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        case VK_FORMAT_R32G32_SFLOAT:       return 8;
        case VK_FORMAT_R16G16_SFLOAT:       return 4;
        case VK_FORMAT_R32_SFLOAT:         return 4;
        case VK_FORMAT_R8G8B8A8_UNORM:     return 4;   // RR code 5, kRrBppE[5] = 4
        // The depth-aspect texel the copy actually writes (D32_SFLOAT = 4 B), NOT the image's
        // combined depth+stencil texel: VkCmdCopyImageToBuffer with aspectMask DEPTH strides in
        // aspect texels, and the chain reads dep at W * 4 (pipeline.cpp:1852).
        case VK_FORMAT_D32_SFLOAT_S8_UINT:  return 4;
        default: return 0;
    }
}

// The row stride, in bytes, that record() reads one gather input at: the very formulas the chain
// pushes as pitch_* constants, mirrored here so the copies below can pad rows to them.  SR:
//   pipeline.cpp:1846 sr_col_bpp table {8,4,4,4,4,16,4}, 1850 pcol = W * sr_col_bpp(fmt_col),
//   1849 sr_mv_bpp = (fmt_mv == 0) ? 4 : 8 with 1851 pmv = W * that, 1852 pdep = W * 4.
// RR: pipeline.cpp:2251 pitch(code) = W * kRrBpp[code] (the same table as texfmt.py RR_BPP).
// Both divide out W - the dispatch width (pipeline.cpp:1862/2261, W = render width, NOT oW; the
// oW pitch at 1868 belongs to sr_gather_mv, which runs when mvhi is set - collect_frame sets
// fp.mvhi = 1 for an output-sized MotionVectors image, and the FFX surface derives the same flag
// from its dispatch's dimensions, so BOTH bridges can now reach that path).  Returns 0 for a slot
// whose code is not mirrored here, so the copy falls back to the image's own packing instead of
// guessing.
static unsigned chain_row_bytes(const ngxd::FrameChain* c, bool rr, const ngxd::FrameParams& fp,
                                const char* name) {
    if (!c) return 0;
    const unsigned W = unsigned(c->width());
    if (rr) {
        static const int kBpp[24] = {8, 4, 4, 4, 2, 4, 4, 4, 1, 2, 8, 16, 4, 4,
                                      8, 8, 4, 4, 4, 4, 2, 2, 2, 1};   // pipeline.cpp:2251
        int code = -1;
        if (!std::strcmp(name, "Color"))              code = fp.fmt_col;
        else if (!std::strcmp(name, "MotionVectors")) code = fp.fmt_mv;
        else if (!std::strcmp(name, "Depth"))         code = fp.fmt_dep;
        else if (!std::strcmp(name, "GBuffer.Normals")) code = fp.fmt_nrm;
        else if (!std::strcmp(name, "DiffuseAlbedo")) code = fp.fmt_alb;
        else if (!std::strcmp(name, "SpecularAlbedo")) code = fp.fmt_spa;
        else if (!std::strcmp(name, "Roughness"))     code = fp.fmt_rgh;
        return (code >= 0 && code < 24) ? W * unsigned(kBpp[code]) : 0u;
    }
    if (!std::strcmp(name, "Color")) {
        static const int kCol[7] = {8, 4, 4, 4, 4, 16, 4};             // pipeline.cpp:1846
        return W * unsigned((fp.fmt_col >= 0 && fp.fmt_col < 7) ? kCol[fp.fmt_col] : 4);
    }
    if (!std::strcmp(name, "MotionVectors")) return W * ((fp.fmt_mv == 0) ? 4u : 8u);   // 1849/1851
    if (!std::strcmp(name, "Depth")) return W * 4u;                                    // 1852
    return 0u;
}

// The bufferRowLength that makes one input's rows land exactly on the chain's row stride, or 0 to
// leave the copy as it has always been.  A stride only matches when two things hold:
//   - the chain's per-texel size for this slot's code is the image's own byte count
//     (chain_row_bytes == W * image bpp).  Element SPACING is decided by the gather shader
//     (sr_gather.comp:112 x*4 / :115 and :118 x*8, rr_gather.comp:81 x*rr_bpp(code)), and no
//     VkBufferImageCopy can move texels inside a row - where those disagree (RR still reads mv and
//     dep as read_frame_params' code 0 = 8 B/texel while ngx_client's kRrInputs hand 4 B/px
//     images with code0 2 and 3, ngx_client/main.cpp:93-94), striding is the wrong tool and the
//     copy stays untouched,
//     named on the frame-1 pitch line instead;
//   - the image is not wider than W rows: bufferRowLength must be 0 or >= imageExtent.width, so a
//     wider image cannot be padded down (that refusal is MotionVectors' own check in collect_frame);
//   - the image is not taller than H: padding widens every row of an oversized copy
//     (chain_row > image width * bpp), so it may only shrink a copy's footprint, never grow one -
//     in_dep is exactly W*H*4 (pipeline.cpp:966) and a row past H would overrun it.
// A narrower image is padded up to W texels, which is what gives every row a right start.
static unsigned copy_row_texels(const ngxd::FrameChain* c, bool rr, const ngxd::FrameParams& fp,
                                const BridgeImage& img) {
    if (!c) return 0u;
    const unsigned W = unsigned(c->width());
    const unsigned bpp = vk_format_bpp(img.format);
    if (!bpp) return 0u;
    if (chain_row_bytes(c, rr, fp, img.name) != W * bpp) return 0u;
    if (img.w > W || img.h > unsigned(c->height())) return 0u;
    return W;
}

// DLSS5_AMD_DUMP, by environment OR by a marker file next to this DLL.  A game launch cannot gain
// a new environment variable without editing the Steam launch options, but dropping a
// "dlss5-amd-dump.enable" file into the game's bin64 arms exactly the same probe (delete the file
// to disarm; environment wins if both are present).  The probe itself stays bounded: frames 0..6
// of one feature, two 40-px slices of the app's output image per frame plus one chain self-dump.
static bool dump_probe_on() {
    static const bool on = [] {
        if (std::getenv("DLSS5_AMD_DUMP")) return true;
        char p[MAX_PATH] = {0};
        if (!module_path_tail("dlss5-amd-dump.enable", p, sizeof p)) return false;
        if (FILE* f = std::fopen(p, "rb")) { std::fclose(f); return true; }
        return false;
    }();
    return on;
}

// Record one frame: move the caller's images into the chain's staging buffers, run the chain, move the
// result back.  A copy has to happen in TRANSFER_*_OPTIMAL or GENERAL, so every image is transitioned
// there and back to whatever layout the caller left it in -- the ngx_client contract is GENERAL in,
// GENERAL out, vkd3d-proton's is SHADER_READ_ONLY / UNORDERED_ACCESS in, the same out.
static NVSDK_NGX_Result record_bridge(VkCommandBuffer cmd, FeatureState* s, const ngxd::FrameParams& fp,
                                      const BridgeImage* ins, size_t n, const BridgeImage& out) {
    ngxd::FrameChain* c = s->chain;
    const VkAccessFlags readable = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    const unsigned long long t0 = ngx_ms();
    // On every frame after the first, the staging buffers still hold what the previous frame left: the
    // host-visible input (did the image->buffer copy land?) and the host-visible output (did the chain
    // write it?).  Both are zeroes on a freshly created chain, so only frames > 0 say anything.
    //
    // The probe reads 2*2^20 host floats, and the staging buffers live in the ReBAR window
    // (DEVICE_LOCAL|HOST_VISIBLE, never HOST_CACHED in make_buffer), so every load is an uncached PCIe
    // transaction -- about 1.3 s per frame, i.e. ~0.7 fps at 100% CPU and an idle GPU.  It is
    // verification-only, so it now costs nothing unless DLSS5_AMD_PROBE asks for it; the frame count and
    // timing line below still log every frame either way.
    static const bool probe = std::getenv("DLSS5_AMD_PROBE") != nullptr;
    if (probe && s->frame_idx > 0) {
        const float* in = static_cast<const float*>(c->in_color());
        long nz_in = 0, nz_out = 0;
        const int N = 1 << 20;
        for (int i = 0; i < N && i < int(c->out_width()) * c->out_height() * 4; ++i) nz_in += (in[i] != 0.0f);
#if DLSS5_AMD_OUT_F16
        // out is RGBA16F now: count non-zero halfwords (0x8000 = -0 is zero too) and widen the
        // first four for the log.
        const uint16_t* oh = static_cast<const uint16_t*>(c->out_color());
        for (int i = 0; i < N && i < int(c->out_width()) * c->out_height() * 4; ++i)
            nz_out += (oh[i] != 0 && oh[i] != 0x8000);
        ngx_log("%s frame %d probe: previous input %ld/%d non-zero in the first %d floats, previous chain "
                "output %ld/%d, out[0..3] = %g %g %g %g",
                feature_name(s), s->frame_idx, nz_in, N, N, nz_out, N,
                ngxd::dlss_half_to_float(oh[0]), ngxd::dlss_half_to_float(oh[1]),
                ngxd::dlss_half_to_float(oh[2]), ngxd::dlss_half_to_float(oh[3]));
#else
        const float* o = static_cast<const float*>(c->out_color());
        for (int i = 0; i < N; ++i) nz_out += (o[i] != 0.0f);
        ngx_log("%s frame %d probe: previous input %ld/%d non-zero in the first %d floats, previous chain "
                "output %ld/%d, out[0..3] = %g %g %g %g",
                feature_name(s), s->frame_idx, nz_in, N, N, nz_out, N, o[0], o[1], o[2], o[3]);
#endif
    }
    const bool rr = s->feature == NVSDK_NGX_Feature_RayReconstruction;
    for (size_t k = 0; k < n; ++k) {
        image_barrier(cmd, ins[k].image, ins[k].layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readable,
                      VK_ACCESS_TRANSFER_READ_BIT, ins[k].aspect);
        VkBufferImageCopy cp{};
        // The aspect the barrier above moved: DEPTH for a vkd3d depth-stencil image (VUID-
        // vkCmdCopyImageToBuffer-*: imageAspect must match the format's aspect), COLOR otherwise.
        cp.imageSubresource = {ins[k].aspect, 0, 0, 1};
        cp.imageExtent = {ins[k].w, ins[k].h, 1};
        // The copy pitch trap: with bufferRowLength = 0 vkCmdCopyImageToBuffer packs rows at the
        // IMAGE width, while record() reads rows at chain_row_bytes() = W * bpp of this slot's code
        // (pipeline.cpp:1851, pmv = W * 8 once fmt_mv is the RG32F code).  When the image is not
        // W texels wide those two strides differ and every row lands shifted - the in-game MV being
        // 1280x720 against a 1280x720 render target hides it, it does not remove it.  The helper
        // answers the row length in TEXELS that reproduces the chain's stride byte for byte, or 0
        // to leave the copy untouched; for an image already W texels wide (the harness path, which
        // this must not change a byte) the two spellings are the same number.
        cp.bufferRowLength = copy_row_texels(c, rr, fp, ins[k]);
        vkCmdCopyImageToBuffer(cmd, ins[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ins[k].buf, 1, &cp);
        image_barrier(cmd, ins[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ins[k].layout,
                      VK_ACCESS_TRANSFER_READ_BIT, readable, ins[k].aspect);
    }
    if (n) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);
    }

    c->record(cmd, fp);

    {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             1, &mb, 0, nullptr, 0, nullptr);
    }
    image_barrier(cmd, out.image, out.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, readable,
                  VK_ACCESS_TRANSFER_WRITE_BIT);
    {
        VkBufferImageCopy cp{};
        cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cp.imageExtent = {out.w, out.h, 1};
        vkCmdCopyBufferToImage(cmd, c->out_color_buf(), out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
    }
    image_barrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, out.layout, VK_ACCESS_TRANSFER_WRITE_BIT,
                  readable);

    // DLSS5_AMD_DUMP: copy two 40-px slices of the app image back out -- far left (x 0..39) and far
    // right (x 3800..3839) -- in the very same command buffer, then report their contents while
    // recording frame 1.  The chain's out buffer is already known to be complete and bit-exact (the
    // frame-0 dump proves that), so anything missing in these slices is missing between
    // vkCmdCopyBufferToImage and the app, with no further suspects in between.
    if (dump_probe_on() && s->frame_idx <= 5 && ensure_diag_slice_buffer()) {
        image_barrier(cmd, out.image, out.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readable,
                      VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy sl[3]{};
        for (int i = 0; i < 2; ++i) {
            sl[i].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            sl[i].imageOffset = {i ? int(out.w - 40) : 0, 0, 0};
            sl[i].imageExtent = {40, out.h, 1};
            sl[i].bufferRowLength = 40;
            sl[i].bufferOffset = uint64_t(i) * 40ull * out.h * 8ull;
        }
        // A full-width strip (4 rows), read at exactly the coordinates the app's own read-back
        // uses: if this says 0..3839 while out_<i>.f32 says 0..959, the disagreement is between two
        // different images (or two different times), not inside one copy.
        const uint64_t strip_off = 2ull * 40ull * out.h * 8ull;
        sl[2].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        sl[2].imageOffset = {0, 0, 0};
        sl[2].imageExtent = {out.w, 4, 1};
        sl[2].bufferRowLength = out.w;
        sl[2].bufferOffset = strip_off;
        vkCmdCopyImageToBuffer(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_diag_buf, 3, sl);
        image_barrier(cmd, out.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.layout, VK_ACCESS_TRANSFER_READ_BIT,
                      readable);
    }

    c->advance();
    s->frame_idx++;
    // DLSS5_AMD_DUMP=1: once, while recording frame 1, the chain's own buffers still hold frame 0
    // (the caller's frame-0 command buffer has been submitted or is ahead of this record), so both
    // can be written out and diffed against the native runner's out_<i>.f32 and the cell's raw
    // inputs.  Without a host-side dump the bridge's two image<->buffer copies are only observable
    // through the app's read-back, where a quarter-wide result cannot be attributed.
    // The increment above means this block runs at the *end* of recording frame N; frame 0's
    // command buffer has only been submitted and executed by the time frame 1 is being recorded,
    // so the dump waits for frame_idx == 2 (record of frame 1 = frame 0 in the buffer).
    if (s->frame_idx == 2 && dump_probe_on()) {
        if (FILE* fo = std::fopen("dlss5-amd-dump_out.f32", "wb")) {
            const uint16_t* h16 = static_cast<const uint16_t*>(c->out_color());
            const size_t npx = size_t(c->out_width()) * c->out_height();
            std::vector<float> wide(npx * 4);
            for (size_t i = 0; i < npx; ++i)
                for (int k = 0; k < 4; ++k) wide[i * 4 + k] = ngxd::dlss_half_to_float(h16[i * 4 + k]);
            std::fwrite(wide.data(), 4, wide.size(), fo);
            std::fclose(fo);
        }
        if (FILE* fi = std::fopen("dlss5-amd-dump_in.bin", "wb")) {
            std::fwrite(c->in_color(), 1,
                        size_t(c->width()) * c->height() * 16, fi);   // the chain's color pitch is 16 B/px
            std::fclose(fi);
        }
        // S3 (SR-quality hunt, 2026-10-31 evening): every other artifact (shaders/weights/formats/
        // jitter/geometry) is byte-identical to the good-era archives, and D1 A06 PASSED the exact
        // in-game geometry headless - so the untested input is the MV/Depth CONTENT VERBATIM.  The
        // eval pitch lines give the planes: MotionVectors 1280x720+8B chain 10240B=1280 tx (8 B/px,
        // in-game fmt mv=1 = RG32F), Depth ...+4B chain 5120B=1280 tx (4 B/px).  Staging is
        // host-visible and allocated *16 - writing 8/4 B/px is in-bounds (pipeline.h:122-127).
        if (FILE* fm = std::fopen("dlss5-amd-dump_mv.bin", "wb")) {
            std::fwrite(c->in_mv(), 1, size_t(c->width()) * c->height() * 8, fm);
            std::fclose(fm);
        }
        if (FILE* fd = std::fopen("dlss5-amd-dump_dep.bin", "wb")) {
            std::fwrite(c->in_depth(), 1, size_t(c->width()) * c->height() * 4, fd);
            std::fclose(fd);
        }
        ngx_log("dumped chain out/in of frame 0 (out %ux%u, in %dx%d) + mv/dep planes",
                c->out_width(), c->out_height(), c->width(), c->height());
    }
    // S7 FLASH hunt (2026-11-01): the user reports "frequently flashes the whole screen bright".
    // Leading port-side suspect (read-only analysis ses_e6ab37de): sr_downsample.comp feeds the
    // OUT plane's alpha as a nearest-neighbor fetch of a low-res input alpha, unsanitized and
    // history-free.  If the game composites out-alpha, junk frames READ as fullscreen flashes.
    // Instrument: every ~300 frames (and each of the S6 dump frames), tag frames where out
    // min/max deviate wildly from its running median (RGB) OR where the ALPHA plane's max
    // diverges from its own running median - the log line is the evidence, no extra IO.
    //
    // COST, measured in-game 2026-11-01: a FULL read of the 4K output (33.2M halfwords of
    // DEVICE_LOCAL|HOST_VISIBLE staging = uncached ReBAR, ~600 ns/load) took ~47 s PER FRAME -
    // "recorded in 47097 ms" on every eval while the block's log gate stayed dead (the FFX
    // surface double-counted frame_idx, so the counter was odd and never matched 10/60/300...).
    // A fullscreen flash moves every pixel, so a spatial stride of ~4096 samples detects it
    // exactly as well for ~16k loads (~10 ms); what sparse sampling can miss is a sub-screen
    // highlight, which is not this hunt's target.
    if (dump_probe_on() && s->frame_idx >= 3) {
        const unsigned oW = c->out_width(), oH = c->out_height();
        const size_t npx = size_t(oW) * oH;
        const uint16_t* h16 = static_cast<const uint16_t*>(c->out_color());
        float rgb_med = 0.0f, a_max = 0.0f, a_med_candidate = 0.0f;   // a_med filled below from history
        (void)a_med_candidate;
        float rs = 0.f, gs = 0.f, bs = 0.f;
        const size_t stride = (npx > 4096) ? npx / 4096 : 1;
        size_t ns = 0;
        for (size_t i = 0; i < npx; i += stride, ++ns) {
            const float r = ngxd::dlss_half_to_float(h16[i * 4 + 0]);
            const float g = ngxd::dlss_half_to_float(h16[i * 4 + 1]);
            const float b = ngxd::dlss_half_to_float(h16[i * 4 + 2]);
            const float a = ngxd::dlss_half_to_float(h16[i * 4 + 3]);
            rs += r; gs += g; bs += b;
            if (a > a_max) a_max = a;
        }
        const float rmean = float(rs) / ns, gmean = float(gs) / ns, bmean = float(bs) / ns;
        static std::vector<float> rgb_hist, a_hist;     // running mean history, running alpha-max
        size_t rm = rgb_hist.size();
        if (rm >= 6) {
            std::vector<float> rh(rm), ah(rm);
            for (size_t i = 0; i < rm; ++i) { rh[i] = rgb_hist[i]; ah[i] = a_hist[i]; }
            std::nth_element(rh.begin(), rh.begin() + rm / 2, rh.end());
            std::nth_element(ah.begin(), ah.begin() + rm / 2, ah.end());
            rgb_med = rh[rm / 2];
            a_med_candidate = ah[rm / 2];
        }
        rgb_hist.push_back((rmean + gmean + bmean) * 0.3333f);
        a_hist.push_back(a_max);
        const float rgb_now = (rmean + gmean + bmean) * 0.3333f;
        const bool rgb_dev = (rgb_med > 0.f) && (std::fabs(rgb_now - rgb_med) > 0.35f * rgb_med + 0.05f);
        const bool a_dev = (a_med_candidate > 0.f) && (std::fabs(a_max - a_med_candidate) > 0.35f * a_med_candidate + 0.05f);
        if ((s->frame_idx % 300) == 0 || s->frame_idx == 10 || s->frame_idx == 60 ||
            s->frame_idx == 300 || s->frame_idx == 900 || s->frame_idx == 1800 || rgb_dev || a_dev)
            ngx_log("S7 flash tick %u: rgbmean=%.4f med=%.4f%s a_max=%.4f a_med=%.4f%s [S7]",
                    s->frame_idx, rgb_now, rgb_med, rgb_dev ? " RGB-DEVIANT" : "",
                    a_max, a_med_candidate, a_dev ? " ALPHA-DEVIANT" : "");
    }
    // S4 (next link of the SR-quality hunt): the frame-2 snapshot above caught the MV plane at
    // the menu (86.9% non-finite channel words; Depth clean).  Sample LATER frames - 10 and 60,
    // well into gameplay - so "sentinel only in menu frames" vs "sentinel in gameplay too" is
    // settled with two more files, and put the finite fraction straight into the port log so the
    // answer is visible without the dump.  The staging still holds the PRE-shader (raw) bytes,
    // so this measures the game's content, not our san_mv() guard.
    // S6 (2026-10-31 night): frames 300/900/1800 under the SAME dump pattern, adding magnitude
    // stats over the same staging: maxAbs and the fraction of channel words with |v| > 0.02.
    // SL's UV-normalized vectors (MV.Scale.X = plane width) in real pans should read maxAbs
    // ~1e-2..1e-1 after unit conversion (i.e. raw ~3e-5 at most screens' widths) while a broken
    // SL MV transfer stays pinned near 1e-5 - the S6 f10/f60 data answered "finite" but not
    // "moving"; these frames decide the UV-units hypothesis over minutes of gameplay.
    if ((s->frame_idx == 10 || s->frame_idx == 60 || s->frame_idx == 300 || s->frame_idx == 900 ||
         s->frame_idx == 1800) && dump_probe_on()) {
        const size_t nfl = size_t(c->width()) * c->height() * 2;   // 2 channels, float32
        const float* mv = static_cast<const float*>(c->in_mv());
        size_t fin = 0, big = 0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < nfl; ++i) {
            const float v = mv[i];
            fin += std::isfinite(v) ? 1u : 0u;
            const float a = std::fabs(v);
            if (a > max_abs) max_abs = a;
            big += a > 0.02f ? 1u : 0u;
        }
        char fn[64];
        std::snprintf(fn, sizeof fn, "dlss5-amd-dump_mv_f%u.bin", s->frame_idx);
        if (FILE* fm = std::fopen(fn, "wb")) { std::fwrite(mv, 4, nfl, fm); std::fclose(fm); }
        // S7 (2026-11-01): the S6 frames answered "no motion" (99.9% exact zeros during pans).
        // ORDERING-vs-NEVER discriminator: diff the plane against the PREVIOUS dumped frame's
        // bytes in RAM (no extra IO).  changed>0 at consecutive sampled frames = the plane DOES
        // get new content between evals but arrives stale/just-in-time (ordering, B: fix by
        // coping one frame late); changed==0 across minutes = the game never writes THIS plane
        // in DLSS mode (C: the MVs go somewhere SL does not hand us - game-side).
        static std::vector<unsigned char> prev_mv;   // raw bytes of the last dumped frame
        size_t changedpx = 0;
        if (prev_mv.size() == nfl * 4) {
            const unsigned char* now = static_cast<const unsigned char*>(c->in_mv());
            for (size_t pxi = 0; pxi < c->width() * c->height(); ++pxi) {
                unsigned anth = 0;
                for (size_t b = 0; b < 8; ++b) anth |= unsigned(prev_mv[pxi * 8 + b] ^ now[pxi * 8 + b]);
                if (anth) ++changedpx;
            }
        }
        prev_mv.assign(static_cast<const unsigned char*>(c->in_mv()),
                       static_cast<const unsigned char*>(c->in_mv()) + nfl * 4);
        if (s->frame_idx == 300 || s->frame_idx == 900 || s->frame_idx == 1800)
            ngx_log("S6 mv frame %u: maxAbs=%.6g |v|>0.02: %zu/%zu (%.3f%%) finite %.1f%% "
                    "delta: %zu px changed vs prev dump -> %s [S6]",
                    s->frame_idx, max_abs, big, nfl, 100.0 * (double)big / (double)nfl,
                    100.0 * (double)fin / (double)nfl, changedpx, fn);
        else
            ngx_log("S4 mv frame %u: %zu/%zu channel words finite (%.1f%%) -> %s [S4]",
                    s->frame_idx, fin, nfl, 100.0 * (double)fin / (double)nfl, fn);
    }
    // S7 ticker (2026-11-01): stats-only sample every ~600 frames - no disk IO, one pass.
    // maxAbs + nonzero fraction + delta-vs-last-sample in the log line itself; decides
    // "plane ever changes in gameplay" at high frequency without waiting for the 300/900/1800
    // instants. Statics: last sample bytes for delta, frame of the last sample.
    if (s->frame_idx >= 2000 && (s->frame_idx % 600) == 0 && dump_probe_on()) {
        const size_t nfl = size_t(c->width()) * c->height() * 2;
        const float* mv = static_cast<const float*>(c->in_mv());
        float max_abs = 0.0f;
        size_t nz = 0;
        for (size_t i = 0; i < nfl; ++i) {
            const float a = std::fabs(mv[i]);
            if (a > max_abs) max_abs = a;
            nz += (mv[i] != 0.0f) ? 1u : 0u;
        }
        static std::vector<float> prev_s7;
        size_t dp = 0;
        if (prev_s7.size() == nfl) {
            for (size_t p = 0; p < c->width() * c->height(); ++p)
                if (prev_s7[p * 2] != mv[p * 2] || prev_s7[p * 2 + 1] != mv[p * 2 + 1]) ++dp;
        }
        prev_s7.assign(mv, mv + nfl);
        ngx_log("S7 mv tick %u: maxAbs=%.6g nonzero %zu/%zu (%.3f%%) delta %zu px [S7]",
                s->frame_idx, max_abs, nz, nfl, 100.0 * (double)nz / (double)nfl, dp);
    }
    // One-time geometry of what the caller actually handed over: a quarter-wide input image would
    // show up here as 480x1080 while the frame line below still says 1920x1080 (that line is the
    // chain's own size, not the resource's).
    if (s->frame_idx == 1) {
        char dims[8 * 48] = "";
        for (size_t k = 0; k < n; ++k)
            std::snprintf(dims + std::strlen(dims), sizeof dims - std::strlen(dims),
                          " in%zu=%s %ux%u", k, ins[k].name, ins[k].w, ins[k].h);
        // desc.Width is the D3D12 resource, not the VkImage the copy actually lands in: a vkd3d image
        // smaller than the extent copied over it is not validated away on RADV, it is just clipped.
        // memoryRequirements.size is the only honest handle on the real image size (tiled RGBA16F:
        // ~8.4 MB at 1920x1080 vs ~33.5 MB at 3840x2160).
        VkMemoryRequirements mr_out{}, mr_in0{};
        if (out.image) vkGetImageMemoryRequirements(g_vk_device, out.image, &mr_out);
        if (n && ins[0].image) vkGetImageMemoryRequirements(g_vk_device, ins[0].image, &mr_in0);
        // diag-v1: id/eval say which feature, which call; the out VkImage and the chain's own
        // out_width()/out_height() say whether the 0x0 on the next line is the resource, the
        // chain, or a null handle whose image bytes read back as zero.
        ngx_log("caller geometry (id=%u, eval #%llu):%s out=%ux%u VkImage=%p (chain %dx%d, out %dx%d) "
                "image bytes: out=%llu in0=%llu",
                s->id, s->eval_id, dims, out.w, out.h, (void*)out.image, c->width(), c->height(),
                c->out_width(), c->out_height(),
                (unsigned long long)mr_out.size, (unsigned long long)mr_in0.size);
        // Once per feature, beside that line: what each slot's rows actually cost, image side vs
        // chain side (the two formulas chain_row_bytes mirrors).  Formats cannot change after
        // CreateFeature, and every slot that outright refuses (a DXGI enum dxgi_to_vk will not
        // decode, a VkFormat want_format does not carry) never reaches this point - so this is the
        // one line that shows the surviving slot the chain will read at the wrong stride: bytes per
        // texel the code does not match, or a resource whose width is not the chain's W.
        char rows[8 * 80] = "";
        const unsigned W = unsigned(c->width());
        const unsigned H = unsigned(c->height());
        bool stuck = false, padded = false;
        for (size_t k = 0; k < n; ++k) {
            const unsigned img_bpp = vk_format_bpp(ins[k].format);
            const unsigned row_bytes = chain_row_bytes(c, rr, fp, ins[k].name);
            const unsigned row_texels = copy_row_texels(c, rr, fp, ins[k]);
            const bool bpp_ok = (img_bpp != 0) && (row_bytes == W * img_bpp);
            // Aligned means the copy as it is (or as the helper made it) puts every row at the
            // offset record() will read it from: bpp equal and either the image already is W wide
            // or the rows were padded to W.
            if (!(bpp_ok && (row_texels || ins[k].w == W))) stuck = true;
            if (row_texels && ins[k].w < W) padded = true;
            // One verdict per slot: MISMATCH is the case no stride can fix (the chain wants a
            // different number of bytes per texel than the image holds - RR still reads mv/dep
            // code 0 = 8 B against the harness's 4 B/px), TOO WIDE is one Vulkan will not let the
            // copy pad, TOO TALL one it must not (the guard in copy_row_texels), PADDED is rows
            // carried up to W, and "ok" means this copy is byte-for-byte what it was before the
            // pitch logic existed.
            std::snprintf(rows + std::strlen(rows), sizeof rows - std::strlen(rows),
                          " %s %ux%u+%uB chain %uB=%u tx -> %s", ins[k].name, ins[k].w, ins[k].h,
                          img_bpp, row_bytes, row_texels,
                          !bpp_ok ? "MISMATCH"
                                  : (ins[k].w > W ? "TOO WIDE"
                                                  : (ins[k].h > H ? "TOO TALL"
                                                                  : (ins[k].w < W ? "PADDED" : "ok"))));
        }
        ngx_log("input pitch (id=%u, eval #%llu):%s [%s]", s->id, s->eval_id, rows,
                stuck ? "some rows cannot land on the chain's stride"
                      : (padded ? "rows padded up to the chain's pitch" : "every row aligned"));
    }
    s->prev_jx = fp.jx;
    s->prev_jy = fp.jy;
    // The eval/tag suffix is how this line is attributed: lines with [eval #N diag-v1] came from
    // this build, untagged ones from whatever else is writing the same log (diag-v1).
    ngx_log("%s frame %d: %zu images in, %dx%d -> %dx%d, recorded in %llu ms [eval #%llu id=%u %s]",
            feature_name(s), s->frame_idx - 1, n, c->width(), c->height(), out.w, out.h, ngx_ms() - t0,
            s->eval_id, s->id, kPortTag);
    return NVSDK_NGX_Result_Success;
}

// Why gather_frame could not produce a frame, remembered at the site that refused it.
//
// Diagnostic intent (the observed in-game log carries one failure pair and one success line per
// frame in equal counts, which the strict single collect_frame call per EvaluateFeature cannot
// produce by itself): collect_frame says only "Fail", while the exact reason sits in an earlier,
// separate line - so a log cannot tell two concurrent calls from a second/older _nvngx.dll in the
// same process, nor from an Output path that quietly reports zero size.  Every failing return
// below therefore also names its slot, its sub-reason and the resource's own numbers here, and
// EvaluateFeature prints the whole tuple with its call id on the way out.  Recording only: no
// control flow, no second fetch pass, nothing gated.
struct FrameDiag {
    const char* slot = "params";   // Color / MotionVectors / Depth / ... / Output / params
    const char* reason = "";       // a literal: the exact refusal, as stated where it happened
    char ctx[160] = "";            // the resource's own numbers, formatted at the refusing site

    void at(const char* sl) { slot = sl; }               // whose fetch is failing right now
    void why(const char* rs, const char* context = "") {  // the sub-reason of that fetch
        reason = rs;
        std::snprintf(ctx, sizeof ctx, "%s", context ? context : "");
    }
    void fail(const char* sl, const char* rs, const char* context = "") { at(sl); why(rs, context); }
};

// The D3D12 half of FrameDiag::ctx: the resource's own numbers, whatever the refusal is.  desc is
// taken before the handle test so even "no VkImage" can quote Width/Height, Format and Dimension,
// and `%p` wants a real void* (cast below); a null VkImage says so in words after the address.
static void d3d_ctx(char* buf, size_t cap, const D3D12_RESOURCE_DESC& desc, VkImage img) {
    std::snprintf(buf, cap, "desc=%llux%llu fmt=%u dim=%u VkImage=%p%s",
                  (unsigned long long)desc.Width, (unsigned long long)desc.Height,
                  unsigned(desc.Format), unsigned(desc.Dimension), (void*)img,
                  img ? "" : " (NULL)");
}

// The Vulkan side of "hand me this image": NGX_Resource_VK as ngx_client sets it.
static bool vk_image_resource(NVSDK_NGX_Parameter* p, const char* key, BridgeImage& o, FrameDiag& dg) {
    void* v = nullptr;
    if (!ngx_ok(p->vtbl->GetVoidPtr(p, key, &v)) || !v) {
        dg.why("no NGX_Resource_VK was set for this key");
        return false;
    }
    const auto* r = reinterpret_cast<const NGX_Resource_VK*>(v);
    if (r->Type != NGX_RESOURCE_VK_IMAGEVIEW) {
        dg.why("the NGX_Resource_VK is not an image view");
        return false;
    }
    if (!r->Resource.ImageViewInfo.Image) {
        char ctx[96];
        std::snprintf(ctx, sizeof ctx, "view=%ux%u VkImage=%p (NULL)",
                      r->Resource.ImageViewInfo.Width, r->Resource.ImageViewInfo.Height,
                      (void*)r->Resource.ImageViewInfo.Image);
        dg.why("the image view carries a null VkImage", ctx);
        return false;
    }
    o.image = r->Resource.ImageViewInfo.Image;
    o.format = r->Resource.ImageViewInfo.Format;
    o.w = r->Resource.ImageViewInfo.Width;
    o.h = r->Resource.ImageViewInfo.Height;
    return true;
}

// The D3D12 side of the same: vkd3d-proton hands the VkImage out through its interop device.
//
// DXGI_FORMAT_R32G32_FLOAT (16) is the in-game MotionVectors format: Crimson Desert hands it
// through untouched (OptiScaler v0.9.4 converts no input format - nvngx input resources are given
// to it as the game wrote them), and not decoding it cost 190,364 refused frames - a 100% refusal
// rate - in one session's log, every one of them the line
// "MotionVectors: DXGI format 16 is not one this bridge decodes".  The measured number is pinned
// here so a future header cannot silently decode a different enum into the MV slot.
static_assert(DXGI_FORMAT_R32G32_FLOAT == 16, "the in-game MotionVectors DXGI format, as the log prints it");
// The in-game Depth of Crimson Desert: "Depth: DXGI format 19 is not one this bridge decodes",
// 63,760 refused frames (100%) in the run that measured it.  DXGI_FORMAT_R32G8X24_TYPELESS is the
// typeless D32+S8 container (mingw dxgiformat.h:58, 0x13).
static_assert(DXGI_FORMAT_R32G8X24_TYPELESS == 19, "the in-game Depth DXGI format, as the log prints it");
// The RR albedo of record: 1,312 x FAIL slot=DiffuseAlbedo fmt=28 per session
// (tools/scenario/REPORT.md:905, RR-OPEN-KNOWN-CAUSE, parked there with "fix = accept R8G8B8A8").
// R8G8B8A8_UNORM is code 5 of the RR texfmt table (texfmt.py RR_BPP[5] = 4 B; rr_texfmt.glsl
// dec_rr code 5 = RGBA8 UNORM /255 little-endian R,G,B,A = exactly VK_FORMAT_R8G8B8A8_UNORM's
// memory layout) - the DECODE chain has carried it all along; what was missing is this map row,
// the slot gate below and the code/pitch re-derivation in collect_frame.
static_assert(DXGI_FORMAT_R8G8B8A8_UNORM == 28, "the in-game RR albedo DXGI format, as the log prints it");

VkFormat ngxd_dxgi_to_vk(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
        case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
        case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        // RR albedo (DXGI 28, measured in-game): the view vkd3d creates for it, carried only into
        // the slots whose gather code can decode it (extra[] below accepts it for DiffuseAlbedo /
        // SpecularAlbedo; every other slot still refuses through want_format).
        case DXGI_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        // Depth-stencil containers, resolved exactly as vkd3d-proton resolves them when it CREATEs
        // the image (libs/vkd3d/utils.c "vkd3d_depth_stencil_formats", table read from the source:
        // 19/20/21 -> VK_FORMAT_D32_SFLOAT_S8_UINT); the copy then reads its DEPTH aspect
        // (record_bridge passes BridgeImage::aspect), whose texels are 4 B raw float32 = the
        // chain's code-0 depth.  R32_TYPELESS/D16/D24 stay unmapped below, on purpose.
        case DXGI_FORMAT_R32G8X24_TYPELESS:         // 19: measured in-game
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:      // 20: the typed sibling, same VkFormat
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS:  // 21: same family, DEPTH plane
            return VK_FORMAT_D32_SFLOAT_S8_UINT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

// The depth formats that remain refused, and why, now that 19/20/21 are carried (they were all
// "not decoded" before the 2026-10-30 in-game run proved 19 is the real one):
//   - D24_UNORM_S8_UINT (45) / R24_UNORM_X8_TYPELESS (47): the copy side is now solved (DEPTH
//     aspect, 4 B packed D in the low 24), and sr_gather.comp:127-129 already decodes a non-zero
//     fmt_dep code as that packed UNORM - but nothing has handed this bridge one yet, so the
//     code-1 re-derivation would be a guess; refuse until an in-game line names it.
//   - D16_UNORM (55): its depth texel is 2 B, while the chain strides dep at W * 4
//     (pipeline.cpp:1852) - no bufferRowLength can halve a texel (the pitch guard below says so
//     for MotionVectors, and it applies here too); converting means a shader, not an enum.
//   - R32_TYPELESS (39): genuinely ambiguous - vkd3d_get_format resolves it to D32_SFLOAT only
//     when asked with depth_stencil=true and to R32_SFLOAT otherwise (utils.c:915-941), and
//     desc.Format alone does not say which usage won.  GetVulkanResourceInfo does not hand the
//     VkFormat back, so the map cannot break the tie honestly.
// A resource still refused here is named by the dxgi_to_vk line above, with both enums.

static bool d3d_image_resource(FeatureState* s, ID3D12Resource* res, D3D12_RESOURCE_STATES state,
                               const char* key, BridgeImage& o, FrameDiag& dg) {
    if (!res) { dg.why("no ID3D12Resource", "desc=none VkImage=none"); return false; }
    if (!s->interop) { dg.why("the feature has no vkd3d interop device"); return false; }
    // GetDesc is a pure read, so it moves above the handle test; that way every refusal below can
    // quote the resource's numbers instead of the caller having to re-derive them.
    const auto desc = res->GetDesc();
    UINT64 handle = 0, offset = 0;
    const bool have = SUCCEEDED(s->interop->GetVulkanResourceInfo(res, &handle, &offset)) && handle != 0;
    o.image = have ? (VkImage)handle : VK_NULL_HANDLE;
    o.format = ngxd_dxgi_to_vk(desc.Format);
    o.w = unsigned(desc.Width);
    o.h = unsigned(desc.Height);
    o.dxgi = unsigned(desc.Format);
    char ctx[160];
    d3d_ctx(ctx, sizeof ctx, desc, o.image);
    if (!have) {
        dg.why("GetVulkanResourceInfo failed or handed back a null VkImage", ctx);
        return false;
    }
    s->interop->GetVulkanImageLayout(res, state, &o.layout);
    if (o.format == VK_FORMAT_UNDEFINED) {
        ngx_log("%s: DXGI format %u is not one this bridge decodes", key, unsigned(desc.Format));
        dg.why("DXGI format is not one this bridge decodes", ctx);
        return false;
    }
    if (o.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        ngx_log("%s: vkd3d reported an undefined layout", key);
        dg.why("vkd3d reported an undefined layout", ctx);
        return false;
    }
    return true;
}

// Which way an NGX parameter hands over an image: Set(key, NGX_Resource_VK*) on the Vulkan path,
// Set(key, ID3D12Resource*) on the D3D12 path.  Either way the slot's staging buffer and the port's
// format code are the same, so both paths collect their frames the same way.
static bool fetch_image(NVSDK_NGX_Parameter* p, FeatureState* s, bool vulkan, const char* key, const char* name,
                        VkFormat want, VkFormat alt, VkBuffer buf, D3D12_RESOURCE_STATES state,
                        BridgeImage& o, FrameDiag& dg) {
    o = BridgeImage{};
    o.buf = buf;
    o.name = name;      // which slot this is, so the caller's per-call lines can name every input
    dg.at(name);        // any refusal below belongs to this slot (diag-v1)
    if (vulkan) {
        if (!vk_image_resource(p, key, o, dg)) {
            ngx_log("Evaluate: no %s image", name);
            return false;
        }
    } else {
        ID3D12Resource* res = nullptr;
        p->vtbl->GetD3D12Resource(p, key, &res);
        if (!res || !d3d_image_resource(s, res, state, name, o, dg)) {
            ngx_log("Evaluate: no %s resource", name);
            if (!res) dg.why("GetD3D12Resource returned null for this key", "no ID3D12Resource");
            return false;
        }
    }
    if (!want_format(name, o.format, want, alt, o.dxgi)) {
        char ctx[160];
        std::snprintf(ctx, sizeof ctx, "VkFormat=%d wanted=%d (DXGI=%u), %ux%u VkImage=%p%s", int(o.format),
                      int(want), o.dxgi, o.w, o.h, (void*)o.image, o.image ? "" : " (NULL)");
        dg.why("VkFormat is not the one the gather shader decodes", ctx);
        return false;
    }
    // One derivation point for both paths: a depth-stencil image carries no COLOR aspect, so every
    // barrier and copy in record_bridge has to name DEPTH instead (see vk_is_depth_stencil).  A
    // depth resource that accepted here is a float32 DEPTH aspect, which is the chain's code 0.
    o.aspect = vk_is_depth_stencil(o.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    return true;
}

// One slot: the parameter key, the name every diagnostic line uses, and the formats the gather can
// actually decode - `fmt` first, `alt` an extra accepted one (VK_FORMAT_UNDEFINED when there is
// none).  MotionVectors takes both NVSDK spellings of the same 2-channel float: RG16F (DXGI 34) is
// what ngx_client hands and what the chain's code 0 reads, RG32F (DXGI 16) is what Crimson Desert
// hands and what code 1/10 read - picking which code from the DECODED resource is done below.
struct Slot { const char* key; const char* name; VkFormat fmt; VkFormat alt; };

// Everything one EvaluateFeature call needs, for either API: the frame parameters, the inputs the
// feature was created with and the output image.
//
// DLSS5_AMD_SKIP_INPUTS=<comma list of col,mv,dep,nrm,alb,spa,rgh> leaves those chain inputs at zero
// (neither fetched nor copied): a bisect switch for telling a bad image round-trip from a bad chain.
static bool input_skipped(const char* short_name) {
    static const char* env = std::getenv("DLSS5_AMD_SKIP_INPUTS");
    if (!env || !*env) return false;
    std::string s(env), t(short_name);
    for (size_t i = 0; i + t.size() <= s.size(); ++i)
        if (s.compare(i, t.size(), t) == 0) {
            const char a = i ? s[i - 1] : ',';
            const char b = (i + t.size() < s.size()) ? s[i + t.size()] : ',';
            if ((a == ',' || a == ' ') && (b == ',' || b == ' ' || b == '\0')) return true;
        }
    return false;
}

static NVSDK_NGX_Result collect_frame(NVSDK_NGX_Parameter* p, FeatureState* s, bool vulkan,
                                      ngxd::FrameChain* c, ngxd::FrameParams& fp,
                                      BridgeImage* ins, size_t* n, BridgeImage& out, FrameDiag& dg) {
    if (!read_frame_params(p, s, fp)) {
        // read_frame_params has already said which sub-rectangle/parameters it refused; here only
        // the slot and the chain's own out size are recorded for the caller's line.
        dg.fail("params", "read_frame_params refused (see the line above)",
                "chain in as created, subrect must be the whole render target");
        return NVSDK_NGX_Result_Fail;
    }

    const Slot common[] = {
        {"Color", "Color", VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED},
        {"MotionVectors", "MotionVectors", VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32G32_SFLOAT},
        // D32 depth-aspect = RAW FLOAT32 at 4 B/px = the same bytes the harness's R32F slot feeds
        // (code 0), so the only thing that differs in-game is how the copy addresses the image
        // (BridgeImage::aspect DEPTH); fmt_dep stays 0's either way (read_frame_params:490).
        {"Depth", "Depth", VK_FORMAT_R32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT},
    };
    // DiffuseAlbedo/SpecularAlbedo take a second format: VK_FORMAT_R8G8B8A8_UNORM (DXGI 28,
    // measured in-game, REPORT.md:905) = RR code 5, 4 B/texel.  The code is re-derived from the
    // fetched image in the loop below (the fmt_mv precedent), so the default RGBA16F path keeps
    // code 0 and every pitch it reads is unchanged.
    const Slot extra[] = {
        {"GBuffer.Normals", "GBuffer.Normals", VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED},
        {"DLSS.Input.DiffuseAlbedo", "DiffuseAlbedo", VK_FORMAT_R16G16B16A16_SFLOAT,
         VK_FORMAT_R8G8B8A8_UNORM},
        {"DLSS.Input.SpecularAlbedo", "SpecularAlbedo", VK_FORMAT_R16G16B16A16_SFLOAT,
         VK_FORMAT_R8G8B8A8_UNORM},
        {"GBuffer.Roughness", "Roughness", VK_FORMAT_R32_SFLOAT, VK_FORMAT_UNDEFINED},
    };
    const char* common_short[] = {"col", "mv", "dep"};
    const char* extra_short[] = {"nrm", "alb", "spa", "rgh"};
    const VkBuffer bufs[] = {c->in_color_buf(), c->in_mv_buf(), c->in_depth_buf()};
    const VkBuffer extra_bufs[] = {c->in_normal_buf(), c->in_albedo_buf(), c->in_specular_buf(),
                                   c->in_rough_buf()};
    const bool rr = s->feature == NVSDK_NGX_Feature_RayReconstruction;
    const D3D12_RESOURCE_STATES input_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

    *n = 0;
    for (size_t k = 0; k < 3; ++k) {
        if (input_skipped(common_short[k])) continue;
        if (!fetch_image(p, s, vulkan, common[k].key, common[k].name, common[k].fmt, common[k].alt, bufs[k],
                         input_state, ins[*n], dg))
            return NVSDK_NGX_Result_Fail;
        ++*n;
    }
    if (rr) {
        for (size_t k = 0; k < 4; ++k) {
            if (input_skipped(extra_short[k])) continue;
            if (!fetch_image(p, s, vulkan, extra[k].key, extra[k].name, extra[k].fmt, extra[k].alt,
                             extra_bufs[k], input_state, ins[*n], dg))
                return NVSDK_NGX_Result_Fail;
            ++*n;
        }
    }

    // THE FORMAT CODE, settled from the resource that was actually fetched rather than assumed.
    // The chain decodes motion vectors per code (sr_gather.comp:109-121: 0 = RG16F 4 B/px,
    // 1 = RG32F two uintBitsToFloat words, "other" = an 8-byte texel read as two f16), and
    // record() derives its row pitch from the same code (pipeline.cpp:1849 sr_mv_bpp = (fmt_mv == 0)
    // ? 4 : 8, pushed at 1851; RR pipeline.cpp:2251 pitch = W * kRrBpp[code]).  The referee's
    // proven combinations are cells sr_M_mv1_dep1 (run_all.sh:183, --mv 1) and rr_F_fmt_a
    // (run_all.sh:192, --mv 10), so an RG32F motion-vector image is code 1 for SR and code 10 for
    // RR (rr_texfmt.glsl: "code == 10 // RG32F"); an RG16F image stays at code 0, which is both
    // what read_frame_params prefilled and the only mismatch-free answer for the harness's
    // DXGI_FORMAT_R16G16_FLOAT MotionVectors (ngx_client/main.cpp:88 code0 = 0).
    for (size_t k = 0; k < *n; ++k) {
        // RR albedo re-derivation (the same rule as fmt_mv below): a fetched R8G8B8A8_UNORM view
        // (DXGI 28) is RR code 5 - 4 B/texel in kRrBppE[5] / kBpp[5] / rr_bpp(5), decoded by
        // dec_rr code 5.  RGBA16F views keep the prefilled code 0, so the default-format path is
        // byte-identical; these slots are only fetched on the RR path (extra[] is rr-only).
        if (ins[k].format == VK_FORMAT_R8G8B8A8_UNORM) {
            if (!std::strcmp(ins[k].name, "DiffuseAlbedo"))   { fp.fmt_alb = 5; continue; }
            if (!std::strcmp(ins[k].name, "SpecularAlbedo"))  { fp.fmt_spa = 5; continue; }
        }
        if (std::strcmp(ins[k].name, "MotionVectors") != 0) continue;
        if (ins[k].format == VK_FORMAT_R32G32_SFLOAT)
            fp.fmt_mv = rr ? 10 : 1;
        // Exactly one MV layout is unreachable and therefore refused rather than sheared: the
        // texel sizes agree (so a row length could have made the strides meet) but the image is
        // WIDER than the W rows the gather reads, and bufferRowLength must be 0 or
        // >= imageExtent.width - the copy cannot narrow it (VUID-vkCmdCopyImageToBuffer).  Under
        // that width, rows are padded up to W in record_bridge (image-wise readable) and every
        // outcome is named on that function's frame-1 "input pitch" line; a texel-size MISMATCH
        // (RR's mv/dep code 0 read against 4 B/px images) is deliberately NOT a refusal, because
        // that is the harness's measured bytecode today and refusing it would trade a degraded
        // frame for no frame.
        const unsigned mv_bpp = vk_format_bpp(ins[k].format);
        const unsigned chain_w = unsigned(c->width());
        if (mv_bpp && chain_row_bytes(c, rr, fp, ins[k].name) == chain_w * mv_bpp && ins[k].w > chain_w) {
            // Display-resolution MVs are not a layout failure when the image is exactly the
            // chain's OUTPUT extent (Gemini-2 claim 4): the chain already has the oW-pitch path -
            // fp.mvhi routes sr_gather's MV half off (sr_gather.comp:142 `if (mvhi == 0)`) and
            // sr_gather_mv reads in_mv_ at the oW pitch (pipeline.cpp:1874-1885, pmv = oW *
            // bpp), which is exactly the tight pack copy_row_texels leaves for an image wider
            // than W (nvngx.cpp:675 returns 0 there); in_mv_ is sized max(W,oW) x max(H,oH) x 16
            // (pipeline.cpp:968) so the copy fits.  Only SR: the RR record path has no
            // sr_gather_mv twin, so RR keeps the original refusal.
            if (!rr && ins[k].w == s->ow && ins[k].h == s->oh) {
                fp.mvhi = 1;
                ngx_log("MotionVectors %ux%u are output-sized (chain %u tx in, %ux%u out): mvhi=1",
                        ins[k].w, ins[k].h, chain_w, s->ow, s->oh);
                break;
            }
            char ctx[160];
            std::snprintf(ctx, sizeof ctx, "image %ux%u=%u B/tx, chain rows %u tx (%u B), W=%u",
                          ins[k].w, ins[k].h, mv_bpp, chain_w, chain_w * mv_bpp, chain_w);
            ngx_log("MotionVectors rows cannot be laid out at the chain's stride: %s", ctx);
            dg.fail("MotionVectors", "the MV image is wider than the row the chain reads", ctx);
            return NVSDK_NGX_Result_Fail;
        }
        break;
    }

    // M2: the app's Output must match the chain's out buffer element format (the buffer->image
    // copy does no conversion; bufferRowLength=0 packs in IMAGE texels). NGX's output resource is
    // normally R16G16B16A16_FLOAT - that is the f16 contract; 0 keeps the old RGBA32F contract.
#if DLSS5_AMD_OUT_F16
    const VkFormat out_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
#else
    const VkFormat out_fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
#endif
    if (!fetch_image(p, s, vulkan, "Output", "Output", out_fmt, VK_FORMAT_UNDEFINED, c->out_color_buf(),
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, out, dg))
        return NVSDK_NGX_Result_Fail;
    if (out.w != unsigned(c->out_width()) || out.h != unsigned(c->out_height())) {
        char ctx[160];
        // Both size pairs in one place: the zero-size-Output case shows up here as 0x0 against
        // whatever the chain was created with, and the chain's own numbers are on the caller's line.
        std::snprintf(ctx, sizeof ctx, "out=%ux%u chain out=%dx%d (VkImage=%p)", out.w, out.h,
                      c->out_width(), c->out_height(), (void*)out.image);
        ngx_log("Output is %ux%u, the feature is %dx%u", out.w, out.h, c->out_width(), c->out_height());
        dg.fail("Output", "the output image does not match the chain's out buffer", ctx);
        return NVSDK_NGX_Result_Fail;
    }
    return NVSDK_NGX_Result_Success;
}

// The per-call outcome lines (diag-v1): one line per EvaluateFeature outcome, both stamped with the
// call id, the FeatureState id and the chain's out_width()/out_height(), so a log can be split back
// into calls - a fail/ok pair with consecutive ids is two calls, and ids that restart at 1 while the
// old untagged lines continue mean a second _nvngx.dll build is writing to the same file.
static void log_eval_fail(unsigned long long eval_id, const FeatureState* s, size_t n,
                          const FrameDiag& dg, const ngxd::FrameChain* c) {
    ngx_log("eval #%llu FAIL: id=%u feature=%d preset=%c n=%zu slot=%s why=%s ctx=[%s] chain out=%dx%d",
            eval_id, s->id, int(s->feature), s->preset, n, dg.slot, dg.reason, dg.ctx,
            c->out_width(), c->out_height());
}

static void log_eval_ok(unsigned long long eval_id, const FeatureState* s, size_t n,
                        const BridgeImage* ins, const BridgeImage& out, const ngxd::FrameChain* c,
                        const ngxd::FrameParams& fp) {
    // Every input that was actually fetched, with its decoded formats and both VkImage handles: a
    // n=1 frame or a 0x0 output on the success path is as diagnostic as one on the failure path.
    char slots[8 * 96] = "";
    for (size_t k = 0; k < n; ++k)
        std::snprintf(slots + std::strlen(slots), sizeof slots - std::strlen(slots),
                      " [%zu %s %ux%u vk=%d dxgi=%u VkImage=%p]", k, ins[k].name, ins[k].w, ins[k].h,
                      int(ins[k].format), ins[k].dxgi, (void*)ins[k].image);
    // The three gather codes sent into record(): the MV one is picked from the fetched resource
    // (read_frame_params prefills 0), so this is where an in-game R32G32_FLOAT frame shows code 1
    // instead of the harness's 0.
    ngx_log("eval #%llu OK: id=%u feature=%d n=%zu out=%ux%u vk=%d VkImage=%p chain out=%dx%d "
            "fmt col=%d mv=%d dep=%d%s",
            eval_id, s->id, int(s->feature), n, out.w, out.h, int(out.format), (void*)out.image,
            c->out_width(), c->out_height(), fp.fmt_col, fp.fmt_mv, fp.fmt_dep, slots);
}

// The discovery pair of the NGX ABI - {SDKVersion, FeatureID, Identifier, ...} and {FeatureSupported,
// MinHWArchitecture, MinOSVersion[255], ...} (nvsdk_ngx_defs.h:490-509) - is what Streamline's
// GetFeatureRequirements probe passes, and the hand-written ngx_abi.h covers neither (it is the
// subset ngx_client calls, tools/verify/ngx_client/ngx_abi.h:1).  Declared here as incomplete types
// exactly like ngx_abi.h's ID3D12Device, so the export signature carries the real SDK names while
// the only two fields the answer needs are written by offset below, the way reference-nr-vulkan's
// core already does it (reference-nr-vulkan/linux/src/pe/nr_ngx_core.cpp:460-473).
struct NVSDK_NGX_FeatureDiscoveryInfo;
struct NVSDK_NGX_FeatureRequirement;

// The FFX surface calls the same record path (ngx_bridge_shared.h declares it). A thin forwarder
// keeps record_bridge's static linkage and its use of the NGX TU's FrameDiag/private helpers.
NVSDK_NGX_Result ngxd_record_bridge(VkCommandBuffer cmd, FeatureState_* s, const ngxd::FrameParams& fp,
                                    const BridgeImage_* ins, size_t n, const BridgeImage_* out) {
    return record_bridge(cmd, static_cast<FeatureState*>(s), fp,
                         reinterpret_cast<const BridgeImage*>(const_cast<BridgeImage_*>(ins)), n,
                         *reinterpret_cast<const BridgeImage*>(out));
}

// The FFX surface's per-frame OK line rides the SAME emitter and the SAME g_eval_seq as the NGX
// export's (ngx_bridge_shared.h declares it): one grammar for both surfaces, so the harness's
// g2-ffx gate greps `eval #N OK: ... fmt col=.. mv=.. dep=..` exactly as it does for the dlss
// cell (plan section 8).  The id is taken and stamped here so the caller's [ffx]-prefixed line
// can echo it.
unsigned long long ngxd_log_eval_ok(FeatureState_* s, size_t n, const BridgeImage_* ins,
                                    const BridgeImage_* out, const ngxd::FrameChain* c,
                                    const ngxd::FrameParams& fp) {
    auto* fs = static_cast<FeatureState*>(s);
    const unsigned long long eval_id = ++g_eval_seq;
    fs->eval_id = eval_id;
    log_eval_ok(eval_id, fs, n, reinterpret_cast<const BridgeImage*>(ins),
                *reinterpret_cast<const BridgeImage*>(out), c, fp);
    return eval_id;
}

// Exports
extern "C" {

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Init(
    unsigned long long appId, const wchar_t* dataPath, ID3D12Device* device,
    const void* featureInfo, int sdkVersion) {
    if (device && !g_d3d12_device) { g_d3d12_device = device; g_d3d12_device->AddRef(); }
    return NVSDK_NGX_Result_Success;
}

// OptiScaler's NVNGXProxy::InitDx12 does not resolve the plain Init above: it GetProcAddress-es
// only these two names (NVNGX_Proxy.h:537-540) and leaves nvResult at NVSDK_NGX_Result_Fail
// (0xBAD00000) when neither is present, so _dx12Inited stays false and every later
// D3D12_CreateFeature call is rejected -- a black screen with no port log. They carry more
// parameters than Init but do the same essential work (bind the device), so both forward here.
__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Init_ProjectID(
    const char* projectId, int engineType, const char* engineVersion,
    const wchar_t* dataPath, ID3D12Device* device, int sdkVersion,
    const void* featureInfo) {
    (void)projectId; (void)engineType; (void)engineVersion; (void)dataPath;
    (void)sdkVersion; (void)featureInfo;
    if (device && !g_d3d12_device) { g_d3d12_device = device; g_d3d12_device->AddRef(); }
    ngx_log("D3D12 Init_ProjectID: project %s (OptiScaler backend now bound) [%s]",
            projectId ? projectId : "(null)", kPortTag);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Init_Ext(
    unsigned long long appId, const wchar_t* dataPath, ID3D12Device* device,
    int sdkVersion, const void* featureInfo) {
    (void)appId; (void)dataPath; (void)sdkVersion; (void)featureInfo;
    if (device && !g_d3d12_device) { g_d3d12_device = device; g_d3d12_device->AddRef(); }
    ngx_log("D3D12 Init_Ext: appId %llu (OptiScaler backend now bound) [%s]",
            (unsigned long long)appId, kPortTag);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* device) {
    if (g_d3d12_device) { g_d3d12_device->Release(); g_d3d12_device = nullptr; }
    return NVSDK_NGX_Result_Success;
}

// The device-less spelling of the line above (NVNGX_Proxy.h:258, PFN_D3D12_Shutdown), resolved by
// OptiScaler's proxy at NVNGX_Proxy.h:541-542 as well as by Streamline.  Shutdown1 is the
// implementation that already knows what to tear down and deliberately ignores its device argument
// (it drops g_d3d12_device whichever pointer the caller hands), so nullptr is the honest argument
// here rather than a second copy of the body.
__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Shutdown(void) {
    return NVSDK_NGX_D3D12_Shutdown1(nullptr);
}

// ---- D5/S2 (in-game S-standalone, 2026-10-31 17:51): the layer ABOVE the Available key.  With
// `SuperSampling.Available` primed (17:08 fix) sl.dlss passes dlssEntry.cpp:937, but the very same
// startup then fails four times at dlssEntry.cpp:819 (Launcher_2026_10_31_17_51_43_328.log:52-58):
// "DLSSContext 'getOptimalSettings' callback is missing" - and no CreateFeature ever follows (port
// log).  The callback is CORE-published, not game- or SL-published: OptiScaler's working surface
// puts both function pointers into every parameter object (NVNGX_Parameter.cpp:726-728 inside
// InitNGXParameters) and exports _PopulateParameters_Impl as the (re)population entry
// (NVNGX_DLSS_Dx12.cpp:558) - the export our port lacked while the game's sl.dlss.dll resolves the
// core's surface.  Same era evidence as above: this line too appears in ZERO of the 21 historic
// Launcher_*.log files and zero runs/ archives - Opti was always the surface that supplied it.
//
// Contract mirrored from NVSDK_NGX_DLSS_GetOptimalSettingsCallback (NVNGX_Parameter.cpp:391-440):
// Width, Height AND PerfQualityValue must be present, else Fail; scale table follows Opti's switch
// (enum order per nvsdk_ngx_defs.h:209-218: MaxPerf=0, Balanced=1, MaxQuality=2, UltraPerformance=3,
// UltraQuality=4, DLAA=5); DRS band mirrors Opti (max = display, min = max(render, half-display),
// DLAA pinned to display).  Outputs are dual-written (string key + "#.." enum alias) exactly like
// Opti - the alias table lives in nvsdk_ngx_defs.h:513+.  Stats callback mirrors Opti's
// NVSDK_NGX_DLSS_GetStatsCallback (NVNGX_Parameter.cpp:688-703): SizeInBytes = W*H*31, always ok.
static NVSDK_NGX_Result NGX_MSABI port_get_optimal_settings(NVSDK_NGX_Parameter* q) {
    auto* p = reinterpret_cast<ParametersImpl*>(q);
    unsigned int W = 0, H = 0;
    int pq = -1;
    if (!p || p->get_uint("Width", &W) != NVSDK_NGX_Result_Success ||
        p->get_uint("Height", &H) != NVSDK_NGX_Result_Success ||
        p->get_int("PerfQualityValue", &pq) != NVSDK_NGX_Result_Success ||
        W == 0 || H == 0)
        return NVSDK_NGX_Result_Fail;
    float ratio;
    switch (pq) {
        case 0:  ratio = 2.0f;  break;   // MaxPerf
        case 1:  ratio = 1.7f;  break;   // Balanced
        case 2:  ratio = 1.5f;  break;   // MaxQuality
        case 3:  ratio = 3.0f;  break;   // UltraPerformance
        case 4:  ratio = 1.3f;  break;   // UltraQuality
        case 5:  ratio = 1.0f;  break;   // DLAA
        default: ratio = 1.7f;  break;   // Opti's default arm
    }
    const unsigned int OW = (unsigned int)((float)W / ratio);
    const unsigned int OH = (unsigned int)((float)H / ratio);
    const float scale = 1.0f / ratio;
    const unsigned int minW = (pq == 5) ? W : (OW < W / 2 ? OW : W / 2);
    const unsigned int minH = (pq == 5) ? H : (OH < H / 2 ? OH : H / 2);
    p->set_float("Scale", scale);
    p->set_float("#\x0f", scale);                         // NVSDK_NGX_EParameter_Scale
    p->set_float("SuperSampling.ScaleFactor", scale);
    p->set_uint("OutWidth", OW);
    p->set_uint("OutHeight", OH);
    p->set_uint("#\x12", OW);                             // NVSDK_NGX_EParameter_OutWidth
    p->set_uint("#\x13", OH);                             // NVSDK_NGX_EParameter_OutHeight
    p->set_uint("DLSS.Get.Dynamic.Max.Render.Width", W);
    p->set_uint("DLSS.Get.Dynamic.Max.Render.Height", H);
    p->set_uint("DLSS.Get.Dynamic.Min.Render.Width", minW);
    p->set_uint("DLSS.Get.Dynamic.Min.Render.Height", minH);
    p->set_uint("SizeInBytes", W * H * 31u);
    p->set_uint("#\x30", W * H * 31u);                    // NVSDK_NGX_EParameter_SizeInBytes
    p->set_uint("DLSSMode", 1u);                          // NVSDK_NGX_DLSS_Mode_DLSS_DLISP
    p->set_uint("#\x41", 1u);                             // NVSDK_NGX_EParameter_DLSSMode
    ngx_log("optimal settings cb (dlssEntry:819): %ux%u pq=%d -> %ux%u scale=%.4f [S2]",
            W, H, pq, OW, OH, (double)scale);
    return NVSDK_NGX_Result_Success;
}

static NVSDK_NGX_Result NGX_MSABI port_get_stats(NVSDK_NGX_Parameter* q) {
    auto* p = reinterpret_cast<ParametersImpl*>(q);
    if (!p) return NVSDK_NGX_Result_Success;
    unsigned int W = 1920, H = 1080;
    if (p->get_uint("Width", &W) != NVSDK_NGX_Result_Success) W = 1920;
    if (p->get_uint("Height", &H) != NVSDK_NGX_Result_Success) H = 1080;
    p->set_uint("SizeInBytes", W * H * 31u);
    p->set_uint("#\x30", W * H * 31u);
    return NVSDK_NGX_Result_Success;
}

// One capability surface for every parameter object the port hands out (AllocateParameters /
// GetParameters / GetCapabilityParameters / PopulateParameters_Impl all land here).
static void prime_params(ParametersImpl* p) {
    // D5/S-route (rr-pixel-diagnosis A1 + A4(i), 2026-10-31): sl.dlss_d's startup queries
    // params->Get("SuperSamplingDenoising.Available") (dlss_dEntry.cpp:1235-1247); an ABSENT key
    // returns the bare NVSDK_NGX_Result_Fail (0xBAD00000) and the plugin dies ("DLSSDContext
    // cannot run" - the in-game RR-black chain).  Truthful value 1: RR (preset F) runs on this
    // port.  (OptiScaler's own writer covers only the non-NVIDIA branch and writes 0 there - the
    // *other* failure, "DLSSDContext is not available".)
    p->set_uint("SuperSamplingDenoising.Available", 1u);
    p->set_uint("SuperSamplingDenoising.FeatureInitResult", 1u);
    p->set_uint("NeedsUpdatedDriver", 0u);
    // D5/S-route, SR side (in-game S-standalone black-3D, 2026-10-31 17:08): sl.dlss's OWN
    // startup queries params->Get("SuperSampling.Available") (dlssEntry.cpp:937 - the key is
    // the neighbouring string in sl.dlss.dll, between "Missing NGX default parameters" and the
    // failure text itself).  An ABSENT key returns 0xBAD00000 => "DLSSContext cannot run" =>
    // 'kFeatureDLSS' context is missing x4 (sl.cpp:179) => uninitialized SR output => black 3D
    // in SR and RR alike.  Never seen before today: Opti's dxgi answered this surface on 10-30.
    p->set_uint("SuperSampling.Available", 1u);
    p->set_uint("#\x01", 1u);                             // NVSDK_NGX_EParameter_SuperSampling_Available
    // The rest of Opti's InitNGXParameters recipe (NVNGX_Parameter.cpp:715+) that SL's plugins
    // read as defaults - SR capability/driver keys (UE floor 10.10 per the Opti UE branch), the
    // two core callbacks (+ their enum-alias twins; Opti dual-registers at :726/:728), and neutral
    // frame-param defaults.
    p->set_uint("SuperSampling.FeatureInitResult", 1u);
    p->set_uint("SuperSampling.NeedsUpdatedDriver", 0u);
    p->set_uint("SuperSampling.MinDriverVersionMajor", 10u);
    p->set_uint("SuperSampling.MinDriverVersionMinor", 10u);
    p->set_uint("Snippet.OptLevel", 0u);
    p->set_uint("Snippet.IsDevBranch", 0u);
    p->set_ptr("DLSSOptimalSettingsCallback", reinterpret_cast<void*>(&port_get_optimal_settings));
    p->set_ptr("#\x3e", reinterpret_cast<void*>(&port_get_optimal_settings));  // EParameter alias (:574)
    p->set_ptr("DLSSDOptimalSettingsCallback", reinterpret_cast<void*>(&port_get_optimal_settings));
    p->set_ptr("DLSSGetStatsCallback", reinterpret_cast<void*>(&port_get_stats));
    p->set_float("Sharpness", 0.0f);
    p->set_float("MV.Scale.X", 1.0f);
    p->set_float("MV.Scale.Y", 1.0f);
    p->set_float("MV.Offset.X", 0.0f);
    p->set_float("MV.Offset.Y", 0.0f);
    p->set_float("DLSS.Exposure.Scale", 1.0f);
    p->set_uint("PerfQualityValue", 0u);
    p->set_uint("SizeInBytes", 1920u * 1080u * 31u);
    p->set_uint("FreeMemOnReleaseFeature", 0u);
    for (const char* k : {"RayReconstruction.Hint.Render.Preset.DLAA",
                          "RayReconstruction.Hint.Render.Preset.UltraQuality",
                          "RayReconstruction.Hint.Render.Preset.Quality",
                          "RayReconstruction.Hint.Render.Preset.Balanced",
                          "RayReconstruction.Hint.Render.Preset.Performance",
                          "RayReconstruction.Hint.Render.Preset.UltraPerformance"})
        p->set_uint(k, 0u);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** out) {
    if (!out) return NVSDK_NGX_Result_Fail;
    auto* p = new ParametersImpl();
    prime_params(p);
    *out = p->as_param();
    return NVSDK_NGX_Result_Success;
}

// The (re)population export Opti ships (NVNGX_DLSS_Dx12.cpp:558 - whole body = InitNGXParameters).
// sl.dlss's "Missing NGX default parameters - DLSSContext cannot run" branch keys off its absence;
// resolving it also gives consumers a way to re-run the priming on an object they already hold.
__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_PopulateParameters_Impl(NVSDK_NGX_Parameter* p) {
    if (!p) return NVSDK_NGX_Result_Fail;
    prime_params(reinterpret_cast<ParametersImpl*>(p));
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* p) {
    if (p) delete reinterpret_cast<ParametersImpl*>(p);
    return NVSDK_NGX_Result_Success;
}

// Of the D3D12 names Streamline's sl.common.dll resolves against the loaded provider before it
// accepts an NGX context, these four were missing (NVSDK_NGX_D3D12_GetParameters,
// _GetCapabilityParameters, _GetScratchBufferSize, _GetFeatureRequirements); _ReleaseFeature was
// already exported and _Shutdown sits above, so after this block all six resolve.  Signatures are
// OptiScaler's own - the consumer that LoadLibrary+GetProcAddress-es them - PFN_D3D12_GetParameters
// (NVNGX_Proxy.h:260), PFN_D3D12_GetCapabilityParameters (:262), PFN_D3D12_GetScratchBufferSize
// (:264-266) and PFN_D3D12_GetFeatureRequirements (:270-272); tools/verify/ngx_client/ngx_abi.h:107-117
// covers only the subset the harness binds and carries no typedef for any of them.
//
// GetParameters / GetCapabilityParameters both answer "hand me a PARAMETER OBJECT" and nothing else,
// so both forward to the AllocateParameters body above.  This is exactly the OptiScaler fallback
// object the port is bypassing: one uniform parameter kind per stream beats a second,
// differently-behaved implementation whose getters would answer a different set of keys than the
// object CreateFeature already receives.
__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_GetParameters(NVSDK_NGX_Parameter** out) {
    return NVSDK_NGX_D3D12_AllocateParameters(out);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** out) {
    return NVSDK_NGX_D3D12_AllocateParameters(out);
}

// GetScratchBufferSize: the caller allocates the buffer ITSELF from this number and only memcpy-es
// into it afterwards, so correct behaviour depends on the value being non-negative, not on the value
// (OptiScaler lies here outright - NVNGX_DLSS_Dx12.cpp:1224-1225 answers 52428800 for every feature
// and LOG_WARNs "-> 52428800").  4 MiB is reference-nr-vulkan's proven answer for the same question,
// chosen there with the reasoning now in nr_ngx_core.cpp:255-258 ("large enough that nobody sizes a
// heap from it and finds it short, and small enough that a caller who really does allocate it is not
// asking for a gigabyte ... nothing on this side ever uses a scratch buffer").  Logged once per
// process: Streamline probes it companionably and it is not a frame-path call.
__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_GetScratchBufferSize(
    NVSDK_NGX_Feature feature, const NVSDK_NGX_Parameter* params, size_t* out) {
    (void)params;
    if (!out) return NVSDK_NGX_Result_Fail;
    static std::once_flag said;
    std::call_once(said, [feature]() {
        ngx_log("GetScratchBufferSize: answer %u bytes for feature %d [%s]", (unsigned)(4u * 1024u * 1024u),
                int(feature), kPortTag);
    });
    *out = 4u * 1024u * 1024u;
    return NVSDK_NGX_Result_Success;
}

// GetFeatureRequirements is the query that gates everything: wrong-negative and Streamline's gate
// stays shut ("a wrong-negative here keeps SL's gate closed"), so every answer except a definite
// "cannot run it" is "Supported" - FeatureSupported 0 IS Supported (nvsdk_ngx_defs.h:402), the
// second word leaves MinHWArchitecture at 0 (no minimum) and byte 8 NULs MinOSVersion (empty), which
// is the twelve bytes write_requirement() writes in reference-nr-vulkan's core
// (nr_ngx_core.cpp:466-473).  The feature id is read positionally from the discovery struct
// (nr_ngx_core.cpp:462-463: {SDKVersion, FeatureID} are both int-sized enums) - this port answers
// Supported for SR, RR and anything else it is asked, because it has no way to prove a negative and
// a false negative is the failure being fixed here.  Logged once per process.
__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_GetFeatureRequirements(
    IDXGIAdapter* adapter, const NVSDK_NGX_FeatureDiscoveryInfo* info,
    NVSDK_NGX_FeatureRequirement* out) {
    (void)adapter;
    if (!out) return NVSDK_NGX_Result_Fail;
    const int feature = info ? reinterpret_cast<const int*>(info)[1] : 0;
    auto* words = reinterpret_cast<unsigned int*>(out);
    words[0] = 0u;   // NVSDK_NGX_FeatureSupportResult_Supported
    words[1] = 0u;   // no minimum hardware architecture
    reinterpret_cast<char*>(out)[8] = '\0';   // MinOSVersion is a minimum, not a maximum: empty
    static std::once_flag said;
    std::call_once(said, [feature]() {
        ngx_log("GetFeatureRequirements: feature %d -> Supported (streamline's gate would stay shut "
                "on a wrong negative) [%s]", feature, kPortTag);
    });
    return NVSDK_NGX_Result_Success;
}

// The Vulkan chain needs a VkDevice; on the D3D12 path that is vkd3d-proton's, reached through its
// interop interface on the same device the game gave us.
bool ngxd_interop_device_for(ID3D12Device* device, ID3D12DXVKInteropDevice** out) {
    if (!device) return false;
    if (FAILED(device->QueryInterface(IID_ID3D12DXVKInteropDevice, reinterpret_cast<void**>(out))) || !*out)
        return false;
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    (*out)->GetVulkanHandles(&inst, &phys, &dev);
    if (!inst || !phys || !dev) { (*out)->Release(); *out = nullptr; return false; }
    g_vk_instance = inst;
    g_vk_phys_dev = phys;
    g_vk_device = dev;
    return load_vk_for_device(inst, phys, dev);
}


static ID3D12Device* device_of(ID3D12GraphicsCommandList* cmd) {
    if (g_d3d12_device) { g_d3d12_device->AddRef(); return g_d3d12_device; }
    if (!cmd) return nullptr;
    ID3D12Device* d = nullptr;
    if (FAILED(cmd->GetDevice(IID_PPV_ARGS(&d)))) return nullptr;
    return d;
}

static bool read_create_params(NVSDK_NGX_Parameter* params, FeatureState* s) {
    unsigned int w = 0, h = 0, ow = 0, oh = 0, pr = 0;
    if (params) {
        params->vtbl->GetUInt(params, "Width", &w);
        params->vtbl->GetUInt(params, "Height", &h);
        params->vtbl->GetUInt(params, "OutWidth", &ow);
        params->vtbl->GetUInt(params, "OutHeight", &oh);
        params->vtbl->GetUInt(params, "DLSS.Hint.Render.Preset.Quality", &pr);
    }
    if (!w || !h || !ow || !oh) {
        // diag-v1: the id names which CreateFeature call this was, so a refusal pairs up with the
        // (id=...) released/ok lines and cannot be mistaken for another build's CreateFeature.
        ngx_log("CreateFeature without Width/Height/OutWidth/OutHeight (%ux%u -> %ux%u, id=%u)",
                w, h, ow, oh, s->id);
        return false;
    }
    s->rw = w; s->rh = h; s->ow = ow; s->oh = oh;
    s->preset = (pr == 12) ? 'L' : 'M';
    if (s->feature == NVSDK_NGX_Feature_RayReconstruction) s->preset = 'F';
    return true;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_CreateFeature(
    ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Feature feature,
    NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out) {
    if (!out) return NVSDK_NGX_Result_Fail;
    *out = nullptr;

    if (!check_nvidia_dll(feature)) {
        ngx_log("D3D12 CreateFeature refused: %s is missing or not its expected size next to _nvngx.dll",
                feature == NVSDK_NGX_Feature_RayReconstruction ? "nvngx_dlssd.dll" : "nvngx_dlss.dll");
        return NVSDK_NGX_Result_Fail;
    }

    auto* s = new FeatureState();
    s->id = g_next_handle_id++;
    s->feature = feature;
    if (!read_create_params(params, s)) { delete s; return NVSDK_NGX_Result_Fail; }

    std::lock_guard<std::mutex> guard(g_bridge_lock);
    ID3D12Device* device = device_of(cmd);
    const bool got = ngxd_interop_device_for(device, &s->interop);
    if (device) device->Release();
    if (!got) {
        ngx_log("D3D12 CreateFeature: no vkd3d-proton interop device, so there is no Vulkan chain to run");
        delete s;
        return NVSDK_NGX_Result_Fail;
    }
    s->d3d_device = device_of(cmd);
    if (!build_chain(s)) {
        if (s->interop) s->interop->Release();
        delete s;
        return NVSDK_NGX_Result_Fail;
    }

    // diag-v1: the creation tuple with the id every later line is keyed by.
    ngx_log("D3D12 CreateFeature: id=%u feature=%d preset=%c %ux%u -> %ux%u [%s]", s->id, int(s->feature),
            s->preset, s->rw, s->rh, s->ow, s->oh, kPortTag);
    *out = reinterpret_cast<NVSDK_NGX_Handle*>(s);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_ReleaseFeature(NVSDK_NGX_Handle* handle) {
    if (!handle) return NVSDK_NGX_Result_Success;
    auto* s = reinterpret_cast<FeatureState*>(handle);
    std::lock_guard<std::mutex> guard(g_bridge_lock);
    // diag-v1: closes the id=... opened by CreateFeature; frames/eval show how far it got.
    ngx_log("ReleaseFeature: id=%u feature=%d released, %d frames, last eval #%llu", s->id,
            int(s->feature), s->frame_idx, s->eval_id);
    if (s->chain && vkDeviceWaitIdle) vkDeviceWaitIdle(g_vk_device);
    delete s->chain;
    s->chain = nullptr;
    if (s->interop) { s->interop->Release(); s->interop = nullptr; }
    if (s->d3d_device) { s->d3d_device->Release(); s->d3d_device = nullptr; }
    delete s;
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle* handle) {
    return NVSDK_NGX_ReleaseFeature(handle);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_ReleaseFeature(NVSDK_NGX_Handle* handle) {
    return NVSDK_NGX_ReleaseFeature(handle);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_EvaluateFeature(
    ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Handle* handle,
    NVSDK_NGX_Parameter* params, void* callback) {
    if (!handle || !params || !cmd) return NVSDK_NGX_Result_Fail;
    auto* s = reinterpret_cast<FeatureState*>(const_cast<NVSDK_NGX_Handle*>(handle));
    std::lock_guard<std::mutex> guard(g_bridge_lock);
    // diag-v1: one id per call, from here on stamped on every outcome this call can have.
    const unsigned long long eval_id = ++g_eval_seq;
    s->eval_id = eval_id;
    if (!s->chain || !s->interop) {
        ngx_log("eval #%llu FAIL: id=%u D3D12 EvaluateFeature with no chain/interop", eval_id, s->id);
        return NVSDK_NGX_Result_Fail;
    }

    ID3D12GraphicsCommandListExt* ext = nullptr;
    if (FAILED(cmd->QueryInterface(IID_ID3D12GraphicsCommandListExt, reinterpret_cast<void**>(&ext))) || !ext) {
        ngx_log("D3D12 EvaluateFeature: the command list has no ID3D12GraphicsCommandListExt (not vkd3d?), "
                "eval #%llu id=%u", eval_id, s->id);
        return NVSDK_NGX_Result_Fail;
    }
    VkCommandBuffer vkcmd = VK_NULL_HANDLE;
    ext->GetVulkanHandle(&vkcmd);
    ext->Release();
    if (!vkcmd) {
        ngx_log("eval #%llu FAIL: id=%u the command list gave no VkCommandBuffer", eval_id, s->id);
        return NVSDK_NGX_Result_Fail;
    }

    ngxd::FrameParams fp;
    ngxd::FrameChain* c = s->chain;
    BridgeImage ins[7], out_img;
    size_t n = 0;
    FrameDiag dg;
    // WHY THIS GUARD IS THE WHOLE BALLGAME: NVSDK_NGX_Result is an enum (Success 0x1,
    // Fail 0xBAD00000, both non-zero), so the original `if (outcome) return Fail;` never fired.  A
    // refused frame fell straight into record_bridge with a never-fetched Output (0x0, null VkImage)
    // and then reported SUCCESS to OptiScaler, which presented that never-written target: measured
    // as 190,364 refusal pairs paired 1:1 with 190,363 `SR frame ... -> 0x0` lines, and in-game as
    // menus/UI drawing normally while the whole 3D scene stayed black.  The guard now tests
    // ngx_ok() and propagates the real result, so a refusal returns Fail (the caller switches to
    // another upscaler instead of presenting a target nobody wrote) after log_eval_fail has named
    // the slot and reason above.  Shipping note: honest failure only looks like "DLSS stopped
    // working" until dxgi_to_vk decodes the game's DXGI_FORMAT_R32G32_FLOAT MotionVectors (the
    // reason every one of those calls refused) - these two changes have to land together.
    const NVSDK_NGX_Result frame_ok = collect_frame(params, s, false, c, fp, ins, &n, out_img, dg);
    if (!ngx_ok(frame_ok)) log_eval_fail(eval_id, s, n, dg, c);
    if (!ngx_ok(frame_ok)) return frame_ok;
    const NVSDK_NGX_Result got = record_bridge(vkcmd, s, fp, ins, n, out_img);
    if (got == NVSDK_NGX_Result_Success) {
        log_eval_ok(eval_id, s, n, ins, out_img, c, fp);
    } else {
        ngx_log("eval #%llu FAIL: id=%u record_bridge returned 0x%08x", eval_id, s->id, unsigned(got));
    }
    return got;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_RequiredExtensions(
    unsigned int* instCount, const char*** inst,
    unsigned int* devCount, const char*** dev) {
    if (instCount) *instCount = 0;
    if (inst) *inst = nullptr;
    if (devCount) *devCount = 0;
    if (dev) *dev = nullptr;
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_Init(
    unsigned long long appId, const wchar_t* dataPath, VkInstance instance,
    VkPhysicalDevice pd, VkDevice device, PFN_vkGetInstanceProcAddr gipa,
    PFN_vkGetDeviceProcAddr gdpa, const void* featureInfo, int sdkVersion) {
    g_vk_instance = instance;
    g_vk_phys_dev = pd;
    g_vk_device = device;
    if (gipa) g_gipa = gipa;
    if (gdpa) g_gdpa = gdpa;
    if (instance && device) load_vk_for_device(instance, pd, device);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_Shutdown1(VkDevice device) {
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_AllocateParameters(NVSDK_NGX_Parameter** out) {
    return NVSDK_NGX_D3D12_AllocateParameters(out);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_PopulateParameters_Impl(NVSDK_NGX_Parameter* p) {
    return NVSDK_NGX_D3D12_PopulateParameters_Impl(p);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_DestroyParameters(NVSDK_NGX_Parameter* p) {
    return NVSDK_NGX_D3D12_DestroyParameters(p);
}

static NVSDK_NGX_Result vulkan_create(VkDevice device, NVSDK_NGX_Feature feature,
                                      NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out) {
    if (!out) return NVSDK_NGX_Result_Fail;
    *out = nullptr;

    if (!check_nvidia_dll(feature)) {
        ngx_log("Vulkan CreateFeature refused: %s is missing or not its expected size next to _nvngx.dll",
                feature == NVSDK_NGX_Feature_RayReconstruction ? "nvngx_dlssd.dll" : "nvngx_dlss.dll");
        return NVSDK_NGX_Result_Fail;
    }

    auto* s = new FeatureState();
    s->id = g_next_handle_id++;
    s->feature = feature;
    if (!read_create_params(params, s)) { delete s; return NVSDK_NGX_Result_Fail; }

    std::lock_guard<std::mutex> guard(g_bridge_lock);
    if (device && device != g_vk_device) {
        // the host never called Init with this device: ask winevulkan where the physical device is
        g_vk_device = device;
    }
    if (!build_chain(s)) { delete s; return NVSDK_NGX_Result_Fail; }

    // diag-v1: the creation tuple with the id every later line is keyed by.
    ngx_log("Vulkan CreateFeature: id=%u feature=%d preset=%c %ux%u -> %ux%u [%s]", s->id, int(s->feature),
            s->preset, s->rw, s->rh, s->ow, s->oh, kPortTag);
    *out = reinterpret_cast<NVSDK_NGX_Handle*>(s);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_CreateFeature1(
    VkDevice device, VkCommandBuffer cmd, NVSDK_NGX_Feature feature,
    NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out) {
    return vulkan_create(device, feature, params, out);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_CreateFeature(
    VkDevice device, VkCommandBuffer cmd, NVSDK_NGX_Feature feature,
    NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out) {
    return vulkan_create(device, feature, params, out);
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_EvaluateFeature(
    VkCommandBuffer cmd, const NVSDK_NGX_Handle* handle,
    NVSDK_NGX_Parameter* params, void* callback) {
    if (!handle || !params || !cmd) return NVSDK_NGX_Result_Fail;
    auto* s = reinterpret_cast<FeatureState*>(const_cast<NVSDK_NGX_Handle*>(handle));
    std::lock_guard<std::mutex> guard(g_bridge_lock);
    // diag-v1: same call id contract as the D3D12 path above.
    const unsigned long long eval_id = ++g_eval_seq;
    s->eval_id = eval_id;
    if (!s->chain) {
        ngx_log("eval #%llu FAIL: id=%u Vulkan EvaluateFeature with no chain", eval_id, s->id);
        return NVSDK_NGX_Result_Fail;
    }

    ngxd::FrameParams fp;
    ngxd::FrameChain* c = s->chain;
    BridgeImage ins[7], out_img;
    size_t n = 0;
    FrameDiag dg;
    // Same fix as the D3D12 path: the enum-guard trap (Success 0x1 / Fail 0xBAD00000 both
    // non-zero) is avoided here too by testing ngx_ok() and propagating the collected result, with
    // log_eval_fail above naming the slot and reason.
    const NVSDK_NGX_Result frame_ok = collect_frame(params, s, true, c, fp, ins, &n, out_img, dg);
    if (!ngx_ok(frame_ok)) log_eval_fail(eval_id, s, n, dg, c);
    if (!ngx_ok(frame_ok)) return frame_ok;
    const NVSDK_NGX_Result got = record_bridge(cmd, s, fp, ins, n, out_img);
    if (got == NVSDK_NGX_Result_Success) {
        log_eval_ok(eval_id, s, n, ins, out_img, c, fp);
    } else {
        ngx_log("eval #%llu FAIL: id=%u record_bridge returned 0x%08x", eval_id, s->id, unsigned(got));
    }
    return got;
}

} // extern "C"

// The FFX surface's view of the device triple (declared extern "C++" in ngx_bridge_shared.h).
// Definitions sit OUTSIDE the exports' extern "C" block so the linkage matches the header.
VkInstance ngxd_vk_instance() { return g_vk_instance; }
VkPhysicalDevice ngxd_vk_phys_dev() { return g_vk_phys_dev; }
VkDevice ngxd_vk_device() { return g_vk_device; }

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_hinstance = hinst;
        g_log_t0 = ngx_ms();
        // Deliberately no AttachConsole(): taking the parent's console and freopen-ing stdout onto
        // CONOUT$ moves every printf in the process -- the host's own [ngx_client] lines included --
        // off the pipe the host redirected them to, and under Wine it can leave the process blocked in
        // the console handshake.  Everything this DLL reports goes to stderr and dlss5-amd-ngx.log.
    }
    return TRUE;
}
