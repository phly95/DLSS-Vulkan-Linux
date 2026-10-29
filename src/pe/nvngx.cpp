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
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <fstream>
#include <algorithm>

#include "../../tools/verify/ngx_client/ngx_abi.h"
#include "../vk/pipeline.h"
#include "../vk/vkfuncs.h"

// vkd3d-proton interop definitions
static const GUID IID_ID3D12GraphicsCommandListExt = 
    { 0x77a86b09, 0x2bea, 0x4801, { 0xb8, 0x9a, 0x37, 0x64, 0x8e, 0x10, 0x4a, 0xf1 } };

static const GUID IID_ID3D12DXVKInteropDevice = 
    { 0x39da4e09, 0xbd1c, 0x4198, { 0x9f, 0xae, 0x86, 0xbb, 0xe3, 0xbe, 0x41, 0xfd } };

MIDL_INTERFACE("77a86b09-2bea-4801-b89a-37648e104af1")
ID3D12GraphicsCommandListExt : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHandle(VkCommandBuffer * pVkCommandBuffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE LaunchCubinShader(void* handle, UINT32 x, UINT32 y, UINT32 z,
                                                        const void* params, UINT32 param_size) = 0;
};

MIDL_INTERFACE("39da4e09-bd1c-4198-9fae-86bbe3be41fd")
ID3D12DXVKInteropDevice : public IUnknown
{
    virtual HRESULT STDMETHODCALLTYPE GetDXGIAdapter(REFIID iid, void** ppvObject) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetInstanceExtensions(UINT* pExtensionCount, const char** ppExtensions) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceExtensions(UINT* pExtensionCount, const char** ppExtensions) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetDeviceFeatures(const VkPhysicalDeviceFeatures2** ppFeatures) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHandles(VkInstance* pVkInstance, VkPhysicalDevice* pVkPhysicalDevice,
                                                       VkDevice* pVkDevice) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanQueueInfo(ID3D12CommandQueue* pCommandQueue, VkQueue* pVkQueue,
                                                         UINT32* pVkQueueFamily) = 0;
    virtual void STDMETHODCALLTYPE GetVulkanImageLayout(ID3D12Resource* pResource, D3D12_RESOURCE_STATES State,
                                                        VkImageLayout* pVkLayout) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetVulkanResourceInfo(ID3D12Resource* pResource, UINT64* pVkHandle,
                                                            UINT64* pBufferOffset) = 0;
    virtual HRESULT STDMETHODCALLTYPE LockCommandQueue(ID3D12CommandQueue* pCommandQueue) = 0;
    virtual HRESULT STDMETHODCALLTYPE UnlockCommandQueue(ID3D12CommandQueue* pCommandQueue) = 0;
};

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

static HINSTANCE g_hinstance = NULL;

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

static unsigned long long g_log_t0 = 0;
static unsigned long long ngx_ms() { return GetTickCount64(); }

static void ngx_log(const char* fmt, ...) {
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
static std::string weights_root() {
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

struct FeatureState {
    uint32_t id = 0;
    NVSDK_NGX_Feature feature = NVSDK_NGX_Feature_SuperSampling;
    char preset = 'M';
    uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
    int frame_idx = 0;
    // the bridge
    ngxd::FrameChain* chain = nullptr;
    float prev_jx = 0.0f, prev_jy = 0.0f;   // NGX hands the current jitter only; the chain needs both
    ID3D12DXVKInteropDevice* interop = nullptr;   // D3D12 path only
    ID3D12Device* d3d_device = nullptr;           // D3D12 path only (AddRef'd)
};

// One input or output image of a frame, with the layout the caller left it in.
struct BridgeImage {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    unsigned w = 0, h = 0;
    VkBuffer buf = VK_NULL_HANDLE;   // the chain's staging buffer for this slot
};

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
    const std::string root = weights_root();
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
static bool want_format(const char* key, VkFormat got, VkFormat want) {
    if (got == want) return true;
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
    // The chain works on whole frames.  A subrect that is not the whole render target would need the
    // gather to read a window of the image, which the port does not have; refuse instead of guessing.
    int sw = int(s->rw), sh = int(s->rh);
    if (ngx_ok(p->vtbl->GetInt(p, "DLSS.Render.Subrect.Dimensions.Width", &i))) sw = i;
    if (ngx_ok(p->vtbl->GetInt(p, "DLSS.Render.Subrect.Dimensions.Height", &i))) sh = i;
    if (sw != int(s->rw) || sh != int(s->rh)) {
        ngx_log("subrect %dx%d is not the whole %ux%u render target: unsupported", sw, sh, s->rw, s->rh);
        return false;
    }
    // ngx_client feeds the default formats only (its Input table), which the gather shaders call code 0;
    // MV.Scale is not read by the chain, exactly as in the CLI runner.
    fp.fmt_col = fp.fmt_mv = fp.fmt_dep = 0;
    return true;
}

static void image_barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout old_l, VkImageLayout new_l,
                          VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.oldLayout = old_l;
    b.newLayout = new_l;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         0, nullptr, 0, nullptr, 1, &b);
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
    if (s->frame_idx > 0) {
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
    for (size_t k = 0; k < n; ++k) {
        image_barrier(cmd, ins[k].image, ins[k].layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readable,
                      VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy cp{};
        cp.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cp.imageExtent = {ins[k].w, ins[k].h, 1};
        vkCmdCopyImageToBuffer(cmd, ins[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ins[k].buf, 1, &cp);
        image_barrier(cmd, ins[k].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ins[k].layout,
                      VK_ACCESS_TRANSFER_READ_BIT, readable);
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

    c->advance();
    s->frame_idx++;
    s->prev_jx = fp.jx;
    s->prev_jy = fp.jy;
    ngx_log("%s frame %d: %zu images in, %dx%d -> %dx%d, recorded in %llu ms", feature_name(s),
            s->frame_idx - 1, n, c->width(), c->height(), out.w, out.h, ngx_ms() - t0);
    return NVSDK_NGX_Result_Success;
}

// The Vulkan side of "hand me this image": NGX_Resource_VK as ngx_client sets it.
static bool vk_image_resource(NVSDK_NGX_Parameter* p, const char* key, BridgeImage& o) {
    void* v = nullptr;
    if (!ngx_ok(p->vtbl->GetVoidPtr(p, key, &v)) || !v) return false;
    const auto* r = reinterpret_cast<const NGX_Resource_VK*>(v);
    if (r->Type != NGX_RESOURCE_VK_IMAGEVIEW || !r->Resource.ImageViewInfo.Image) return false;
    o.image = r->Resource.ImageViewInfo.Image;
    o.format = r->Resource.ImageViewInfo.Format;
    o.w = r->Resource.ImageViewInfo.Width;
    o.h = r->Resource.ImageViewInfo.Height;
    return true;
}

// The D3D12 side of the same: vkd3d-proton hands the VkImage out through its interop device.
static VkFormat dxgi_to_vk(DXGI_FORMAT f) {
    switch (f) {
        case DXGI_FORMAT_R16G16B16A16_FLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
        case DXGI_FORMAT_R16G16_FLOAT: return VK_FORMAT_R16G16_SFLOAT;
        case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
        case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
        default: return VK_FORMAT_UNDEFINED;
    }
}

static bool d3d_image_resource(FeatureState* s, ID3D12Resource* res, D3D12_RESOURCE_STATES state,
                               const char* key, BridgeImage& o) {
    if (!res || !s->interop) return false;
    UINT64 handle = 0, offset = 0;
    if (FAILED(s->interop->GetVulkanResourceInfo(res, &handle, &offset)) || !handle) return false;
    const auto desc = res->GetDesc();
    o.image = (VkImage)handle;
    o.format = dxgi_to_vk(desc.Format);
    o.w = unsigned(desc.Width);
    o.h = unsigned(desc.Height);
    s->interop->GetVulkanImageLayout(res, state, &o.layout);
    if (o.format == VK_FORMAT_UNDEFINED) {
        ngx_log("%s: DXGI format %u is not one this bridge decodes", key, unsigned(desc.Format));
        return false;
    }
    if (o.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        ngx_log("%s: vkd3d reported an undefined layout", key);
        return false;
    }
    return true;
}

// Which way an NGX parameter hands over an image: Set(key, NGX_Resource_VK*) on the Vulkan path,
// Set(key, ID3D12Resource*) on the D3D12 path.  Either way the slot's staging buffer and the port's
// format code are the same, so both paths collect their frames the same way.
static bool fetch_image(NVSDK_NGX_Parameter* p, FeatureState* s, bool vulkan, const char* key, const char* name,
                        VkFormat want, VkBuffer buf, D3D12_RESOURCE_STATES state, BridgeImage& o) {
    o = BridgeImage{};
    o.buf = buf;
    if (vulkan) {
        if (!vk_image_resource(p, key, o)) {
            ngx_log("Evaluate: no %s image", name);
            return false;
        }
    } else {
        ID3D12Resource* res = nullptr;
        p->vtbl->GetD3D12Resource(p, key, &res);
        if (!res || !d3d_image_resource(s, res, state, name, o)) {
            ngx_log("Evaluate: no %s resource", name);
            return false;
        }
    }
    return want_format(name, o.format, want);
}

struct Slot { const char* key; const char* name; VkFormat fmt; };

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
                                      BridgeImage* ins, size_t* n, BridgeImage& out) {
    if (!read_frame_params(p, s, fp)) return NVSDK_NGX_Result_Fail;

    const Slot common[] = {
        {"Color", "Color", VK_FORMAT_R16G16B16A16_SFLOAT},
        {"MotionVectors", "MotionVectors", VK_FORMAT_R16G16_SFLOAT},
        {"Depth", "Depth", VK_FORMAT_R32_SFLOAT},
    };
    const Slot extra[] = {
        {"GBuffer.Normals", "GBuffer.Normals", VK_FORMAT_R16G16B16A16_SFLOAT},
        {"DLSS.Input.DiffuseAlbedo", "DiffuseAlbedo", VK_FORMAT_R16G16B16A16_SFLOAT},
        {"DLSS.Input.SpecularAlbedo", "SpecularAlbedo", VK_FORMAT_R16G16B16A16_SFLOAT},
        {"GBuffer.Roughness", "Roughness", VK_FORMAT_R32_SFLOAT},
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
        if (!fetch_image(p, s, vulkan, common[k].key, common[k].name, common[k].fmt, bufs[k], input_state,
                         ins[*n]))
            return NVSDK_NGX_Result_Fail;
        ++*n;
    }
    if (rr) {
        for (size_t k = 0; k < 4; ++k) {
            if (input_skipped(extra_short[k])) continue;
            if (!fetch_image(p, s, vulkan, extra[k].key, extra[k].name, extra[k].fmt, extra_bufs[k],
                             input_state, ins[*n]))
                return NVSDK_NGX_Result_Fail;
            ++*n;
        }
    }

    // M2: the app's Output must match the chain's out buffer element format (the buffer->image
    // copy does no conversion; bufferRowLength=0 packs in IMAGE texels). NGX's output resource is
    // normally R16G16B16A16_FLOAT - that is the f16 contract; 0 keeps the old RGBA32F contract.
#if DLSS5_AMD_OUT_F16
    const VkFormat out_fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
#else
    const VkFormat out_fmt = VK_FORMAT_R32G32B32A32_SFLOAT;
#endif
    if (!fetch_image(p, s, vulkan, "Output", "Output", out_fmt, c->out_color_buf(),
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, out))
        return NVSDK_NGX_Result_Fail;
    if (out.w != unsigned(c->out_width()) || out.h != unsigned(c->out_height())) {
        ngx_log("Output is %ux%u, the feature is %dx%u", out.w, out.h, c->out_width(), c->out_height());
        return NVSDK_NGX_Result_Fail;
    }
    return NVSDK_NGX_Result_Success;
}

// Exports
extern "C" {

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Init(
    unsigned long long appId, const wchar_t* dataPath, ID3D12Device* device,
    const void* featureInfo, int sdkVersion) {
    if (device && !g_d3d12_device) { g_d3d12_device = device; g_d3d12_device->AddRef(); }
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* device) {
    if (g_d3d12_device) { g_d3d12_device->Release(); g_d3d12_device = nullptr; }
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** out) {
    if (!out) return NVSDK_NGX_Result_Fail;
    auto* p = new ParametersImpl();
    *out = p->as_param();
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* p) {
    if (p) delete reinterpret_cast<ParametersImpl*>(p);
    return NVSDK_NGX_Result_Success;
}

// The Vulkan chain needs a VkDevice; on the D3D12 path that is vkd3d-proton's, reached through its
// interop interface on the same device the game gave us.
static bool interop_device_for(ID3D12Device* device, ID3D12DXVKInteropDevice** out) {
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
        ngx_log("CreateFeature without Width/Height/OutWidth/OutHeight (%ux%u -> %ux%u)", w, h, ow, oh);
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
    const bool got = interop_device_for(device, &s->interop);
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

    *out = reinterpret_cast<NVSDK_NGX_Handle*>(s);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_ReleaseFeature(NVSDK_NGX_Handle* handle) {
    if (!handle) return NVSDK_NGX_Result_Success;
    auto* s = reinterpret_cast<FeatureState*>(handle);
    std::lock_guard<std::mutex> guard(g_bridge_lock);
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
    if (!s->chain || !s->interop) return NVSDK_NGX_Result_Fail;

    ID3D12GraphicsCommandListExt* ext = nullptr;
    if (FAILED(cmd->QueryInterface(IID_ID3D12GraphicsCommandListExt, reinterpret_cast<void**>(&ext))) || !ext) {
        ngx_log("D3D12 EvaluateFeature: the command list has no ID3D12GraphicsCommandListExt (not vkd3d?)");
        return NVSDK_NGX_Result_Fail;
    }
    VkCommandBuffer vkcmd = VK_NULL_HANDLE;
    ext->GetVulkanHandle(&vkcmd);
    ext->Release();
    if (!vkcmd) return NVSDK_NGX_Result_Fail;

    ngxd::FrameParams fp;
    ngxd::FrameChain* c = s->chain;
    BridgeImage ins[7], out_img;
    size_t n = 0;
    if (!collect_frame(params, s, false, c, fp, ins, &n, out_img)) return NVSDK_NGX_Result_Fail;
    return record_bridge(vkcmd, s, fp, ins, n, out_img);
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
    if (!s->chain) return NVSDK_NGX_Result_Fail;

    ngxd::FrameParams fp;
    ngxd::FrameChain* c = s->chain;
    BridgeImage ins[7], out_img;
    size_t n = 0;
    if (!collect_frame(params, s, true, c, fp, ins, &n, out_img)) return NVSDK_NGX_Result_Fail;
    return record_bridge(cmd, s, fp, ins, n, out_img);
}

} // extern "C"

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
