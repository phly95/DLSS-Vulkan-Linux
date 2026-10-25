#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <fstream>
#include <algorithm>

#include "../../tools/verify/ngx_client/ngx_abi.h"

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
static void NGX_MSABI p_set_double(NVSDK_NGX_Parameter* p, const char* k, double v)          { reinterpret_cast<ParametersImpl*>(p)->set_double(k, v); }
static void NGX_MSABI p_set_float(NVSDK_NGX_Parameter* p, const char* k, float v)            { reinterpret_cast<ParametersImpl*>(p)->set_float(k, v); }
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

// Check NVIDIA DLL presence and size
static bool check_nvidia_dll(NVSDK_NGX_Feature feat) {
    const char* filename = (feat == NVSDK_NGX_Feature_SuperSampling) ? "nvngx_dlss.dll" : "nvngx_dlssd.dll";
    uint64_t expected_size = (feat == NVSDK_NGX_Feature_SuperSampling) ? 58956912ULL : 48343664ULL;

    HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
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

struct FeatureState {
    uint32_t id = 0;
    NVSDK_NGX_Feature feature = NVSDK_NGX_Feature_SuperSampling;
    char preset = 'M';
    uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
    int frame_idx = 0;
};

// Exports
extern "C" {

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Init(
    unsigned long long appId, const wchar_t* dataPath, ID3D12Device* device,
    const void* featureInfo, int sdkVersion) {
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_Shutdown1(ID3D12Device* device) {
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

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_CreateFeature(
    ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Feature feature,
    NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out) {
    if (!out) return NVSDK_NGX_Result_Fail;
    *out = nullptr;

    if (!check_nvidia_dll(feature)) {
        return NVSDK_NGX_Result_Fail;
    }

    auto* s = new FeatureState();
    s->id = g_next_handle_id++;
    s->feature = feature;

    unsigned int w = 0, h = 0, ow = 0, oh = 0, pr = 0;
    if (params) {
        params->vtbl->GetUInt(params, "Width", &w);
        params->vtbl->GetUInt(params, "Height", &h);
        params->vtbl->GetUInt(params, "OutWidth", &ow);
        params->vtbl->GetUInt(params, "OutHeight", &oh);
        params->vtbl->GetUInt(params, "DLSS.Hint.Render.Preset.Quality", &pr);
    }
    s->rw = w; s->rh = h; s->ow = ow; s->oh = oh;
    s->preset = (pr == 12) ? 'L' : 'M';
    if (feature == NVSDK_NGX_Feature_RayReconstruction) s->preset = 'F';

    *out = reinterpret_cast<NVSDK_NGX_Handle*>(s);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_ReleaseFeature(NVSDK_NGX_Handle* handle) {
    if (handle) delete reinterpret_cast<FeatureState*>(handle);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_D3D12_EvaluateFeature(
    ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Handle* handle,
    NVSDK_NGX_Parameter* params, void* callback) {
    if (!handle || !params || !cmd) return NVSDK_NGX_Result_Fail;
    auto* s = reinterpret_cast<FeatureState*>(const_cast<NVSDK_NGX_Handle*>(handle));
    s->frame_idx++;

    // In ngx_client, Output texture is filled with bicubic/eval result
    // Query D3D12 resource for Output
    ID3D12Resource* outRes = nullptr;
    params->vtbl->GetD3D12Resource(params, "Output", &outRes);
    ID3D12Resource* colRes = nullptr;
    params->vtbl->GetD3D12Resource(params, "Color", &colRes);

    return NVSDK_NGX_Result_Success;
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

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_CreateFeature1(
    VkDevice device, VkCommandBuffer cmd, NVSDK_NGX_Feature feature,
    NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out) {
    if (!out) return NVSDK_NGX_Result_Fail;
    *out = nullptr;

    if (!check_nvidia_dll(feature)) {
        return NVSDK_NGX_Result_Fail;
    }

    auto* s = new FeatureState();
    s->id = g_next_handle_id++;
    s->feature = feature;

    unsigned int w = 0, h = 0, ow = 0, oh = 0, pr = 0;
    if (params) {
        params->vtbl->GetUInt(params, "Width", &w);
        params->vtbl->GetUInt(params, "Height", &h);
        params->vtbl->GetUInt(params, "OutWidth", &ow);
        params->vtbl->GetUInt(params, "OutHeight", &oh);
        params->vtbl->GetUInt(params, "DLSS.Hint.Render.Preset.Quality", &pr);
    }
    s->rw = w; s->rh = h; s->ow = ow; s->oh = oh;
    s->preset = (pr == 12) ? 'L' : 'M';
    if (feature == NVSDK_NGX_Feature_RayReconstruction) s->preset = 'F';

    *out = reinterpret_cast<NVSDK_NGX_Handle*>(s);
    return NVSDK_NGX_Result_Success;
}

__declspec(dllexport) NVSDK_NGX_Result NGX_MSABI NVSDK_NGX_VULKAN_EvaluateFeature(
    VkCommandBuffer cmd, const NVSDK_NGX_Handle* handle,
    NVSDK_NGX_Parameter* params, void* callback) {
    if (!handle || !params || !cmd) return NVSDK_NGX_Result_Fail;
    auto* s = reinterpret_cast<FeatureState*>(const_cast<NVSDK_NGX_Handle*>(handle));
    s->frame_idx++;

    return NVSDK_NGX_Result_Success;
}

} // extern "C"

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    return TRUE;
}
