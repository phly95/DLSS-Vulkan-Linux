#pragma once

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#define NGX_MSABI __attribute__((ms_abi))
#else
#define NGX_MSABI
#endif

struct ID3D12Device;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;
struct ID3D11Resource;

using NrVkInstance = void*;
using NrVkPhysicalDevice = void*;
using NrVkDevice = void*;
using NrVkCommandBuffer = void*;

typedef enum NVSDK_NGX_Result {
    NVSDK_NGX_Result_Success = 0x1,
    NVSDK_NGX_Result_Fail = 0xBAD00000,
    NVSDK_NGX_Result_FAIL_FeatureNotSupported = NVSDK_NGX_Result_Fail | 1,
    NVSDK_NGX_Result_FAIL_PlatformError = NVSDK_NGX_Result_Fail | 2,
    NVSDK_NGX_Result_FAIL_FeatureAlreadyExists = NVSDK_NGX_Result_Fail | 3,
    NVSDK_NGX_Result_FAIL_FeatureNotFound = NVSDK_NGX_Result_Fail | 4,
    NVSDK_NGX_Result_FAIL_InvalidParameter = NVSDK_NGX_Result_Fail | 5,
    NVSDK_NGX_Result_FAIL_NotInitialized = NVSDK_NGX_Result_Fail | 7,
    NVSDK_NGX_Result_FAIL_UnableToInitializeFeature = NVSDK_NGX_Result_Fail | 11,
} NVSDK_NGX_Result;

typedef enum NVSDK_NGX_Feature {
    NVSDK_NGX_Feature_SuperSampling = 1,
    NVSDK_NGX_Feature_RayReconstruction = 13,
} NVSDK_NGX_Feature;

typedef struct NVSDK_NGX_Handle {
    unsigned int Id;
} NVSDK_NGX_Handle;

struct NVSDK_NGX_Parameter;

struct NVSDK_NGX_ParameterVtbl {
    void (NGX_MSABI* SetVoidPtr)(NVSDK_NGX_Parameter*, const char*, void*);
    void (NGX_MSABI* SetD3D12Resource)(NVSDK_NGX_Parameter*, const char*, ID3D12Resource*);
    void (NGX_MSABI* SetD3D11Resource)(NVSDK_NGX_Parameter*, const char*, ID3D11Resource*);
    void (NGX_MSABI* SetInt)(NVSDK_NGX_Parameter*, const char*, int);
    void (NGX_MSABI* SetUInt)(NVSDK_NGX_Parameter*, const char*, unsigned int);
    void (NGX_MSABI* SetDouble)(NVSDK_NGX_Parameter*, const char*, double);
    void (NGX_MSABI* SetFloat)(NVSDK_NGX_Parameter*, const char*, float);
    void (NGX_MSABI* SetULL)(NVSDK_NGX_Parameter*, const char*, unsigned long long);

    NVSDK_NGX_Result (NGX_MSABI* GetVoidPtr)(const NVSDK_NGX_Parameter*, const char*, void**);
    NVSDK_NGX_Result (NGX_MSABI* GetD3D12Resource)(const NVSDK_NGX_Parameter*, const char*, ID3D12Resource**);
    NVSDK_NGX_Result (NGX_MSABI* GetD3D11Resource)(const NVSDK_NGX_Parameter*, const char*, ID3D11Resource**);
    NVSDK_NGX_Result (NGX_MSABI* GetInt)(const NVSDK_NGX_Parameter*, const char*, int*);
    NVSDK_NGX_Result (NGX_MSABI* GetUInt)(const NVSDK_NGX_Parameter*, const char*, unsigned int*);
    NVSDK_NGX_Result (NGX_MSABI* GetDouble)(const NVSDK_NGX_Parameter*, const char*, double*);
    NVSDK_NGX_Result (NGX_MSABI* GetFloat)(const NVSDK_NGX_Parameter*, const char*, float*);
    NVSDK_NGX_Result (NGX_MSABI* GetULL)(const NVSDK_NGX_Parameter*, const char*, unsigned long long*);

    void (NGX_MSABI* Reset)(NVSDK_NGX_Parameter*);
};

struct NVSDK_NGX_Parameter {
    const NVSDK_NGX_ParameterVtbl* vtbl;
    void Set(const char* k, void* v) { vtbl->SetVoidPtr(this, k, v); }
    void Set(const char* k, ID3D12Resource* v) { vtbl->SetD3D12Resource(this, k, v); }
    void Set(const char* k, int v) { vtbl->SetInt(this, k, v); }
    void Set(const char* k, unsigned int v) { vtbl->SetUInt(this, k, v); }
    void Set(const char* k, float v) { vtbl->SetFloat(this, k, v); }
    void Set(const char* k, double v) { vtbl->SetDouble(this, k, v); }
    void Set(const char* k, unsigned long long v) { vtbl->SetULL(this, k, v); }
};

// D3D12 Entry Points
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_D3D12_Init)(unsigned long long appId, const wchar_t* logDir, ID3D12Device* device);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_D3D12_Shutdown)(void);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_D3D12_GetCapabilityParameters)(NVSDK_NGX_Parameter** out);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_D3D12_CreateFeature)(ID3D12GraphicsCommandList* cmd, NVSDK_NGX_Feature featureId, NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_D3D12_ReleaseFeature)(NVSDK_NGX_Handle* handle);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_D3D12_EvaluateFeature)(ID3D12GraphicsCommandList* cmd, const NVSDK_NGX_Handle* handle, const NVSDK_NGX_Parameter* params, void* callback);

// Vulkan Entry Points
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_VULKAN_Init)(unsigned long long appId, const wchar_t* logDir, NrVkInstance instance, NrVkPhysicalDevice physDevice, NrVkDevice device);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_VULKAN_Shutdown)(void);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_VULKAN_GetCapabilityParameters)(NVSDK_NGX_Parameter** out);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_VULKAN_CreateFeature1)(NrVkDevice device, NrVkCommandBuffer cmd, NVSDK_NGX_Feature featureId, NVSDK_NGX_Parameter* params, NVSDK_NGX_Handle** out);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_VULKAN_ReleaseFeature)(NVSDK_NGX_Handle* handle);
typedef NVSDK_NGX_Result (NGX_MSABI* PFN_NVSDK_NGX_VULKAN_EvaluateFeature)(NrVkCommandBuffer cmd, const NVSDK_NGX_Handle* handle, const NVSDK_NGX_Parameter* params, void* callback);
