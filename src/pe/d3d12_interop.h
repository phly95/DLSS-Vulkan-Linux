// src/pe/d3d12_interop.h -- the vkd3d-proton interop interface declarations.
//
// No d3d12 header in this tree (or the mingw one) carries these two, so they are written out by
// hand - GUIDs/IIDs/prototypes as vkd3d-proton's implementations answer them
// (ID3D12DXVKInteropDevice = winevulkan's dxvk interop; ID3D12GraphicsCommandListExt =
// vkd3d-proton's command-list extension, which GetVulkanHandle uses).
// Both PE surfaces (src/pe/nvngx.cpp, src/ffx/ffx_front.cpp) include this header.
#pragma once

#include <windows.h>
#include <d3d12.h>

extern const GUID ngxd_IID_ID3D12GraphicsCommandListExt;
extern const GUID ngxd_IID_ID3D12DXVKInteropDevice;

MIDL_INTERFACE("77a86b09-2bea-4801-b89a-37648e104af1")
ID3D12GraphicsCommandListExt : public IUnknown
{
public:
    virtual HRESULT STDMETHODCALLTYPE GetVulkanHandle(VkCommandBuffer * pVkCommandBuffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE LaunchCubinShader(void* handle, UINT32 x, UINT32 y, UINT32 z,
                                                        const void* params, UINT32 param_size) = 0;
};

MIDL_INTERFACE("39da4e09-bd1c-4198-9fae-86bbe3be41fd")
ID3D12DXVKInteropDevice : public IUnknown
{
public:
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
