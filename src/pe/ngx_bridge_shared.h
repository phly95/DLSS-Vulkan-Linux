// src/pe/ngx_bridge_shared.h -- the bridge internals shared between the two PE surfaces this
// package builds: the NGX bridge (src/pe/nvngx.cpp) and the FFX front end (src/ffx/ffx_front.cpp).
//
// Why a shared header: both surfaces must run the VERY SAME frame chain through the VERY SAME
// resource-acquisition machinery (one dxgi_to_vk map, one interop dance, one record_bridge), or
// the two packages drift apart.  Every function/variable here is defined in nvngx.cpp (the
// `static` dropped); nvngx.cpp's own exports and internals stay private to it.
//
// Both TUs compile into dlss5-amd-ffx.dll (src/ffx/build.sh); _nvngx.dll keeps building from
// nvngx.cpp + pipeline.cpp alone (src/vk/build.sh), byte-compatible as before.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <vulkan/vulkan.h>

#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <string>

#include "d3d12_interop.h"      // ID3D12DXVKInteropDevice / ID3D12GraphicsCommandListExt + IIDs
#include "../../tools/verify/ngx_client/ngx_abi.h"  // the NGX ABI subset (single source of truth)
#include "../vk/pipeline.h"     // ngxd::FrameChain, ngxd::FrameParams
#include "../vk/vkfuncs.h"

extern "C++" {

// One input or output image of a frame, with the layout the caller left it in.  nvngx.cpp's
// BridgeImage derives from it; the FFX surface fills BridgeImage_ directly.
struct BridgeImage_ {
    VkImage image = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    // Which aspect carries the texels the copies/barriers operate on: COLOR for every colour
    // image, DEPTH for vkd3d's depth-stencil images (a D32_SFLOAT_S8_UINT image has no COLOR
    // aspect, so an ASPECT_COLOR barrier there is a VUID violation, not a no-op - see the depth
    // audit in nvngx.cpp's d3d_image_resource).  On the copy this also decides the texel size:
    // the DEPTH aspect of D32_SFLOAT_S8_UINT is VK_FORMAT_D32_SFLOAT = 4 B/texel, not 8 B.
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    unsigned w = 0, h = 0;
    VkBuffer buf = VK_NULL_HANDLE;   // the chain's staging buffer for this slot
    const char* name = "";           // the slot's name (Color/MotionVectors/.../Output), for the log
    unsigned dxgi = 0;               // the DXGI format the D3D12 caller gave, 0 on the Vulkan path
};

// What a feature is, shared by both surfaces.  nvngx.cpp's FeatureState derives from it; the
// FFX surface uses it directly.
struct FeatureState_ {
    uint32_t id = 0;
    int feature = 0;                        // NVSDK_NGX_Feature as plain int on the shared surface
    char preset = 'M';
    uint32_t rw = 0, rh = 0, ow = 0, oh = 0;
    int frame_idx = 0;
    unsigned long long eval_id = 0;
    ngxd::FrameChain* chain = nullptr;
    float prev_jx = 0.0f, prev_jy = 0.0f;   // the caller hands the current jitter only
    ID3D12DXVKInteropDevice* interop = nullptr;
    ID3D12Device* d3d_device = nullptr;     // AddRef'd
    mutable char s6_logged = 0;             // S6: one [S6] mv-scale line per feature
};

// The timestamp clock every log line uses (nvngx.cpp: ngx_ms).
extern "C" unsigned long long ngx_ms();
// The bridge log: stderr + the CWD's dlss5-amd-ngx.log - ONE file for both surfaces; the FFX
// surface prefixes its fmt with "[ffx] " so the harness gates grep one surface exactly.
extern "C" void ngx_log(const char* fmt, ...);
// DllMain's HINSTANCE (module_path_tail users).
extern HINSTANCE ngxd_hinstance;
// DLSS5_AMD_WEIGHTS resolved the way the Win32 CRT can open it (nvngx.cpp: ngxd_weights_root);
// returns the raw value, may be empty.  C++-linkage on purpose (std::string by value).
std::string ngxd_weights_root();

// The DXGI formats the port's gather shaders decode (nvngx.cpp: dxgi_to_vk).
VkFormat ngxd_dxgi_to_vk(DXGI_FORMAT f);
// The vkd3d-proton interop device from an ID3D12Device: fills g_vk_instance/phys/device and loads
// the entry points (nvngx.cpp: interop_device_for).
extern "C" bool ngxd_interop_device_for(ID3D12Device* device, ID3D12DXVKInteropDevice** out);
// The device triple the interop dance filled (nvngx.cpp g_vk_instance/g_vk_phys_dev/g_vk_device;
// build_chain uses exactly these three).  The FFX surface's chain build goes through them too.
extern "C++" VkInstance ngxd_vk_instance();
extern "C++" VkPhysicalDevice ngxd_vk_phys_dev();
extern "C++" VkDevice ngxd_vk_device();
// The record side: copy the caller's images into the chain's staging buffers, record the chain,
// copy the result back out (nvngx.cpp: record_bridge, renamed forwarder ngxd_record_bridge).
NVSDK_NGX_Result ngxd_record_bridge(VkCommandBuffer cmd, FeatureState_* s, const ngxd::FrameParams& fp,
                                    const BridgeImage_* ins, size_t n, const BridgeImage_* out);
// The per-frame OK line (diag-v1): the SAME emitter the NGX export uses (nvngx.cpp: log_eval_ok),
// so both surfaces write the one grammar the harness g2 gate greps - `eval #N OK: ... fmt col=..
// mv=.. dep=..`.  Takes the next id off the shared eval sequence; returns it for the caller's own
// [ffx]-prefixed 1 Hz line.
unsigned long long ngxd_log_eval_ok(FeatureState_* s, size_t n, const BridgeImage_* ins,
                                    const BridgeImage_* out, const ngxd::FrameChain* c,
                                    const ngxd::FrameParams& fp);

}  // extern "C++"
