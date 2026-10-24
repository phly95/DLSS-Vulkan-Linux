// ngx_client: Windows mingw test client for DLSS NGX SuperSampling (SR) and RayReconstruction (RR).
// Uses only public NGX entry points (Init, GetCapabilityParameters, CreateFeature, EvaluateFeature, ReleaseFeature, Shutdown)
//
// Usage:
//   ngx_client.exe --api <d3d12|vulkan> --feature <sr|rr> --frames <frames_dir> --out <out_dir> [--test-cycles]
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ngx_abi.h"

#define HR_CHECK(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { \
    std::fprintf(stderr, "%s:%d HRESULT failed: 0x%08lx\n", __FILE__, __LINE__, (unsigned long)hr_); return 1; } } while(0)

#define NGX_CHECK(x) do { NVSDK_NGX_Result r_ = (x); if (r_ != NVSDK_NGX_Result_Success) { \
    std::fprintf(stderr, "%s:%d NGX call failed: 0x%08x\n", __FILE__, __LINE__, r_); return 1; } } while(0)

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

static void write_file(const std::string& p, const void* d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(d), std::streamsize(n));
}

static std::vector<double> jget(const std::string& txt, const std::string& key) {
    std::vector<double> v;
    size_t k = txt.find("\"" + key + "\"");
    if (k == std::string::npos) return v;
    size_t i = txt.find(':', k) + 1;
    while (i < txt.size() && (txt[i] == ' ' || txt[i] == '[' || txt[i] == '\n')) ++i;
    for (;;) {
        char* e = nullptr;
        double d = std::strtod(txt.c_str() + i, &e);
        if (e == txt.c_str() + i) {
            if (txt.compare(i, 4, "true") == 0) { v.push_back(1); i += 4; }
            else if (txt.compare(i, 5, "false") == 0) { v.push_back(0); i += 5; }
            else break;
        } else { v.push_back(d); i = size_t(e - txt.c_str()); }
        while (i < txt.size() && (txt[i] == ' ' || txt[i] == ',' || txt[i] == '\n')) ++i;
        if (i >= txt.size() || txt[i] == ']' || txt[i] == '}') break;
    }
    return v;
}

// Helpers for D3D12 resource creation and upload
static ID3D12Resource* create_texture(ID3D12Device* dev, UINT w, UINT h, DXGI_FORMAT fmt, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = fmt;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* res = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res));
    return res;
}

static ID3D12Resource* create_buffer(ID3D12Device* dev, UINT64 size, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    ID3D12Resource* res = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&res));
    return res;
}

static void upload_to_texture(ID3D12Device* dev, ID3D12GraphicsCommandList* cmd, ID3D12Resource* dst,
                              ID3D12Resource* upload, const void* data, UINT w, UINT h, UINT bpp) {
    D3D12_SUBRESOURCE_FOOTPRINT fp{};
    fp.Format = dst->GetDesc().Format;
    fp.Width = w;
    fp.Height = h;
    fp.Depth = 1;
    fp.RowPitch = (w * bpp + 255) & ~255;

    void* ptr = nullptr;
    upload->Map(0, nullptr, &ptr);
    const uint8_t* src = static_cast<const uint8_t*>(data);
    uint8_t* dptr = static_cast<uint8_t*>(ptr);
    for (UINT y = 0; y < h; ++y) {
        std::memcpy(dptr + y * fp.RowPitch, src + y * (w * bpp), w * bpp);
    }
    upload->Unmap(0, nullptr);

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = upload;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    srcLoc.PlacedFootprint.Offset = 0;
    srcLoc.PlacedFootprint.Footprint = fp;

    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = dst;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dstLoc.SubresourceIndex = 0;

    cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
}

// Main runner for D3D12
static int run_d3d12(NVSDK_NGX_Feature feat, const std::string& fdir, const std::string& odir, bool test_cycles) {
    std::cout << "[ngx_client] Running D3D12 path..." << std::endl;
    HMODULE ngx = LoadLibraryA("_nvngx.dll");
    if (!ngx) ngx = LoadLibraryA("nvngx.dll");
    if (!ngx) ngx = LoadLibraryA("dlssnr_core.dll");
    if (!ngx) {
        std::fprintf(stderr, "Failed to load NGX library (_nvngx.dll / nvngx.dll / dlssnr_core.dll)\n");
        return 1;
    }

    auto fn_init = (PFN_NVSDK_NGX_D3D12_Init)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Init");
    auto fn_shutdown = (PFN_NVSDK_NGX_D3D12_Shutdown)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Shutdown");
    auto fn_get_params = (PFN_NVSDK_NGX_D3D12_GetCapabilityParameters)GetProcAddress(ngx, "NVSDK_NGX_D3D12_GetCapabilityParameters");
    auto fn_create = (PFN_NVSDK_NGX_D3D12_CreateFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_CreateFeature");
    auto fn_release = (PFN_NVSDK_NGX_D3D12_ReleaseFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_ReleaseFeature");
    auto fn_evaluate = (PFN_NVSDK_NGX_D3D12_EvaluateFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_EvaluateFeature");

    if (!fn_init || !fn_get_params || !fn_create || !fn_release || !fn_evaluate) {
        std::fprintf(stderr, "Failed to resolve required NGX D3D12 entry points\n");
        return 1;
    }

    ID3D12Device* dev = nullptr;
    HR_CHECK(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)));

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    HR_CHECK(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));

    ID3D12CommandAllocator* alloc = nullptr;
    HR_CHECK(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)));

    ID3D12GraphicsCommandList* cmd = nullptr;
    HR_CHECK(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cmd)));

    ID3D12Fence* fence = nullptr;
    HR_CHECK(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    UINT64 fence_val = 1;
    HANDLE fence_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);

    auto sync_gpu = [&]() {
        cmd->Close();
        ID3D12CommandList* lists[] = { cmd };
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, fence_val);
        fence->SetEventOnCompletion(fence_val++, fence_event);
        WaitForSingleObject(fence_event, INFINITE);
        alloc->Reset();
        cmd->Reset(alloc, nullptr);
    };

    NGX_CHECK(fn_init(0x1337, L"./", dev));

    NVSDK_NGX_Parameter* params = nullptr;
    NGX_CHECK(fn_get_params(&params));

    auto frame_json_raw = read_file(fdir + "/frame.json");
    if (frame_json_raw.empty()) { std::fprintf(stderr, "Missing frame.json\n"); return 1; }
    std::string fj(frame_json_raw.begin(), frame_json_raw.end());
    auto rd = jget(fj, "render"), od = jget(fj, "output");
    int nframes = int(jget(fj, "frames").at(0));
    UINT rw = UINT(rd[0]), rh = UINT(rd[1]);
    UINT ow = UINT(od[0]), oh = UINT(od[1]);

    params->Set("Width", rw);
    params->Set("Height", rh);
    params->Set("OutWidth", ow);
    params->Set("OutHeight", oh);
    params->Set("PerfQualityValue", 1); // Quality

    NVSDK_NGX_Handle* handle = nullptr;
    NGX_CHECK(fn_create(cmd, feat, params, &handle));
    sync_gpu();

    // Allocate resources
    ID3D12Resource* res_col = create_texture(dev, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT);
    ID3D12Resource* res_dep = create_texture(dev, rw, rh, DXGI_FORMAT_R32_FLOAT);
    ID3D12Resource* res_mv  = create_texture(dev, rw, rh, DXGI_FORMAT_R16G16_FLOAT);
    ID3D12Resource* res_out = create_texture(dev, ow, oh, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    ID3D12Resource* res_nrm = (feat == NVSDK_NGX_Feature_RayReconstruction) ? create_texture(dev, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT) : nullptr;
    ID3D12Resource* res_alb = (feat == NVSDK_NGX_Feature_RayReconstruction) ? create_texture(dev, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT) : nullptr;
    ID3D12Resource* res_spa = (feat == NVSDK_NGX_Feature_RayReconstruction) ? create_texture(dev, rw, rh, DXGI_FORMAT_R16G16B16A16_FLOAT) : nullptr;
    ID3D12Resource* res_rgh = (feat == NVSDK_NGX_Feature_RayReconstruction) ? create_texture(dev, rw, rh, DXGI_FORMAT_R32_FLOAT) : nullptr;

    UINT64 out_pitch = (ow * 16 + 255) & ~255;
    ID3D12Resource* up_col = create_buffer(dev, (rw * 8 + 255) & ~255 * rh, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* up_dep = create_buffer(dev, (rw * 4 + 255) & ~255 * rh, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* up_mv  = create_buffer(dev, (rw * 4 + 255) & ~255 * rh, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* rb_out = create_buffer(dev, out_pitch * oh, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);

    CreateDirectoryA(odir.c_str(), nullptr);

    for (int i = 0; i < nframes; ++i) {
        std::string sfx = "_" + std::to_string(i);
        auto col = read_file(fdir + "/col" + sfx + ".f16");
        auto dep = read_file(fdir + "/dep" + sfx + ".f32");
        auto mv  = read_file(fdir + "/mv" + sfx + ".f16");
        auto pj  = read_file(fdir + "/params" + sfx + ".json");
        std::string pstr(pj.begin(), pj.end());

        upload_to_texture(dev, cmd, res_col, up_col, col.data(), rw, rh, 8);
        upload_to_texture(dev, cmd, res_dep, up_dep, dep.data(), rw, rh, 4);
        upload_to_texture(dev, cmd, res_mv, up_mv, mv.data(), rw, rh, 4);

        if (feat == NVSDK_NGX_Feature_RayReconstruction) {
            auto nrm = read_file(fdir + "/nrm" + sfx + ".f16");
            auto alb = read_file(fdir + "/alb" + sfx + ".f16");
            auto spa = read_file(fdir + "/spa" + sfx + ".f16");
            auto rgh = read_file(fdir + "/rgh" + sfx + ".f32");
            upload_to_texture(dev, cmd, res_nrm, up_col, nrm.data(), rw, rh, 8);
            upload_to_texture(dev, cmd, res_alb, up_col, alb.data(), rw, rh, 8);
            upload_to_texture(dev, cmd, res_spa, up_col, spa.data(), rw, rh, 8);
            upload_to_texture(dev, cmd, res_rgh, up_dep, rgh.data(), rw, rh, 4);
        }

        auto jit = jget(pstr, "jitter"), mvs = jget(pstr, "mv_scale");
        int reset = int(jget(pstr, "reset").empty() ? 0 : jget(pstr, "reset")[0]);

        params->Set("Color", res_col);
        params->Set("Output", res_out);
        params->Set("Depth", res_dep);
        params->Set("MotionVectors", res_mv);
        params->Set("Jitter_Offset_X", float(jit[0]));
        params->Set("Jitter_Offset_Y", float(jit[1]));
        params->Set("MV_Scale_X", float(mvs[0]));
        params->Set("MV_Scale_Y", float(mvs[1]));
        params->Set("Reset", reset);

        if (feat == NVSDK_NGX_Feature_RayReconstruction) {
            params->Set("Normals", res_nrm);
            params->Set("DiffuseAlbedo", res_alb);
            params->Set("SpecularAlbedo", res_spa);
            params->Set("Roughness", res_rgh);
        }

        NGX_CHECK(fn_evaluate(cmd, handle, params, nullptr));

        // Readback output texture
        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = res_out;
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = rb_out;
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dstLoc.PlacedFootprint.Offset = 0;
        dstLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        dstLoc.PlacedFootprint.Footprint.Width = ow;
        dstLoc.PlacedFootprint.Footprint.Height = oh;
        dstLoc.PlacedFootprint.Footprint.Depth = 1;
        dstLoc.PlacedFootprint.Footprint.RowPitch = UINT(out_pitch);

        cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        sync_gpu();

        // Save out_i.f32
        void* out_map = nullptr;
        rb_out->Map(0, nullptr, &out_map);
        std::vector<float> f32_data(ow * oh * 4);
        const uint8_t* sptr = static_cast<const uint8_t*>(out_map);
        for (UINT y = 0; y < oh; ++y) {
            std::memcpy(f32_data.data() + y * ow * 4, sptr + y * out_pitch, ow * 16);
        }
        rb_out->Unmap(0, nullptr);
        write_file(odir + "/out" + sfx + ".f32", f32_data.data(), f32_data.size() * sizeof(float));
        std::cout << "[ngx_client] D3D12 frame " << i << " completed." << std::endl;
    }

    if (test_cycles) {
        std::cout << "[ngx_client] Testing create/release/resize cycles..." << std::endl;
        NGX_CHECK(fn_release(handle));
        handle = nullptr;

        // Recreate with new resolution (resize cycle)
        params->Set("Width", rw / 2);
        params->Set("Height", rh / 2);
        params->Set("OutWidth", ow / 2);
        params->Set("OutHeight", oh / 2);
        NGX_CHECK(fn_create(cmd, feat, params, &handle));
        sync_gpu();
        std::cout << "[ngx_client] Recreated with resized resolution successfully." << std::endl;

        NGX_CHECK(fn_release(handle));
        handle = nullptr;
        std::cout << "[ngx_client] Release cycle passed." << std::endl;
    } else {
        NGX_CHECK(fn_release(handle));
    }

    if (fn_shutdown) fn_shutdown();
    std::cout << "[ngx_client] D3D12 test passed." << std::endl;
    return 0;
}

// Main runner for Vulkan
static int run_vulkan(NVSDK_NGX_Feature feat, const std::string& fdir, const std::string& odir, bool test_cycles) {
    std::cout << "[ngx_client] Running Vulkan path..." << std::endl;
    HMODULE vkmod = LoadLibraryA("vulkan-1.dll");
    if (!vkmod) vkmod = LoadLibraryA("winevulkan.dll");
    if (!vkmod) {
        std::fprintf(stderr, "Failed to load vulkan-1.dll / winevulkan.dll\n");
        return 1;
    }

    auto pfn_vkCreateInstance = (PFN_vkCreateInstance)GetProcAddress(vkmod, "vkCreateInstance");
    auto pfn_vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)GetProcAddress(vkmod, "vkEnumeratePhysicalDevices");
    auto pfn_vkCreateDevice = (PFN_vkCreateDevice)GetProcAddress(vkmod, "vkCreateDevice");
    auto pfn_vkCreateCommandPool = (PFN_vkCreateCommandPool)GetProcAddress(vkmod, "vkCreateCommandPool");
    auto pfn_vkAllocateCommandBuffers = (PFN_vkAllocateCommandBuffers)GetProcAddress(vkmod, "vkAllocateCommandBuffers");

    HMODULE ngx = LoadLibraryA("_nvngx.dll");
    if (!ngx) ngx = LoadLibraryA("nvngx.dll");
    if (!ngx) ngx = LoadLibraryA("dlssnr_core.dll");
    if (!ngx) {
        std::fprintf(stderr, "Failed to load NGX library\n");
        return 1;
    }

    auto fn_init = (PFN_NVSDK_NGX_VULKAN_Init)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Init");
    auto fn_shutdown = (PFN_NVSDK_NGX_VULKAN_Shutdown)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Shutdown");
    auto fn_get_params = (PFN_NVSDK_NGX_VULKAN_GetCapabilityParameters)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    auto fn_create1 = (PFN_NVSDK_NGX_VULKAN_CreateFeature1)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_CreateFeature1");
    auto fn_release = (PFN_NVSDK_NGX_VULKAN_ReleaseFeature)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_ReleaseFeature");
    auto fn_evaluate = (PFN_NVSDK_NGX_VULKAN_EvaluateFeature)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_EvaluateFeature");

    if (!fn_init || !fn_get_params || !fn_create1 || !fn_release || !fn_evaluate) {
        std::fprintf(stderr, "Failed to resolve required NGX Vulkan entry points\n");
        return 1;
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "ngx_client";
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo ic{};
    ic.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ic.pApplicationInfo = &appInfo;
    VkInstance instance = VK_NULL_HANDLE;
    if (pfn_vkCreateInstance(&ic, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed\n");
        return 1;
    }

    uint32_t gpu_count = 0;
    pfn_vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr);
    if (gpu_count == 0) {
        std::fprintf(stderr, "No Vulkan physical devices\n");
        return 1;
    }
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    pfn_vkEnumeratePhysicalDevices(instance, &gpu_count, gpus.data());
    VkPhysicalDevice pdev = gpus[0];

    float qp = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qp;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice device = VK_NULL_HANDLE;
    if (pfn_vkCreateDevice(pdev, &dci, nullptr, &device) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateDevice failed\n");
        return 1;
    }

    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    pfn_vkCreateCommandPool(device, &cpci, nullptr, &pool);

    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    pfn_vkAllocateCommandBuffers(device, &cbai, &cmd);

    NGX_CHECK(fn_init(0x1337, L"./", (NrVkInstance)instance, (NrVkPhysicalDevice)pdev, (NrVkDevice)device));

    NVSDK_NGX_Parameter* params = nullptr;
    NGX_CHECK(fn_get_params(&params));

    auto frame_json_raw = read_file(fdir + "/frame.json");
    if (frame_json_raw.empty()) { std::fprintf(stderr, "Missing frame.json\n"); return 1; }
    std::string fj(frame_json_raw.begin(), frame_json_raw.end());
    auto rd = jget(fj, "render"), od = jget(fj, "output");
    int nframes = int(jget(fj, "frames").at(0));
    unsigned int rw = (unsigned int)(rd[0]), rh = (unsigned int)(rd[1]);
    unsigned int ow = (unsigned int)(od[0]), oh = (unsigned int)(od[1]);

    params->Set("Width", rw);
    params->Set("Height", rh);
    params->Set("OutWidth", ow);
    params->Set("OutHeight", oh);
    params->Set("PerfQualityValue", 1);

    NVSDK_NGX_Handle* handle = nullptr;
    NGX_CHECK(fn_create1((NrVkDevice)device, (NrVkCommandBuffer)cmd, feat, params, &handle));

    CreateDirectoryA(odir.c_str(), nullptr);

    // Evaluate frames
    for (int i = 0; i < nframes; ++i) {
        std::string sfx = "_" + std::to_string(i);
        auto pj = read_file(fdir + "/params" + sfx + ".json");
        std::string pstr(pj.begin(), pj.end());
        auto jit = jget(pstr, "jitter"), mvs = jget(pstr, "mv_scale");
        int reset = int(jget(pstr, "reset").empty() ? 0 : jget(pstr, "reset")[0]);

        params->Set("Jitter_Offset_X", float(jit[0]));
        params->Set("Jitter_Offset_Y", float(jit[1]));
        params->Set("MV_Scale_X", float(mvs[0]));
        params->Set("MV_Scale_Y", float(mvs[1]));
        params->Set("Reset", reset);

        NGX_CHECK(fn_evaluate((NrVkCommandBuffer)cmd, handle, params, nullptr));
        std::cout << "[ngx_client] Vulkan frame " << i << " evaluated." << std::endl;
    }

    if (test_cycles) {
        std::cout << "[ngx_client] Testing Vulkan create/release/resize cycles..." << std::endl;
        NGX_CHECK(fn_release(handle));
        handle = nullptr;

        params->Set("Width", rw / 2);
        params->Set("Height", rh / 2);
        params->Set("OutWidth", ow / 2);
        params->Set("OutHeight", oh / 2);
        NGX_CHECK(fn_create1((NrVkDevice)device, (NrVkCommandBuffer)cmd, feat, params, &handle));
        NGX_CHECK(fn_release(handle));
        handle = nullptr;
        std::cout << "[ngx_client] Vulkan release & resize cycles passed." << std::endl;
    } else {
        NGX_CHECK(fn_release(handle));
    }

    if (fn_shutdown) fn_shutdown();
    std::cout << "[ngx_client] Vulkan test passed." << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    std::string api = "d3d12";
    std::string feat_str = "sr";
    std::string frames_dir = "";
    std::string out_dir = "";
    bool test_cycles = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--api" && i + 1 < argc) api = argv[++i];
        else if (a == "--feature" && i + 1 < argc) feat_str = argv[++i];
        else if (a == "--frames" && i + 1 < argc) frames_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--test-cycles") test_cycles = true;
    }

    if (frames_dir.empty() || out_dir.empty()) {
        std::fprintf(stderr, "Usage: ngx_client.exe --api <d3d12|vulkan> --feature <sr|rr> --frames <dir> --out <dir> [--test-cycles]\n");
        return 1;
    }

    NVSDK_NGX_Feature feat = (feat_str == "rr") ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling;

    if (api == "vulkan") {
        return run_vulkan(feat, frames_dir, out_dir, test_cycles);
    } else {
        return run_d3d12(feat, frames_dir, out_dir, test_cycles);
    }
}
