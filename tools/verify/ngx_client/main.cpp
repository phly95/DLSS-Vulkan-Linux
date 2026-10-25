// ngx_client: a Windows program (mingw, run under Proton) that drives DLSS-SR (feature 1) or DLSS-RR
// (feature 13) through NGX's public entry points exactly as a game does, on a frames directory of the
// referee's (tools/verify/README.md), and writes what the feature produced (run_all.sh V9).
//
//   ngx_client.exe --api d3d12|vulkan --feature sr|rr --frames DIR --out DIR [--preset M|L]
//                  [--ngx-dll NAME] [--test-cycles] [--expect-create-fail]
//
// Every frame's inputs are uploaded into real textures (D3D12 resources or Vulkan images), the feature
// is evaluated into an RGBA32F output and the output is copied back and written as out_<i>.f32.
// --test-cycles also releases the feature, creates it again at half size, evaluates one frame there and
// releases it. --expect-create-fail inverts the result: success only if CreateFeature refuses (the
// package must refuse when the game has no, or the wrong, NVIDIA DLL). Exit 0 = success.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "ngx_abi.h"

#define FAIL(...) do { std::fprintf(stderr, "[ngx_client] FAIL: " __VA_ARGS__); std::fprintf(stderr, "\n"); return 1; } while (0)
#define HR(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) FAIL("%s:%d %s = 0x%08lx", __FILE__, __LINE__, #x, (unsigned long)hr_); } while (0)
#define VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) FAIL("%s:%d %s = %d", __FILE__, __LINE__, #x, int(r_)); } while (0)
#define NGX(x) do { NVSDK_NGX_Result r_ = (x); if (!ngx_ok(r_)) FAIL("%s:%d %s = 0x%08x", __FILE__, __LINE__, #x, unsigned(r_)); } while (0)

static std::vector<uint8_t> read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}
static bool write_file(const std::string& p, const void* d, size_t n) {
    std::ofstream f(p, std::ios::binary);
    f.write(static_cast<const char*>(d), std::streamsize(n));
    return bool(f);
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
static double jnum(const std::string& txt, const std::string& key, double dflt) {
    auto v = jget(txt, key);
    return v.empty() ? dflt : v[0];
}

// ---- the frames directory (default formats only: the format cells are run_all's V3, not V9's) ----
struct Input { const char* file; const char* key; DXGI_FORMAT dxgi; VkFormat vk; unsigned bpp; int code0; };
static const Input kSrInputs[] = {
    {"col", "Color", DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, 0},
    {"mv", "MotionVectors", DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, 4, 0},
    {"dep", "Depth", DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, 4, 0},
};
static const Input kRrInputs[] = {
    {"col", "Color", DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, 0},
    {"mv", "MotionVectors", DXGI_FORMAT_R16G16_FLOAT, VK_FORMAT_R16G16_SFLOAT, 4, 2},
    {"dep", "Depth", DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, 4, 3},
    {"nrm", "GBuffer.Normals", DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, 0},
    {"alb", "DLSS.Input.DiffuseAlbedo", DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, 0},
    {"spa", "DLSS.Input.SpecularAlbedo", DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, 8, 0},
    {"rgh", "GBuffer.Roughness", DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, 4, 3},
};

struct Run {
    bool rr = false, cycles = false, expect_fail = false;
    char preset = 'M';
    std::string fdir, odir, dll;
    unsigned rw = 0, rh = 0, ow = 0, oh = 0;
    int frames = 0;
    std::vector<Input> inputs;
    std::vector<std::string> params;      // params_<i>.json texts

    int load() {
        auto fj = read_file(fdir + "/frame.json");
        if (fj.empty()) FAIL("no %s/frame.json", fdir.c_str());
        std::string s(fj.begin(), fj.end());
        auto r = jget(s, "render"), o = jget(s, "output"), n = jget(s, "frames");
        if (r.size() != 2 || o.size() != 2 || n.empty()) FAIL("frame.json lacks render / output / frames");
        rw = unsigned(r[0]); rh = unsigned(r[1]); ow = unsigned(o[0]); oh = unsigned(o[1]); frames = int(n[0]);
        inputs = rr ? std::vector<Input>(std::begin(kRrInputs), std::end(kRrInputs))
                    : std::vector<Input>(std::begin(kSrInputs), std::end(kSrInputs));
        for (int i = 0; i < frames; ++i) {
            auto p = read_file(fdir + "/params_" + std::to_string(i) + ".json");
            if (p.empty()) FAIL("no params_%d.json", i);
            params.emplace_back(p.begin(), p.end());
            for (auto& in : inputs)
                if (jnum(params.back(), std::string("fmt_") + in.file, in.code0) != in.code0)
                    FAIL("params_%d.json sets fmt_%s: ngx_client feeds default formats only", i, in.file);
            for (const char* k : {"mvhi", "ldr"})
                if (jnum(params.back(), k, 0) != 0) FAIL("params_%d.json sets %s: not an ngx_client input", i, k);
        }
        return 0;
    }
    // the bytes of input `in` for frame i, exactly rw * rh texels
    bool texels(int i, const Input& in, std::vector<uint8_t>& out) const {
        const char* ext = in.bpp == 4 && std::strcmp(in.file, "mv") ? ".f32" : ".f16";
        const std::string p = fdir + "/" + in.file + "_" + std::to_string(i) + ext;
        out = read_file(p);
        if (out.size() != size_t(rw) * rh * in.bpp) {
            std::fprintf(stderr, "[ngx_client] FAIL: %s: %zu bytes, expected %zu\n", p.c_str(), out.size(), size_t(rw) * rh * in.bpp);
            return false;
        }
        return true;
    }
    int quality() const {
        const double r = double(ow) / rw;
        struct { double r; int q; } t[] = {{1.0, NGX_PQ_DLAA}, {1.3, NGX_PQ_ULTRA_QUALITY}, {1.5, NGX_PQ_MAX_QUALITY},
                                           {1.724, NGX_PQ_BALANCED}, {2.0, NGX_PQ_MAX_PERF}, {3.0, NGX_PQ_ULTRA_PERF}};
        int best = 0;
        for (int k = 1; k < 6; ++k) if (std::fabs(t[k].r - r) < std::fabs(t[best].r - r)) best = k;
        return t[best].q;
    }
    void create_params(NVSDK_NGX_Parameter* p, unsigned w, unsigned h, unsigned w2, unsigned h2) const {
        const std::string& p0 = params.at(0);
        p->Set("Width", w); p->Set("Height", h); p->Set("OutWidth", w2); p->Set("OutHeight", h2);
        p->Set("PerfQualityValue", quality());
        p->Set("CreationNodeMask", 1u); p->Set("VisibilityNodeMask", 1u);
        int flags = NGX_FLAG_IS_HDR | NGX_FLAG_MV_LOW_RES;
        if (jnum(p0, "depth_inverted", 0) != 0) flags |= NGX_FLAG_DEPTH_INVERTED;
        if (jnum(p0, "auto_exposure", 0) != 0) flags |= NGX_FLAG_AUTO_EXPOSURE;
        p->Set("DLSS.Feature.Create.Flags", flags);
        if (rr) {
            p->Set("DLSS.Denoise.Mode", 1);                 // DL unified (ray reconstruction)
            p->Set("DLSS.Roughness.Mode", 0);               // roughness in its own texture
            p->Set("DLSS.Use.HW.Depth", int(jnum(p0, "hw_depth", 1) != 0));
        } else {
            const unsigned pr = preset == 'L' ? NGX_PRESET_L : NGX_PRESET_M;
            for (const char* m : {"DLAA", "Quality", "Balanced", "Performance", "UltraPerformance", "UltraQuality"})
                p->Set((std::string("DLSS.Hint.Render.Preset.") + m).c_str(), pr);
        }
    }
    void frame_params(NVSDK_NGX_Parameter* p, int i, unsigned w, unsigned h) const {
        const std::string& s = params.at(i);
        auto jit = jget(s, "jitter"), mvs = jget(s, "mv_scale");
        p->Set("Jitter.Offset.X", float(jit.at(0))); p->Set("Jitter.Offset.Y", float(jit.at(1)));
        p->Set("MV.Scale.X", float(mvs.at(0))); p->Set("MV.Scale.Y", float(mvs.at(1)));
        p->Set("Reset", int(jnum(s, "reset", 0) != 0));
        p->Set("DLSS.Pre.Exposure", float(jnum(s, "pre_exposure", 1.0)));
        p->Set("DLSS.Exposure.Scale", float(jnum(s, "exposure", 1.0)));
        p->Set("DLSS.Render.Subrect.Dimensions.Width", w);
        p->Set("DLSS.Render.Subrect.Dimensions.Height", h);
    }
};

static HMODULE load_ngx(const Run& r) {
    if (!r.dll.empty()) return LoadLibraryA(r.dll.c_str());
    HMODULE m = LoadLibraryA("_nvngx.dll");
    if (!m) m = LoadLibraryA("nvngx.dll");
    return m;
}

// ------------------------------------------------------------------------------------------------
// D3D12
// ------------------------------------------------------------------------------------------------
static ID3D12Resource* d3d_texture(ID3D12Device* dev, unsigned w, unsigned h, DXGI_FORMAT f, bool uav, D3D12_RESOURCE_STATES st) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = w; d.Height = h; d.DepthOrArraySize = 1; d.MipLevels = 1; d.Format = f; d.SampleDesc.Count = 1;
    d.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    D3D12_HEAP_PROPERTIES hp{}; hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    ID3D12Resource* r = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, st, nullptr, IID_PPV_ARGS(&r)))) return nullptr;
    return r;
}
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

static int run_d3d12(Run& r) {
    std::printf("[ngx_client] D3D12, feature %s, %ux%u -> %ux%u, %d frames\n", r.rr ? "RR" : "SR", r.rw, r.rh, r.ow, r.oh, r.frames);
    HMODULE ngx = load_ngx(r);
    if (!ngx) FAIL("cannot load the NGX DLL (%s)", r.dll.empty() ? "_nvngx.dll / nvngx.dll" : r.dll.c_str());
    auto init = (PFN_NGX_D3D12_Init)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Init");
    auto shutdown1 = (PFN_NGX_D3D12_Shutdown1)GetProcAddress(ngx, "NVSDK_NGX_D3D12_Shutdown1");
    auto alloc_params = (PFN_NGX_AllocateParameters)GetProcAddress(ngx, "NVSDK_NGX_D3D12_AllocateParameters");
    auto destroy_params = (PFN_NGX_DestroyParameters)GetProcAddress(ngx, "NVSDK_NGX_D3D12_DestroyParameters");
    auto create = (PFN_NGX_D3D12_CreateFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_CreateFeature");
    auto release = (PFN_NGX_ReleaseFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_ReleaseFeature");
    auto evaluate = (PFN_NGX_D3D12_EvaluateFeature)GetProcAddress(ngx, "NVSDK_NGX_D3D12_EvaluateFeature");
    if (!init || !alloc_params || !create || !release || !evaluate) FAIL("the NGX DLL lacks a D3D12 entry point");

    ID3D12Device* dev = nullptr;
    HR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev)));
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* q = nullptr; HR(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&q)));
    ID3D12CommandAllocator* ca = nullptr; HR(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&ca)));
    ID3D12GraphicsCommandList* cl = nullptr;
    HR(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, ca, nullptr, IID_PPV_ARGS(&cl)));
    ID3D12Fence* fence = nullptr; HR(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    UINT64 fv = 0;
    HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    auto submit = [&]() -> bool {
        if (FAILED(cl->Close())) return false;
        ID3D12CommandList* l[] = {cl};
        q->ExecuteCommandLists(1, l);
        q->Signal(fence, ++fv);
        if (fence->GetCompletedValue() < fv) { fence->SetEventOnCompletion(fv, ev); WaitForSingleObject(ev, INFINITE); }
        if (dev->GetDeviceRemovedReason() != S_OK) return false;
        return SUCCEEDED(ca->Reset()) && SUCCEEDED(cl->Reset(ca, nullptr));
    };

    NGX(init(0x1337, L".", dev, nullptr, NGX_VERSION_API));
    NVSDK_NGX_Parameter* p = nullptr;
    NGX(alloc_params(&p));
    const NVSDK_NGX_Feature feat = r.rr ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling;

    auto one_pass = [&](unsigned rw, unsigned rh, unsigned ow, unsigned oh, int nframes, bool write) -> int {
        r.create_params(p, rw, rh, ow, oh);
        NVSDK_NGX_Handle* h = nullptr;
        const NVSDK_NGX_Result cr = create(cl, feat, p, &h);
        if (!submit()) FAIL("device lost after CreateFeature");
        if (r.expect_fail) {
            if (ngx_ok(cr) && h) FAIL("CreateFeature succeeded, but the package should have refused");
            std::printf("[ngx_client] CreateFeature refused (0x%08x), as expected\n", unsigned(cr));
            return 0;
        }
        if (!ngx_ok(cr) || !h) FAIL("CreateFeature = 0x%08x", unsigned(cr));
        std::vector<ID3D12Resource*> tex, up;
        for (auto& in : r.inputs) {
            tex.push_back(d3d_texture(dev, rw, rh, in.dxgi, false, D3D12_RESOURCE_STATE_COPY_DEST));
            up.push_back(d3d_buffer(dev, UINT64(pitch256(rw * in.bpp)) * rh, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ));
            if (!tex.back() || !up.back()) FAIL("cannot create the %s texture", in.file);
        }
        ID3D12Resource* out = d3d_texture(dev, ow, oh, DXGI_FORMAT_R32G32B32A32_FLOAT, true, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        const UINT opitch = pitch256(ow * 16);
        ID3D12Resource* rb = d3d_buffer(dev, UINT64(opitch) * oh, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        if (!out || !rb) FAIL("cannot create the output");
        std::vector<float> host(size_t(ow) * oh * 4);
        for (int i = 0; i < nframes; ++i) {
            for (size_t k = 0; k < r.inputs.size(); ++k) {
                const Input& in = r.inputs[k];
                std::vector<uint8_t> t;
                if (!r.texels(i, in, t)) return 1;
                void* m = nullptr;
                HR(up[k]->Map(0, nullptr, &m));
                const UINT pt = pitch256(rw * in.bpp);
                // the half-size cycle uploads the top-left part of the full-size frame
                for (unsigned y = 0; y < rh; ++y)
                    std::memcpy(static_cast<uint8_t*>(m) + size_t(y) * pt, t.data() + size_t(y) * r.rw * in.bpp, rw * in.bpp);
                up[k]->Unmap(0, nullptr);
                if (i > 0) d3d_barrier(cl, tex[k], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
                D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
                src.pResource = up[k]; src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                src.PlacedFootprint.Footprint = {in.dxgi, rw, rh, 1, pt};
                dst.pResource = tex[k]; dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                d3d_barrier(cl, tex[k], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                p->Set(in.key, tex[k]);
            }
            p->Set("Output", out);
            r.frame_params(p, i, rw, rh);
            NGX(evaluate(cl, h, p, nullptr));
            d3d_barrier(cl, out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
            src.pResource = out; src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dst.pResource = rb; dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, ow, oh, 1, opitch};
            cl->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            d3d_barrier(cl, out, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!submit()) FAIL("device lost in frame %d", i);
            void* m = nullptr;
            D3D12_RANGE range{0, size_t(opitch) * oh};
            HR(rb->Map(0, &range, &m));
            for (unsigned y = 0; y < oh; ++y) std::memcpy(host.data() + size_t(y) * ow * 4, static_cast<uint8_t*>(m) + size_t(y) * opitch, ow * 16);
            rb->Unmap(0, nullptr);
            if (write && !write_file(r.odir + "/out_" + std::to_string(i) + ".f32", host.data(), host.size() * 4))
                FAIL("cannot write %s/out_%d.f32", r.odir.c_str(), i);
            std::printf("[ngx_client] D3D12 frame %d read back\n", i);
        }
        NGX(release(h));
        for (auto* t : tex) t->Release();
        for (auto* u : up) u->Release();
        out->Release(); rb->Release();
        return 0;
    };
    if (one_pass(r.rw, r.rh, r.ow, r.oh, r.frames, true)) return 1;
    if (r.cycles && !r.expect_fail) {
        if (one_pass(r.rw / 2, r.rh / 2, r.ow / 2, r.oh / 2, 1, false)) return 1;
        std::printf("[ngx_client] release, half-size create, evaluate, release: ok\n");
    }
    if (destroy_params) destroy_params(p);
    if (shutdown1) shutdown1(dev);
    std::printf("[ngx_client] D3D12 done\n");
    return 0;
}

// ------------------------------------------------------------------------------------------------
// Vulkan (through Proton's winevulkan)
// ------------------------------------------------------------------------------------------------
#define VK_FNS                                                                                              \
    VKF(vkEnumeratePhysicalDevices) VKF(vkGetPhysicalDeviceProperties) VKF(vkGetPhysicalDeviceQueueFamilyProperties) \
    VKF(vkGetPhysicalDeviceMemoryProperties) VKF(vkGetPhysicalDeviceFeatures2) VKF(vkEnumerateDeviceExtensionProperties) \
    VKF(vkCreateDevice) VKF(vkGetDeviceQueue) VKF(vkCreateCommandPool) VKF(vkAllocateCommandBuffers)        \
    VKF(vkBeginCommandBuffer) VKF(vkEndCommandBuffer) VKF(vkQueueSubmit) VKF(vkQueueWaitIdle) VKF(vkCreateImage) \
    VKF(vkCreateImageView) VKF(vkGetImageMemoryRequirements) VKF(vkAllocateMemory) VKF(vkBindImageMemory)   \
    VKF(vkCreateBuffer) VKF(vkGetBufferMemoryRequirements) VKF(vkBindBufferMemory) VKF(vkMapMemory)         \
    VKF(vkUnmapMemory) VKF(vkCmdPipelineBarrier) VKF(vkCmdCopyBufferToImage) VKF(vkCmdCopyImageToBuffer)    \
    VKF(vkResetCommandBuffer) VKF(vkDestroyImage) VKF(vkDestroyImageView) VKF(vkDestroyBuffer) VKF(vkFreeMemory) \
    VKF(vkDeviceWaitIdle)
struct VkFns {
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    PFN_vkGetDeviceProcAddr gdpa = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
#define VKF(n) PFN_##n n = nullptr;
    VK_FNS
#undef VKF
};

static int run_vulkan(Run& r) {
    std::printf("[ngx_client] Vulkan, feature %s, %ux%u -> %ux%u, %d frames\n", r.rr ? "RR" : "SR", r.rw, r.rh, r.ow, r.oh, r.frames);
    HMODULE vkm = LoadLibraryA("vulkan-1.dll");
    if (!vkm) FAIL("cannot load vulkan-1.dll");
    VkFns f;
    f.gipa = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(vkm, "vkGetInstanceProcAddr");
    if (!f.gipa) FAIL("vulkan-1.dll has no vkGetInstanceProcAddr");
    HMODULE ngx = load_ngx(r);
    if (!ngx) FAIL("cannot load the NGX DLL (%s)", r.dll.empty() ? "_nvngx.dll / nvngx.dll" : r.dll.c_str());
    auto req_ext = (PFN_NGX_VULKAN_RequiredExtensions)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_RequiredExtensions");
    auto init = (PFN_NGX_VULKAN_Init)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Init");
    auto shutdown1 = (PFN_NGX_VULKAN_Shutdown1)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Shutdown1");
    auto alloc_params = (PFN_NGX_AllocateParameters)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_AllocateParameters");
    auto destroy_params = (PFN_NGX_DestroyParameters)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_DestroyParameters");
    auto create = (PFN_NGX_VULKAN_CreateFeature1)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_CreateFeature1");
    auto release = (PFN_NGX_ReleaseFeature)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_ReleaseFeature");
    auto evaluate = (PFN_NGX_VULKAN_EvaluateFeature)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_EvaluateFeature");
    if (!init || !alloc_params || !create || !release || !evaluate) FAIL("the NGX DLL lacks a Vulkan entry point");

    unsigned ni = 0, nd = 0;
    const char** ie = nullptr;
    const char** de = nullptr;
    if (req_ext && !ngx_ok(req_ext(&ni, &ie, &nd, &de))) ni = nd = 0;
    f.vkCreateInstance = (PFN_vkCreateInstance)f.gipa(nullptr, "vkCreateInstance");
    if (!f.vkCreateInstance) FAIL("no vkCreateInstance");
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ngx_client";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app; ici.enabledExtensionCount = ni; ici.ppEnabledExtensionNames = ie;
    VkInstance inst = VK_NULL_HANDLE;
    VK(f.vkCreateInstance(&ici, nullptr, &inst));
#define VKF(n) f.n = (PFN_##n)f.gipa(inst, #n); if (!f.n) FAIL("no %s", #n);
    VK_FNS
#undef VKF
    f.gdpa = (PFN_vkGetDeviceProcAddr)f.gipa(inst, "vkGetDeviceProcAddr");
    uint32_t n = 0;
    VK(f.vkEnumeratePhysicalDevices(inst, &n, nullptr));
    if (!n) FAIL("no Vulkan device");
    std::vector<VkPhysicalDevice> pds(n);
    VK(f.vkEnumeratePhysicalDevices(inst, &n, pds.data()));
    VkPhysicalDevice pd = pds[0];
    for (auto d : pds) {
        VkPhysicalDeviceProperties pr;
        f.vkGetPhysicalDeviceProperties(d, &pr);
        if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { pd = d; break; }
    }
    VkPhysicalDeviceProperties props;
    f.vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("[ngx_client] Vulkan device: %s\n", props.deviceName);
    uint32_t nq = 0;
    f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    f.vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; ++i)
        if ((qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) { qfi = i; break; }
    if (qfi == UINT32_MAX) FAIL("no graphics + compute queue");

    // Every feature the driver offers, and the extensions NGX asks for plus the two a compute port of
    // these networks needs when present: the device a game would make for DLSS on this card.
    uint32_t nx = 0;
    f.vkEnumerateDeviceExtensionProperties(pd, nullptr, &nx, nullptr);
    std::vector<VkExtensionProperties> xs(nx);
    f.vkEnumerateDeviceExtensionProperties(pd, nullptr, &nx, xs.data());
    auto has = [&](const char* e) { for (auto& x : xs) if (!std::strcmp(x.extensionName, e)) return true; return false; };
    std::vector<const char*> dex;
    for (unsigned i = 0; i < nd; ++i) if (has(de[i])) dex.push_back(de[i]);
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    void** tail = &v13.pNext;
    auto add_ext = [&](const char* e) { for (auto* x : dex) if (!std::strcmp(x, e)) return; dex.push_back(e); };
    if (has(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) { add_ext(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME); *tail = &cm; tail = &cm.pNext; }
    if (has(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) { add_ext(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME); *tail = &f8; tail = &f8.pNext; }
    f.vkGetPhysicalDeviceFeatures2(pd, &feats);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &feats};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(dex.size()); dci.ppEnabledExtensionNames = dex.data();
    VkDevice dev = VK_NULL_HANDLE;
    VK(f.vkCreateDevice(pd, &dci, nullptr, &dev));
    VkQueue q;
    f.vkGetDeviceQueue(dev, qfi, 0, &q);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = qfi;
    VkCommandPool pool;
    VK(f.vkCreateCommandPool(dev, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = pool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK(f.vkAllocateCommandBuffers(dev, &cbi, &cb));
    VkPhysicalDeviceMemoryProperties mp;
    f.vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    auto mem_type = [&](uint32_t bits, VkMemoryPropertyFlags want) -> uint32_t {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
        return UINT32_MAX;
    };
    auto begin = [&]() {
        VkCommandBufferBeginInfo b{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        return f.vkBeginCommandBuffer(cb, &b);
    };
    auto submit = [&]() -> VkResult {
        VkResult e = f.vkEndCommandBuffer(cb);
        if (e != VK_SUCCESS) return e;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        if ((e = f.vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)) != VK_SUCCESS) return e;
        if ((e = f.vkQueueWaitIdle(q)) != VK_SUCCESS) return e;
        return f.vkResetCommandBuffer(cb, 0);
    };
    auto barrier = [&](VkImage img, VkImageLayout a, VkImageLayout b) {
        VkImageMemoryBarrier m{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        m.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        m.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        m.oldLayout = a; m.newLayout = b;
        m.srcQueueFamilyIndex = m.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        m.image = img; m.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        f.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &m);
    };
    struct Img { VkImage img = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; NGX_Resource_VK res{}; };
    struct Buf { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; };
    auto make_img = [&](unsigned w, unsigned h, VkFormat fmt, bool rw, Img& o) -> bool {
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D; ii.format = fmt; ii.extent = {w, h, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   (rw ? VK_IMAGE_USAGE_STORAGE_BIT : 0);
        if (f.vkCreateImage(dev, &ii, nullptr, &o.img) != VK_SUCCESS) return false;
        VkMemoryRequirements mr;
        f.vkGetImageMemoryRequirements(dev, o.img, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size; ai.memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX || f.vkAllocateMemory(dev, &ai, nullptr, &o.mem) != VK_SUCCESS) return false;
        if (f.vkBindImageMemory(dev, o.img, o.mem, 0) != VK_SUCCESS) return false;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = o.img; vi.viewType = VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (f.vkCreateImageView(dev, &vi, nullptr, &o.view) != VK_SUCCESS) return false;
        o.res.Type = NGX_RESOURCE_VK_IMAGEVIEW;
        o.res.ReadWrite = rw;
        o.res.Resource.ImageViewInfo = {o.view, o.img, vi.subresourceRange, fmt, w, h};
        return true;
    };
    auto make_buf = [&](VkDeviceSize size, VkBufferUsageFlags u, Buf& o) -> bool {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size; bi.usage = u;
        if (f.vkCreateBuffer(dev, &bi, nullptr, &o.buf) != VK_SUCCESS) return false;
        VkMemoryRequirements mr;
        f.vkGetBufferMemoryRequirements(dev, o.buf, &mr);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size;
        ai.memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (ai.memoryTypeIndex == UINT32_MAX || f.vkAllocateMemory(dev, &ai, nullptr, &o.mem) != VK_SUCCESS) return false;
        return f.vkBindBufferMemory(dev, o.buf, o.mem, 0) == VK_SUCCESS;
    };

    NGX(init(0x1337, L".", inst, pd, dev, f.gipa, f.gdpa, nullptr, NGX_VERSION_API));
    NVSDK_NGX_Parameter* p = nullptr;
    NGX(alloc_params(&p));
    const NVSDK_NGX_Feature feat = r.rr ? NVSDK_NGX_Feature_RayReconstruction : NVSDK_NGX_Feature_SuperSampling;

    auto one_pass = [&](unsigned rw, unsigned rh, unsigned ow, unsigned oh, int nframes, bool write) -> int {
        r.create_params(p, rw, rh, ow, oh);
        NVSDK_NGX_Handle* h = nullptr;
        VK(begin());
        const NVSDK_NGX_Result cr = create(dev, cb, feat, p, &h);
        VK(submit());
        if (r.expect_fail) {
            if (ngx_ok(cr) && h) FAIL("CreateFeature succeeded, but the package should have refused");
            std::printf("[ngx_client] CreateFeature refused (0x%08x), as expected\n", unsigned(cr));
            return 0;
        }
        if (!ngx_ok(cr) || !h) FAIL("CreateFeature = 0x%08x", unsigned(cr));
        std::vector<Img> tex(r.inputs.size());
        std::vector<Buf> up(r.inputs.size());
        Img out;
        Buf rb;
        for (size_t k = 0; k < r.inputs.size(); ++k)
            if (!make_img(rw, rh, r.inputs[k].vk, false, tex[k]) ||
                !make_buf(VkDeviceSize(rw) * rh * r.inputs[k].bpp, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, up[k]))
                FAIL("cannot create the %s image", r.inputs[k].file);
        if (!make_img(ow, oh, VK_FORMAT_R32G32B32A32_SFLOAT, true, out) ||
            !make_buf(VkDeviceSize(ow) * oh * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, rb))
            FAIL("cannot create the output image");
        VK(begin());
        for (auto& t : tex) barrier(t.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        barrier(out.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VK(submit());
        std::vector<float> host(size_t(ow) * oh * 4);
        for (int i = 0; i < nframes; ++i) {
            VK(begin());
            for (size_t k = 0; k < r.inputs.size(); ++k) {
                const Input& in = r.inputs[k];
                std::vector<uint8_t> t;
                if (!r.texels(i, in, t)) return 1;
                void* m = nullptr;
                VK(f.vkMapMemory(dev, up[k].mem, 0, VK_WHOLE_SIZE, 0, &m));
                for (unsigned y = 0; y < rh; ++y)
                    std::memcpy(static_cast<uint8_t*>(m) + size_t(y) * rw * in.bpp, t.data() + size_t(y) * r.rw * in.bpp, rw * in.bpp);
                f.vkUnmapMemory(dev, up[k].mem);
                barrier(tex[k].img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkBufferImageCopy c{};
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                c.imageExtent = {rw, rh, 1};
                f.vkCmdCopyBufferToImage(cb, up[k].buf, tex[k].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
                barrier(tex[k].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
                p->Set(in.key, static_cast<void*>(&tex[k].res));
            }
            p->Set("Output", static_cast<void*>(&out.res));
            r.frame_params(p, i, rw, rh);
            NGX(evaluate(cb, h, p, nullptr));
            barrier(out.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy c{};
            c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent = {ow, oh, 1};
            f.vkCmdCopyImageToBuffer(cb, out.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb.buf, 1, &c);
            barrier(out.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
            VK(submit());
            void* m = nullptr;
            VK(f.vkMapMemory(dev, rb.mem, 0, VK_WHOLE_SIZE, 0, &m));
            std::memcpy(host.data(), m, host.size() * 4);
            f.vkUnmapMemory(dev, rb.mem);
            if (write && !write_file(r.odir + "/out_" + std::to_string(i) + ".f32", host.data(), host.size() * 4))
                FAIL("cannot write %s/out_%d.f32", r.odir.c_str(), i);
            std::printf("[ngx_client] Vulkan frame %d read back\n", i);
        }
        VK(f.vkDeviceWaitIdle(dev));
        NGX(release(h));
        auto free_img = [&](Img& o) { f.vkDestroyImageView(dev, o.view, nullptr); f.vkDestroyImage(dev, o.img, nullptr); f.vkFreeMemory(dev, o.mem, nullptr); };
        auto free_buf = [&](Buf& o) { f.vkDestroyBuffer(dev, o.buf, nullptr); f.vkFreeMemory(dev, o.mem, nullptr); };
        for (auto& t : tex) free_img(t);
        for (auto& u : up) free_buf(u);
        free_img(out);
        free_buf(rb);
        return 0;
    };
    if (one_pass(r.rw, r.rh, r.ow, r.oh, r.frames, true)) return 1;
    if (r.cycles && !r.expect_fail) {
        if (one_pass(r.rw / 2, r.rh / 2, r.ow / 2, r.oh / 2, 1, false)) return 1;
        std::printf("[ngx_client] release, half-size create, evaluate, release: ok\n");
    }
    if (destroy_params) destroy_params(p);
    if (shutdown1) shutdown1(dev);
    std::printf("[ngx_client] Vulkan done\n");
    return 0;
}

int main(int argc, char** argv) {
    Run r;
    std::string api;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--api") api = next();
        else if (a == "--feature") {
            const std::string v = next();
            if (v != "sr" && v != "rr") FAIL("--feature sr|rr");
            r.rr = v == "rr";
        } else if (a == "--frames") r.fdir = next();
        else if (a == "--out") r.odir = next();
        else if (a == "--preset") {
            const std::string v = next();
            if (v != "M" && v != "L") FAIL("--preset M|L");
            r.preset = v[0];
        } else if (a == "--ngx-dll") r.dll = next();
        else if (a == "--test-cycles") r.cycles = true;
        else if (a == "--expect-create-fail") r.expect_fail = true;
        else FAIL("unknown argument %s", a.c_str());
    }
    if ((api != "d3d12" && api != "vulkan") || r.fdir.empty() || (r.odir.empty() && !r.expect_fail))
        FAIL("usage: ngx_client.exe --api d3d12|vulkan --feature sr|rr --frames DIR --out DIR [--preset M|L] "
             "[--ngx-dll NAME] [--test-cycles] [--expect-create-fail]");
    if (r.load()) return 1;
    if (!r.odir.empty()) CreateDirectoryA(r.odir.c_str(), nullptr);
    return api == "vulkan" ? run_vulkan(r) : run_d3d12(r);
}
