// src/vk/pipeline.h -- the SR/RR frame chain, extracted from the CLI runner so that
// NVSDK_NGX_{VULKAN,D3D12}_EvaluateFeature can record the very same dispatches into a
// caller's command buffer instead of only into vk_runner's own.
//
// The chain owns every buffer, pipeline, descriptor set and piece of history state; the
// caller owns the command buffer, the submit and (for the runner) the frame files.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "vkfuncs.h"

namespace ngxd {

// Small file / JSON helpers shared by the runner and the chain.
std::vector<uint8_t> slurp(const std::string& path);
std::vector<double> jget(const std::string& txt, const std::string& key);
int jint(const std::string& txt, const std::string& key);

// Everything one frame says about the inputs the caller provides.
// Field names and defaults mirror params_i.json / ngx_client's parameter set.
struct FrameParams {
    int fmt_col = 0;   // colour format code  (0 = f16)
    int fmt_mv = 0;    // motion-vector format code
    int fmt_dep = 0;   // depth format code
    int mvhi = 0;      // motion vectors are at output resolution
    int ldr = 0;       // LDR input
    float exposure = 1.0f;
    float pre_exposure = 1.0f;
    int reset = 0;     // drop the temporal history
    float jx = 0.0f, jy = 0.0f;      // this frame's jitter
    float jpx = 0.0f, jpy = 0.0f;    // the previous frame's jitter
};

// A host-visible (and, for the bridge, transfer-capable) buffer.
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t size = 0;
};

struct ChainPipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
};

class FrameChain {
public:
    // mode: "sr" | "rr"; preset: "M" | "L" (sr) | "F" (rr).
    // shader_dir is where <name>.spv is looked up first ("build/shaders" for the runner,
    // the DLL's own shaders/ directory for the bridge); "shaders" stays the fallback.
    // Returns nullptr and fills *err when the weights or the shaders cannot be read.
    static FrameChain* create(VkDevice dev, VkPhysicalDevice phys,
                              const std::string& mode, const std::string& preset,
                              int W, int H, int oW, int oH,
                              const std::string& weights_dir,
                              const std::string& shader_dir,
                              std::string* err);
    ~FrameChain();
    FrameChain(const FrameChain&) = delete;
    FrameChain& operator=(const FrameChain&) = delete;

    // Host-visible staging memory the caller fills (memcpy) before record().
    // Sizes are W*H * 16/4/4/8/4/8/8 bytes respectively; out_color() is oW*oH*16.
    void* in_color()    { return in_col.mapped; }
    void* in_depth()    { return in_dep.mapped; }
    void* in_mv()       { return in_mv_.mapped; }
    void* in_normal()   { return in_norm.mapped; }
    void* in_rough()    { return in_rough_.mapped; }
    void* in_albedo()   { return in_alb.mapped; }
    void* in_specular() { return in_spa.mapped; }
    void* out_color()   { return out.mapped; }

    // Runner-side convenience: copy at most the buffer's capacity into a staging input.
    void fill_color(const void* p, size_t n)    { fill(in_col, p, n); }
    void fill_depth(const void* p, size_t n)    { fill(in_dep, p, n); }
    void fill_mv(const void* p, size_t n)       { fill(in_mv_, p, n); }
    void fill_normal(const void* p, size_t n)   { fill(in_norm, p, n); }
    void fill_rough(const void* p, size_t n)    { fill(in_rough_, p, n); }
    void fill_albedo(const void* p, size_t n)   { fill(in_alb, p, n); }
    void fill_specular(const void* p, size_t n) { fill(in_spa, p, n); }

    // The same slots as VkBuffer, for a caller that moves whole images in and out with
    // vkCmdCopyImageToBuffer / vkCmdCopyBufferToImage instead of memcpy.
    VkBuffer in_color_buf()    const { return in_col.buffer; }
    VkBuffer in_depth_buf()    const { return in_dep.buffer; }
    VkBuffer in_mv_buf()       const { return in_mv_.buffer; }
    VkBuffer in_normal_buf()   const { return in_norm.buffer; }
    VkBuffer in_rough_buf()    const { return in_rough_.buffer; }
    VkBuffer in_albedo_buf()   const { return in_alb.buffer; }
    VkBuffer in_specular_buf() const { return in_spa.buffer; }
    VkBuffer out_color_buf()   const { return out.buffer; }

    // Record one frame into cmd: history reset, the exposure slot, then the whole
    // dispatch chain. The host-side writes happen before record() returns, so the
    // caller may submit cmd afterwards.
    void record(VkCommandBuffer cmd, const FrameParams& fp);

    // Diagnosis: follow the oracle's buffer contents from a dispatch point onward.
    // dir is a tools/verify/diffdump.py dump root (containing frame0/, frame1/ with
    // NNN.<buffer>.bin files); mask is a comma list of dispatch-point names ("post,downsample"),
    // empty = every point.  With dir empty the chain records exactly what it always did.
    void set_injection(const std::string& dir, const std::string& mask);

    // History state advances exactly as it always has: p ^= 1, fc = (fc + 1) & 7.
    void advance() { p ^= 1; fc = (fc + 1) & 7; }
    int parity() const { return p; }
    int frame_counter() const { return fc; }
    size_t pipeline_count() const;

    int width() const { return W; }
    int height() const { return H; }
    int out_width() const { return oW; }
    int out_height() const { return oH; }

private:
    FrameChain() = default;

    static void fill(Buffer& b, const void* p, size_t n) {
        if (p && b.mapped) std::memcpy(b.mapped, p, std::min(b.size, n));
    }

    void dispatch(VkCommandBuffer cmd, const std::string& name,
                  const std::vector<Buffer*>& bufs, const void* push,
                  size_t push_size, uint32_t gx, uint32_t gy, uint32_t gz);

    // Oracle-injection diagnosis (see set_injection).  point names the dispatch just
    // recorded; launch is that point's launch number in the dump of the *current* frame
    // (the dumps' indices shift between frame 0 and frame 1, so callers pass the table's
    // per-frame value).  No-op unless injection is enabled for this frame and point.
    void inject_after(VkCommandBuffer cmd, const char* point, int launch);

    VkDevice dev = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    std::string mode, preset;

    int W = 0, H = 0, oW = 0, oH = 0;
    int W0 = 0, H0 = 0, L1w = 0, L1h = 0, L2w = 0, L2h = 0, L3w = 0, L3h = 0;
    int W15 = 0, H15 = 0, hW15 = 0, hH15 = 0, mvpitch = 0;

    int p = 0, fc = 0;
    float dec0_s0 = 8.0f, dec0_f0 = 0.078633f, dec0_f1 = 0.0f;

    VkDescriptorPool dpool = VK_NULL_HANDLE;
    std::map<std::string, ChainPipeline> pipelines;

    Buffer in_col, in_dep, in_mv_, in_norm, in_rough_, in_alb, in_spa;
    Buffer out_alb, out_spa, out_norm, out_rough_;
    Buffer g_col, g_dep, g_mv, g_mvhi;
    Buffer expo;
    Buffer prev_mvec[2], mv_ero, mv_nondil, hidden[2];
    Buffer gates, factors, aniso;
    Buffer dhist, full15, cf2, cf6, cfa, c0e, bfa, bfe, bf6, bec;
    Buffer ihist[2];
    Buffer t_act[9];
    Buffer ae_next, ae_exp;   // rr_ae_final's bindings 2/3 (writeonly, never read back)
    Buffer out, wbuf, w_enc0, w_dec0;

    // Oracle-injection diagnosis state (inert when set_injection was never called).
    std::string inj_dir;
    std::vector<std::string> inj_mask;
    std::map<std::string, Buffer*> inj_targets;         // dump buffer name -> chain buffer
    std::map<std::string, Buffer> inj_shadow;           // per (frame, point, name) staging
    bool inj_wanted(const char* point) const;
};

} // namespace ngxd
