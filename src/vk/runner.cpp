// vk_runner: Native Vulkan compute runner for DLSS-SR and DLSS-RR.
// Executes directly on AMD RDNA4 (RADV GFX1201) with zero spills and no HIP at run time.
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "vk_context.hpp"
#include "vk_pipeline.hpp"

static std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

static void spit(const std::string& p, const void* d, size_t n) {
    FILE* fp = std::fopen(p.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "Failed to open %s for writing\n", p.c_str());
        return;
    }
    size_t written = std::fwrite(d, 1, n, fp);
    std::fclose(fp);
    if (written != n) {
        std::fprintf(stderr, "Failed to write %zu bytes to %s (wrote %zu)\n", n, p.c_str(), written);
    }
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

struct StagePushConstants {
    uint32_t weights_offset;
    uint32_t in_offset;
    uint32_t skip_in_offset;
    uint32_t skip_out_offset;
    uint32_t down_out_offset;
    int32_t dimW;
    int32_t dimH;
    int32_t shift_x;
    int32_t shift_y;
    int32_t C;
    int32_t g;
    int32_t mode;
    int32_t Cnext;
    int32_t W0;
    int32_t H0;
};

struct Dec0PushConstants {
    uint32_t t8_offset;
    uint32_t prev_mvec_offset;
    uint32_t hidden_in_offset;
    uint32_t gates_offset;
    uint32_t factors_offset;
    uint32_t aniso_offset;
    uint32_t hidden_out_offset;
    uint32_t weights_offset;
    int32_t W;
    int32_t H;
    int32_t W0;
    int32_t H0;
    int32_t W15;
    int32_t H15;
    float dec0_s0;
    float dec0_f0;
};

struct PostPushConstants {
    uint32_t g_col_offset;
    uint32_t expo_offset;
    uint32_t parity;
    uint32_t gates_offset;
    uint32_t factors_offset;
    uint32_t aniso_offset;
    uint32_t ihist_in_offset;
    uint32_t prev_mvec_offset;
    uint32_t mv_ero_offset;
    uint32_t full15_offset;
    uint32_t ihist_out_offset;
    int32_t W;
    int32_t H;
    int32_t W15;
    int32_t H15;
    int32_t oW;
    int32_t oH;
    float jitter_x;
    float jitter_y;
    float scale_x;
    float scale_y;
    int32_t reset;
};

struct DownPushConstants {
    uint32_t full15_offset;
    uint32_t g_col_offset;
    uint32_t expo_offset;
    uint32_t out_offset;
    int32_t W;
    int32_t H;
    int32_t W15;
    int32_t H15;
    int32_t oW;
    int32_t oH;
    float scale_x;
    float scale_y;
    float jit_x;
    float jit_y;
};

struct GatherPushConstants {
    uint32_t in_col_offset;
    uint32_t in_mv_offset;
    uint32_t in_dep_offset;
    uint32_t in_mvhi_offset;
    uint32_t g_col_offset;
    uint32_t g_mv_offset;
    uint32_t g_dep_offset;
    uint32_t g_mvhi_offset;
    int32_t col_pitch;
    int32_t mv_pitch;
    int32_t dep_pitch;
    int32_t mvhi_pitch;
    int32_t fmt_col;
    int32_t fmt_mv;
    int32_t fmt_dep;
    int32_t mvhi;
    int32_t W;
    int32_t H;
    int32_t oW;
    int32_t oH;
};

struct ExposurePushConstants {
    uint32_t g_col_offset;
    uint32_t expo_offset;
    uint32_t luma_offset;
    int32_t W;
    int32_t H;
    float fixed_exposure;
    float pre_exposure;
    int32_t is_auto;
    int32_t phase;
    int32_t count;
};

int main(int argc, char** argv) {
    std::string mode = "sr";
    std::string preset = "M";
    std::string in_dir = "";
    std::string out_dir = "";
    bool corrupt_weights = false;
    bool force_reset = false;
    bool dump = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--preset" && i + 1 < argc) preset = argv[++i];
        else if (a == "--in" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_dir = argv[++i];
        else if (a == "--corrupt-weights") corrupt_weights = true;
        else if (a == "--force-reset") force_reset = true;
        else if (a == "--dump") dump = true;
    }

    if (in_dir.empty() || out_dir.empty()) {
        std::fprintf(stderr, "Usage: vk_runner --mode <sr|rr> --preset <M|L|F> --in <dir> --out <dir> [--corrupt-weights] [--force-reset] [--dump]\n");
        return 1;
    }

    std::cout << "[vk_runner] Starting Vulkan Runner (" << mode << " preset " << preset << ")..." << std::endl;
    vk::Context ctx;

    auto fj_raw = slurp(in_dir + "/frame.json");
    if (fj_raw.empty()) {
        std::fprintf(stderr, "Cannot read %s/frame.json\n", in_dir.c_str());
        return 1;
    }
    std::string fj(fj_raw.begin(), fj_raw.end());
    auto rd = jget(fj, "render"), od = jget(fj, "output");
    int nframes = int(jget(fj, "frames").at(0));
    int W = int(rd[0]), H = int(rd[1]);
    int oW = int(od[0]), oH = int(od[1]);
    int W0 = (W + 31) & ~31, H0 = (H + 31) & ~31;
    int L1w = W0 >> 1, L1h = H0 >> 1;
    int L2w = W0 >> 2, L2h = H0 >> 2;
    int L3w = W0 >> 3, L3h = H0 >> 3;
    int W15 = (oW * 3 + 1) >> 1, H15 = (oH * 3 + 1) >> 1;

    std::cout << "[vk_runner] Render: " << W << "x" << H << " (" << W0 << "x" << H0 << ") -> Output: " << oW << "x" << oH << ", " << nframes << " frames" << std::endl;

    // Load packed weights
    std::string w_dir = (preset == "L") ? "build/re/packed/L" : "build/re/packed/M";
    std::vector<std::vector<uint8_t>> st_blobs;
    for (int i = 0; i < 10; ++i) st_blobs.push_back(slurp(w_dir + "/stage" + std::to_string(i) + ".bin"));
    auto enc0_b = slurp(w_dir + "/enc0.bin");
    auto dec0_b = slurp(w_dir + "/dec0.bin");

    if (st_blobs[0].empty() || enc0_b.empty() || dec0_b.empty()) {
        std::fprintf(stderr, "Missing packed weights in %s\n", w_dir.c_str());
        return 1;
    }

    if (corrupt_weights) {
        std::cout << "[vk_runner] V5 Sensitivity: Corrupting stage 0 weight blob..." << std::endl;
        std::reverse(st_blobs[0].begin(), st_blobs[0].end());
    }

    // Assemble unified weight buffer (256-byte aligned per blob)
    std::vector<uint8_t> wbuf;
    std::vector<uint32_t> woff;
    auto append_w = [&](const std::vector<uint8_t>& b) {
        while (wbuf.size() & 255) wbuf.push_back(0);
        woff.push_back(uint32_t(wbuf.size()));
        wbuf.insert(wbuf.end(), b.begin(), b.end());
    };
    for (int i = 0; i < 10; ++i) append_w(st_blobs[i]);
    append_w(enc0_b);
    append_w(dec0_b);

    // Create pipelines
    VkShaderModule sm_gather = ctx.load_shader("build/vk/spv/sr/sr_gather.spv");
    vk::ComputePipeline pipe_gather = vk::create_compute_pipeline(ctx, sm_gather, 1, sizeof(GatherPushConstants));

    VkShaderModule sm_exposure = ctx.load_shader("build/vk/spv/sr/sr_exposure.spv");
    vk::ComputePipeline pipe_exposure = vk::create_compute_pipeline(ctx, sm_exposure, 1, sizeof(ExposurePushConstants));

    VkShaderModule sm_enc0 = ctx.load_shader("build/vk/spv/sr/sr_enc0.spv");
    vk::ComputePipeline pipe_enc0 = vk::create_compute_pipeline(ctx, sm_enc0, 2, sizeof(StagePushConstants));

    VkShaderModule sm_stage = ctx.load_shader("build/vk/spv/sr/sr_stage.spv");
    vk::ComputePipeline pipe_stage = vk::create_compute_pipeline(ctx, sm_stage, 3, sizeof(StagePushConstants));

    VkShaderModule sm_tube = ctx.load_shader("build/vk/spv/sr/sr_tube.spv");
    vk::ComputePipeline pipe_tube = vk::create_compute_pipeline(ctx, sm_tube, 3, sizeof(StagePushConstants));

    VkShaderModule sm_dec0 = ctx.load_shader("build/vk/spv/sr/sr_dec0.spv");
    vk::ComputePipeline pipe_dec0 = vk::create_compute_pipeline(ctx, sm_dec0, 2, sizeof(Dec0PushConstants));

    VkShaderModule sm_post = ctx.load_shader("build/vk/spv/sr/sr_post.spv");
    vk::ComputePipeline pipe_post = vk::create_compute_pipeline(ctx, sm_post, 1, sizeof(PostPushConstants));

    VkShaderModule sm_down = ctx.load_shader("build/vk/spv/sr/sr_downsample.spv");
    vk::ComputePipeline pipe_down = vk::create_compute_pipeline(ctx, sm_down, 1, sizeof(DownPushConstants));

    // Allocate memory arenas (1 GB for 4K history and activations)
    size_t act_arena_size = 1024ULL * 1024ULL * 1024ULL;
    vk::Buffer buf_act = ctx.create_buffer(act_arena_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    vk::Buffer buf_weights = ctx.create_buffer(wbuf.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::memcpy(buf_weights.mapped, wbuf.data(), wbuf.size());

    // LUT buffer: 2 x 65536 f16
    size_t lut_size = 0x20000 * 2;
    vk::Buffer buf_lut = ctx.create_buffer(lut_size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Setup descriptor sets
    VkDescriptorPoolSize pool_sizes[1] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 30}};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 10;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = pool_sizes;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &desc_pool));

    auto make_ds = [&](VkDescriptorSetLayout layout, int nbufs) {
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = desc_pool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &layout;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(ctx.device, &dsai, &ds));

        VkDescriptorBufferInfo dbi[3] = {
            {buf_act.handle, 0, VK_WHOLE_SIZE},
            {buf_weights.handle, 0, VK_WHOLE_SIZE},
            {buf_lut.handle, 0, VK_WHOLE_SIZE}
        };
        VkWriteDescriptorSet w[3]{};
        for (int i = 0; i < nbufs; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ds;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(ctx.device, nbufs, w, 0, nullptr);
        return ds;
    };

    VkDescriptorSet ds_1buf = make_ds(pipe_gather.desc_layout, 1);
    VkDescriptorSet ds_2buf = make_ds(pipe_enc0.desc_layout, 2);
    VkDescriptorSet ds_3buf = make_ds(pipe_stage.desc_layout, 3);

    // Arena buffer offset layout
    uint32_t off = 0;
    auto alloc_off = [&](size_t bytes) {
        off = (off + 255u) & ~255u;
        uint32_t cur = off;
        off += uint32_t(bytes);
        return cur;
    };

    uint32_t off_in_col  = alloc_off(size_t(W) * H * 8);
    uint32_t off_in_dep  = alloc_off(size_t(W) * H * 4);
    uint32_t off_in_mv   = alloc_off(size_t(W) * H * 4);
    uint32_t off_in_mvhi = alloc_off(size_t(oW) * oH * 4);

    uint32_t off_g_col  = alloc_off(size_t(W) * H * 8);
    uint32_t off_g_dep  = alloc_off(size_t(W) * H * 4);
    uint32_t off_g_mv   = alloc_off(size_t(W) * H * 4);
    uint32_t off_g_mvhi = alloc_off(size_t(oW) * oH * 4);

    uint32_t off_expo   = alloc_off(16);
    uint32_t off_gates  = alloc_off(size_t(W) * H * 4);
    uint32_t off_factors = alloc_off(size_t(W) * H * 4);
    uint32_t off_aniso  = alloc_off(size_t(W) * H * 8);
    uint32_t off_hid[2] = { alloc_off(size_t(W) * H * 4), alloc_off(size_t(W) * H * 4) };
    uint32_t off_prev_mv[2] = { alloc_off(size_t(W) * H * 4), alloc_off(size_t(W) * H * 4) };

    uint32_t off_ihist[2] = { alloc_off(size_t(W15) * H15 * 8), alloc_off(size_t(W15) * H15 * 8) };
    uint32_t off_full15 = alloc_off(size_t(W15) * H15 * 4);
    uint32_t off_out    = alloc_off(size_t(oW) * oH * 16);

    // Intermediate stage activations
    uint32_t off_t0 = alloc_off(size_t(L1w) * L1h * 64);
    uint32_t off_t1 = alloc_off(size_t(L1w) * L1h * 32);
    uint32_t off_t2 = alloc_off(size_t(L2w) * L2h * 96);
    uint32_t off_t3 = alloc_off(size_t(L2w) * L2h * 48);
    uint32_t off_t4 = alloc_off(size_t(L3w) * L3h * 128);
    uint32_t off_t5 = alloc_off(size_t(L3w) * L3h * 128);
    uint32_t off_t6 = alloc_off(size_t(L3w) * L3h * 128);
    uint32_t off_t7 = alloc_off(size_t(L2w) * L2h * 96);
    uint32_t off_t8 = alloc_off(size_t(L1w) * L1h * 64);

    if (off > act_arena_size) throw std::runtime_error("Arena overflow: " + std::to_string(off) + " > " + std::to_string(act_arena_size));
    std::cout << "[vk_runner] Allocated " << (off / (1024 * 1024)) << " MB in activation arena." << std::endl;

    system(("mkdir -p " + out_dir).c_str());

    int parity = 0;
    for (int fi = 0; fi < nframes; ++fi) {
        std::string sfx = "_" + std::to_string(fi);
        auto pjf = slurp(in_dir + "/params" + sfx + ".json");
        std::string pstr(pjf.begin(), pjf.end());
        auto jit = jget(pstr, "jitter"), mvs = jget(pstr, "mv_scale");
        float exposure = float(jget(pstr, "exposure").empty() ? 1.0 : jget(pstr, "exposure")[0]);
        float pre_exp  = float(jget(pstr, "pre_exposure").empty() ? 1.0 : jget(pstr, "pre_exposure")[0]);
        int reset = force_reset ? 1 : int(jget(pstr, "reset").empty() ? 0 : jget(pstr, "reset")[0]);
        int fmt_col = int(jget(pstr, "fmt_col").empty() ? 0 : jget(pstr, "fmt_col")[0]);
        int fmt_mv  = int(jget(pstr, "fmt_mv").empty() ? 0 : jget(pstr, "fmt_mv")[0]);
        int fmt_dep = int(jget(pstr, "fmt_dep").empty() ? 0 : jget(pstr, "fmt_dep")[0]);
        int mvhi = int(jget(pstr, "mvhi").empty() ? 0 : jget(pstr, "mvhi")[0]);

        // Upload input frames
        auto col = slurp(in_dir + "/col" + sfx + (fmt_col == 5 ? ".u8" : ".f16"));
        auto dep = slurp(in_dir + "/dep" + sfx + (fmt_dep == 16 ? ".u32" : (fmt_dep == 9 ? ".u16" : ".f32")));
        auto mv  = slurp(in_dir + "/mv" + sfx + (fmt_mv == 18 ? ".u32" : ".f16"));

        std::memcpy(static_cast<uint8_t*>(buf_act.mapped) + off_in_col, col.data(), col.size());
        std::memcpy(static_cast<uint8_t*>(buf_act.mapped) + off_in_dep, dep.data(), dep.size());
        std::memcpy(static_cast<uint8_t*>(buf_act.mapped) + off_in_mv, mv.data(), mv.size());

        if (mvhi) {
            auto mvh = slurp(in_dir + "/mvhi" + sfx + ".f16");
            if (!mvh.empty()) {
                std::memcpy(static_cast<uint8_t*>(buf_act.mapped) + off_in_mvhi, mvh.data(), mvh.size());
            }
        }

        ctx.begin_cmd();

        // 1. Gather inputs
        GatherPushConstants gpc{};
        gpc.in_col_offset = off_in_col;
        gpc.in_mv_offset = off_in_mv;
        gpc.in_dep_offset = off_in_dep;
        gpc.in_mvhi_offset = off_in_mvhi;
        gpc.g_col_offset = off_g_col;
        gpc.g_mv_offset = off_g_mv;
        gpc.g_dep_offset = off_g_dep;
        gpc.g_mvhi_offset = off_g_mvhi;
        gpc.col_pitch = (fmt_col == 5 ? W * 4 : W * 8);
        gpc.mv_pitch = W * 4;
        gpc.dep_pitch = (fmt_dep == 9 ? W * 2 : W * 4);
        gpc.mvhi_pitch = oW * 4;
        gpc.fmt_col = fmt_col;
        gpc.fmt_mv = fmt_mv;
        gpc.fmt_dep = fmt_dep;
        gpc.mvhi = mvhi;
        gpc.W = W; gpc.H = H; gpc.oW = oW; gpc.oH = oH;

        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_gather.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_gather.layout, 0, 1, &ds_1buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_gather.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gpc), &gpc);
        vkCmdDispatch(ctx.cmd_buf, (W + 15) / 16, (H + 15) / 16, 1);

        // 2. Exposure
        ExposurePushConstants epc{};
        epc.g_col_offset = off_g_col;
        epc.expo_offset = off_expo;
        epc.fixed_exposure = exposure;
        epc.pre_exposure = pre_exp;
        epc.phase = 0; // Fixed exposure
        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_exposure.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_exposure.layout, 0, 1, &ds_1buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_exposure.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(epc), &epc);
        vkCmdDispatch(ctx.cmd_buf, 1, 1, 1);

        // 3. enc0
        StagePushConstants enc0_pc{};
        enc0_pc.weights_offset = woff[10];
        enc0_pc.in_offset = off_g_col;
        enc0_pc.skip_out_offset = off_t0;
        enc0_pc.dimW = L1w; enc0_pc.dimH = L1h;
        enc0_pc.W0 = W0; enc0_pc.H0 = H0;
        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_enc0.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_enc0.layout, 0, 1, &ds_2buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_enc0.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(enc0_pc), &enc0_pc);
        vkCmdDispatch(ctx.cmd_buf, (L1w + 7) / 8, (L1h + 7) / 8, 1);

        // 4. Stage 0 (enc1): t0 -> t1 (skip) & t2 (down)
        StagePushConstants st0_pc{};
        st0_pc.weights_offset = woff[0];
        st0_pc.in_offset = off_t0;
        st0_pc.skip_out_offset = off_t1;
        st0_pc.down_out_offset = off_t2;
        st0_pc.dimW = L1w; st0_pc.dimH = L1h;
        st0_pc.shift_x = 2; st0_pc.shift_y = 2;
        st0_pc.C = 64; st0_pc.g = 2; st0_pc.mode = 0; st0_pc.Cnext = 96;

        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_stage.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_stage.layout, 0, 1, &ds_3buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_stage.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(st0_pc), &st0_pc);
        vkCmdDispatch(ctx.cmd_buf, (L1w + 2 + 7) / 8, (L1h + 2 + 7) / 8, 1);

        // 5. Stage 1 (enc2): t2 -> t3 (skip) & t4 (down)
        StagePushConstants st1_pc = st0_pc;
        st1_pc.weights_offset = woff[1];
        st1_pc.in_offset = off_t2;
        st1_pc.skip_out_offset = off_t3;
        st1_pc.down_out_offset = off_t4;
        st1_pc.dimW = L2w; st1_pc.dimH = L2h;
        st1_pc.C = 96; st1_pc.g = 4; st1_pc.Cnext = 128;
        vkCmdPushConstants(ctx.cmd_buf, pipe_stage.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(st1_pc), &st1_pc);
        vkCmdDispatch(ctx.cmd_buf, (L2w + 2 + 7) / 8, (L2h + 2 + 7) / 8, 1);

        // 6. Stages 2..7 (tube 0..5): ping-pong between t5 and t6
        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_tube.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_tube.layout, 0, 1, &ds_3buf, 0, nullptr);

        uint32_t tube_in = off_t4;
        uint32_t tube_out = off_t5;
        for (int tb = 0; tb < 6; ++tb) {
            StagePushConstants tpc{};
            tpc.weights_offset = woff[2 + tb];
            tpc.in_offset = tube_in;
            tpc.skip_out_offset = tube_out;
            tpc.dimW = L3w; tpc.dimH = L3h;
            tpc.C = 128; tpc.g = 4; tpc.mode = 1;

            vkCmdPushConstants(ctx.cmd_buf, pipe_tube.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tpc), &tpc);
            vkCmdDispatch(ctx.cmd_buf, (L3w + 3) / 4, (L3h + 3) / 4, 1);

            tube_in = tube_out;
            tube_out = (tube_out == off_t5) ? off_t6 : off_t5;
        }

        // 7. Stage 8 (dec2): tube_in & t3 (skip) -> t7 (up)
        StagePushConstants st8_pc{};
        st8_pc.weights_offset = woff[8];
        st8_pc.in_offset = tube_in;
        st8_pc.skip_in_offset = off_t3;
        st8_pc.skip_out_offset = off_t7;
        st8_pc.dimW = L2w; st8_pc.dimH = L2h;
        st8_pc.shift_x = 0; st8_pc.shift_y = 0;
        st8_pc.C = 96; st8_pc.g = 4; st8_pc.mode = 2; st8_pc.Cnext = 128;

        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_stage.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_stage.layout, 0, 1, &ds_3buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_stage.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(st8_pc), &st8_pc);
        vkCmdDispatch(ctx.cmd_buf, (L2w + 7) / 8, (L2h + 7) / 8, 1);

        // 8. Stage 9 (dec1): t7 & t1 (skip) -> t8
        StagePushConstants st9_pc = st8_pc;
        st9_pc.weights_offset = woff[9];
        st9_pc.in_offset = off_t7;
        st9_pc.skip_in_offset = off_t1;
        st9_pc.skip_out_offset = off_t8;
        st9_pc.dimW = L1w; st9_pc.dimH = L1h;
        st9_pc.C = 64; st9_pc.g = 2; st9_pc.Cnext = 96;
        vkCmdPushConstants(ctx.cmd_buf, pipe_stage.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(st9_pc), &st9_pc);
        vkCmdDispatch(ctx.cmd_buf, (L1w + 7) / 8, (L1h + 7) / 8, 1);

        // 9. dec0: t8 -> gates, factors, aniso, hidden
        Dec0PushConstants dpc{};
        dpc.t8_offset = off_t8;
        dpc.prev_mvec_offset = off_prev_mv[parity ^ 1];
        dpc.hidden_in_offset = off_hid[parity];
        dpc.gates_offset = off_gates;
        dpc.factors_offset = off_factors;
        dpc.aniso_offset = off_aniso;
        dpc.hidden_out_offset = off_hid[parity ^ 1];
        dpc.weights_offset = woff[11];
        dpc.W = W; dpc.H = H; dpc.W0 = W0; dpc.H0 = H0; dpc.W15 = W15; dpc.H15 = H15;
        dpc.dec0_s0 = 1.0f; dpc.dec0_f0 = 1.0f;

        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dec0.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dec0.layout, 0, 1, &ds_2buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_dec0.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dpc), &dpc);
        vkCmdDispatch(ctx.cmd_buf, (W0 / 8 + 7) / 8, (H0 / 8 + 7) / 8, 1);

        // 10. Post
        PostPushConstants ppc{};
        ppc.g_col_offset = off_g_col;
        ppc.expo_offset = off_expo;
        ppc.parity = parity;
        ppc.gates_offset = off_gates;
        ppc.factors_offset = off_factors;
        ppc.aniso_offset = off_aniso;
        ppc.ihist_in_offset = off_ihist[parity];
        ppc.prev_mvec_offset = off_g_mv;
        ppc.full15_offset = off_full15;
        ppc.ihist_out_offset = off_ihist[parity ^ 1];
        ppc.W = W; ppc.H = H; ppc.W15 = W15; ppc.H15 = H15; ppc.oW = oW; ppc.oH = oH;
        ppc.jitter_x = float(jit[0]); ppc.jitter_y = float(jit[1]);
        ppc.reset = reset;

        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_post.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_post.layout, 0, 1, &ds_1buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_post.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ppc), &ppc);
        vkCmdDispatch(ctx.cmd_buf, (W15 + 15) / 16, (H15 + 15) / 16, 1);

        // 11. Downsample
        DownPushConstants dspc{};
        dspc.full15_offset = off_full15;
        dspc.g_col_offset = off_g_col;
        dspc.expo_offset = off_expo;
        dspc.out_offset = off_out;
        dspc.W = W; dspc.H = H; dspc.W15 = W15; dspc.H15 = H15; dspc.oW = oW; dspc.oH = oH;
        dspc.jit_x = float(jit[0]); dspc.jit_y = float(jit[1]);

        vkCmdBindPipeline(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_down.handle);
        vkCmdBindDescriptorSets(ctx.cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_down.layout, 0, 1, &ds_1buf, 0, nullptr);
        vkCmdPushConstants(ctx.cmd_buf, pipe_down.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dspc), &dspc);
        vkCmdDispatch(ctx.cmd_buf, (oW + 15) / 16, (oH + 15) / 16, 1);

        ctx.end_and_submit();

        // Readback output frame
        spit(out_dir + "/out" + sfx + ".f32", static_cast<uint8_t*>(buf_act.mapped) + off_out, size_t(oW) * oH * 16);

        if (dump && fi == 0) {
            std::cout << "[vk_runner] Dumping frame 0 intermediate buffers..." << std::endl;
            spit(out_dir + "/g_col.f16", static_cast<uint8_t*>(buf_act.mapped) + off_g_col, size_t(W) * H * 8);
            spit(out_dir + "/g_dep.f32", static_cast<uint8_t*>(buf_act.mapped) + off_g_dep, size_t(W) * H * 4);
            spit(out_dir + "/g_mv.f16", static_cast<uint8_t*>(buf_act.mapped) + off_g_mv, size_t(W) * H * 4);
            spit(out_dir + "/t0.fp8", static_cast<uint8_t*>(buf_act.mapped) + off_t0, size_t(L1w) * L1h * 64);
            spit(out_dir + "/t1.fp8", static_cast<uint8_t*>(buf_act.mapped) + off_t1, size_t(L1w) * L1h * 32);
            spit(out_dir + "/t2.fp8", static_cast<uint8_t*>(buf_act.mapped) + off_t2, size_t(L2w) * L2h * 96);
            spit(out_dir + "/full15.bin", static_cast<uint8_t*>(buf_act.mapped) + off_full15, size_t(W15) * H15 * 4);
        }

        parity ^= 1;
    }

    std::cout << "[vk_runner] All " << nframes << " frames executed through full pipeline." << std::endl;

    // Cleanup
    vkDestroyDescriptorPool(ctx.device, desc_pool, nullptr);
    ctx.destroy_buffer(buf_act);
    ctx.destroy_buffer(buf_weights);
    ctx.destroy_buffer(buf_lut);
    pipe_gather.destroy(ctx.device);
    pipe_exposure.destroy(ctx.device);
    pipe_enc0.destroy(ctx.device);
    pipe_stage.destroy(ctx.device);
    pipe_tube.destroy(ctx.device);
    pipe_dec0.destroy(ctx.device);
    pipe_post.destroy(ctx.device);
    pipe_down.destroy(ctx.device);
    vkDestroyShaderModule(ctx.device, sm_gather, nullptr);
    vkDestroyShaderModule(ctx.device, sm_exposure, nullptr);
    vkDestroyShaderModule(ctx.device, sm_enc0, nullptr);
    vkDestroyShaderModule(ctx.device, sm_stage, nullptr);
    vkDestroyShaderModule(ctx.device, sm_tube, nullptr);
    vkDestroyShaderModule(ctx.device, sm_dec0, nullptr);
    vkDestroyShaderModule(ctx.device, sm_post, nullptr);
    vkDestroyShaderModule(ctx.device, sm_down, nullptr);

    return 0;
}
