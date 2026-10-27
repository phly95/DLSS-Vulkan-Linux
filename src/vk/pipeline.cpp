// src/vk/pipeline.cpp -- the SR/RR frame chain.  Everything below used to live inside
// runner.cpp's main(); it is unchanged apart from being a member of FrameChain and from
// taking the command buffer as an argument, so that the NGX bridge can record it too.

#include "pipeline.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace ngxd {

std::vector<uint8_t> slurp(const std::string& p) {
    // stdio, not <fstream>: a Windows DLL that pulls in libstdc++'s stream/locale machinery carries a
    // dense run of .refptr relocations, which tools/verify/r2scan.py's 64-byte windows then find in
    // NeuralHub's program (pure linker metadata, but the rule is the rule).  Reads the whole file, as
    // the ifstream version did: same bytes for a file, empty for a missing one or a directory.
    std::FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0) { std::fclose(f); return {}; }
    std::vector<uint8_t> v;
    v.resize(size_t(n));
    size_t got = std::fread(v.data(), 1, size_t(n), f);
    std::fclose(f);
    v.resize(got);
    return v;
}

// RR U-Net item table from build/rr/packed/params.json: for each of the 16 units, the 21 meta offsets
// (-1 = absent) and s1d2 / s1d4 / s1d6 as u16 bits, 24 uint32 per unit (rr_level binding 13).
static bool rr_table_from_params(const std::string& pj, uint32_t* tab) {
    size_t cur = 0;
    for (int k = 0; k < 16; ++k) {
        size_t m = pj.find("\"meta\"", cur);
        if (m == std::string::npos) return false;
        size_t i = pj.find('[', m);
        if (i == std::string::npos) return false;
        ++i;
        for (int j = 0; j < 21; ++j) {
            while (i < pj.size() && !(pj[i] == '-' || (pj[i] >= '0' && pj[i] <= '9'))) ++i;
            char* e = nullptr;
            long v = std::strtol(pj.c_str() + i, &e, 10);
            if (e == pj.c_str() + i) return false;
            tab[k * 24 + j] = uint32_t(int32_t(v));
            i = size_t(e - pj.c_str());
        }
        cur = i;
        static const char* const keys[3] = {"\"s1d2\"", "\"s1d4\"", "\"s1d6\""};
        for (int s = 0; s < 3; ++s) {
            size_t q = pj.find(keys[s], cur);
            if (q == std::string::npos) return false;
            size_t c = pj.find(':', q);
            if (c == std::string::npos) return false;
            ++c;
            while (c < pj.size() && (pj[c] == ' ' || pj[c] == '\n' || pj[c] == '\t' || pj[c] == '\r')) ++c;
            char* e = nullptr;
            long v = std::strtol(pj.c_str() + c, &e, 10);
            if (e == pj.c_str() + c) return false;
            tab[k * 24 + 21 + s] = uint32_t(v) & 0xFFFFu;
            cur = size_t(e - pj.c_str());
        }
    }
    return true;
}

// Flat JSON parser
std::vector<double> jget(const std::string& txt, const std::string& key) {
    std::vector<double> v;
    size_t k = txt.find("\"" + key + "\"");
    if (k == std::string::npos) return v;
    size_t i = txt.find(':', k) + 1;
    while (i < txt.size() && (txt[i] == ' ' || txt[i] == '[')) ++i;
    for (;;) {
        char* e = nullptr;
        double d = std::strtod(txt.c_str() + i, &e);
        if (e == txt.c_str() + i) {
            if (txt.compare(i, 4, "true") == 0) { v.push_back(1); i += 4; }
            else if (txt.compare(i, 5, "false") == 0) { v.push_back(0); i += 5; }
            else break;
        } else { v.push_back(d); i = size_t(e - txt.c_str()); }
        while (i < txt.size() && (txt[i] == ' ' || txt[i] == ',')) ++i;
        if (i >= txt.size() || txt[i] == ']' || txt[i] == '}' || txt[i] == '\n') break;
    }
    return v;
}

int jint(const std::string& txt, const std::string& key) {
    auto v = jget(txt, key);
    return v.empty() ? 0 : int(v[0]);
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    // Fallback to any matching type
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
        if (type_filter & (1 << i)) return i;
    }
    return 0;
}

static Buffer create_buffer(VkDevice dev, VkPhysicalDevice phys, size_t size, VkBufferUsageFlags usage) {
    Buffer b;
    b.size = (size + 255) & ~size_t(255);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = b.size;
    bci.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(dev, &bci, nullptr, &b.buffer);

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b.buffer, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vkAllocateMemory(dev, &mai, nullptr, &b.memory);
    vkBindBufferMemory(dev, b.buffer, b.memory, 0);
    vkMapMemory(dev, b.memory, 0, b.size, 0, &b.mapped);
    std::memset(b.mapped, 0, b.size);
    return b;
}

static void destroy_buffer(VkDevice dev, Buffer& b) {
    if (b.mapped) vkUnmapMemory(dev, b.memory);
    if (b.buffer) vkDestroyBuffer(dev, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(dev, b.memory, nullptr);
    b = {};
}

static std::string g_pipeline_error;

static ChainPipeline create_compute_pipeline(VkDevice dev, const std::vector<uint32_t>& spv, uint32_t num_bindings, uint32_t push_size) {
    ChainPipeline p;
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = uint32_t(bindings.size());
    dslci.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &p.dsl);

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = (push_size + 15) & ~15u;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p.dsl;
    if (push_size > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }
    vkCreatePipelineLayout(dev, &plci, nullptr, &p.layout);

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4;
    smci.pCode = spv.data();
    VkShaderModule mod;
    vkCreateShaderModule(dev, &smci, nullptr, &mod);

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = p.layout;
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline);
    vkDestroyShaderModule(dev, mod, nullptr);
    if (r != VK_SUCCESS || p.pipeline == VK_NULL_HANDLE) {
        // The usual reason is a device that was not created with the features these shaders need
        // (16-bit and 8-bit storage, shaderFloat16); say so instead of dispatching a null handle.
        g_pipeline_error = std::string("vkCreateComputePipelines failed (VkResult ") + std::to_string(int(r)) +
                           ") - this VkDevice lacks the features the shaders are built for";
        p.pipeline = VK_NULL_HANDLE;
    }
    return p;
}

static uint16_t f32_to_f16_rne(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint16_t s = uint16_t((u >> 16) & 0x8000);
    uint32_t a = u & 0x7fffffff;
    if (a > 0x7f800000) return s | 0x7e00;
    if (a >= 0x477ff000) return s | 0x7c00;
    if (a < 0x38800000) {
        float af; std::memcpy(&af, &a, 4);
        return s | uint16_t(std::nearbyint(double(af) * 16777216.0));
    }
    return s | uint16_t(((a & 0x3fff) == 0x3000 || (a & 0x1fff) > 0x1000) + (a >> 13) + 0x4000);
}

FrameChain* FrameChain::create(VkDevice dev, VkPhysicalDevice phys,
                               const std::string& mode, const std::string& preset,
                               int W, int H, int oW, int oH,
                               const std::string& wdir,
                               const std::string& shader_dir,
                               std::string* err) {
    FrameChain* c = new FrameChain();
    c->dev = dev;
    c->phys = phys;
    c->mode = mode;
    c->preset = preset;
    c->W = W; c->H = H; c->oW = oW; c->oH = oH;

    // Dimensions
    c->W0 = (W + 31) & ~31;
    c->H0 = (H + 31) & ~31;
    c->L1w = c->W0 >> 1; c->L1h = c->H0 >> 1;
    c->L2w = c->W0 >> 2; c->L2h = c->H0 >> 2;
    c->L3w = c->W0 >> 3; c->L3h = c->H0 >> 3;
    c->W15 = (oW * 3 + 1) >> 1; c->H15 = (oH * 3 + 1) >> 1;
    c->hW15 = (c->W15 + 1) / 2; c->hH15 = (c->H15 + 1) / 2;
    c->mvpitch = (W + 63) & ~63;

    int W0 = c->W0, H0 = c->H0, L1w = c->L1w, L1h = c->L1h, L2w = c->L2w, L2h = c->L2h;
    int L3w = c->L3w, L3h = c->L3h, W15 = c->W15, H15 = c->H15;
    int hW15 = c->hW15, hH15 = c->hH15, mvpitch = c->mvpitch;

    // Load Shaders and Create Pipelines
    // Locate shaders directory: build/shaders or shaders
    std::string sdir = shader_dir;
    auto load_spv = [&](const std::string& name) -> std::vector<uint32_t> {
        auto raw = slurp(sdir + "/" + name + ".spv");
        if (raw.empty()) raw = slurp("shaders/" + name + ".spv");
        if (raw.empty()) return {};
        std::vector<uint32_t> spv(raw.size() / 4);
        std::memcpy(spv.data(), raw.data(), raw.size());
        return spv;
    };

    auto add_pipe = [&](const std::string& name, uint32_t bindings, uint32_t push_size) {
        auto spv = load_spv(name);
        if (spv.empty()) {
            std::fprintf(stderr, "Missing shader %s.spv\n", name.c_str());
            return;
        }
        c->pipelines[name] = create_compute_pipeline(dev, spv, bindings, push_size);
    };

    if (mode == "sr") {
        add_pipe("sr_gather", 6, 88);
        add_pipe("sr_gather_mv", 2, 32);
        add_pipe("sr_exposure", 1, 32);
        add_pipe("sr_ae_luma", 2, 48);
        add_pipe("sr_ae_reduce", 2, 32);
        add_pipe("sr_ae_copy", 2, 24);
        add_pipe("sr_enc0", 11, 240);
        add_pipe("sr_stage", 5, 56);
        add_pipe("sr_tube_split", 3, 56);
        add_pipe("sr_dec0", 8, 160);
        add_pipe("sr_post", 12, 256);
        add_pipe("sr_downsample", 4, 88);
        add_pipe("sr_unorm8_table", 1, 0);
        if (preset == "L") {
            add_pipe("sr_enc0L_in", 11, 148);
            add_pipe("sr_l0", 3, 0);
        }
    } else {
        add_pipe("rr_gather", 14, 8);
        add_pipe("rr_ae_block", 2, 16);
        // rr_ae_final.comp declares bindings 0..3 (Accumulator, PreviousAE, NextAE, Exposure); with
        // only 2 in the set layout, bindings 2/3 had no descriptor and the shader addressed garbage
        // - undefined behaviour that is harmless on a native vkCreateDevice and a GPUVM fault on the
        // device the NGX host hands over.  Push stays 8 bytes, as it always was.
        add_pipe("rr_ae_final", 4, 8);
        add_pipe("rr_enc0_feat", 13, 16);
        add_pipe("rr_level", 14, 68);
        add_pipe("rr_post_prep", 4, 40);
        add_pipe("rr_post", 6, 120);
        add_pipe("rr_downsample", 4, 48);
        add_pipe("rr_pooled2", 2, 32);
        add_pipe("rr_avg2x2", 2, 8);
        add_pipe("rr_ds_t", 2, 8);
        add_pipe("rr_ds_fused", 2, 8);
    }

    if (!g_pipeline_error.empty()) {
        if (err) *err = g_pipeline_error;
        g_pipeline_error.clear();
        delete c;
        return nullptr;
    }

    if (c->pipelines.empty()) {
        if (err) *err = "no pipeline could be loaded from " + sdir;
        delete c;
        return nullptr;
    }

    // Descriptor pool
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 256;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &c->dpool);

    // Allocate Buffers
    c->in_col = create_buffer(dev, phys, size_t(W) * H * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->in_dep = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->in_mv_ = create_buffer(dev, phys, size_t(std::max(W, oW)) * std::max(H, oH) * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->in_norm = create_buffer(dev, phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->in_rough_ = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->in_alb = create_buffer(dev, phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->in_spa = create_buffer(dev, phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->out_alb = create_buffer(dev, phys, size_t(W) * H * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->out_spa = create_buffer(dev, phys, size_t(W) * H * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->out_norm = create_buffer(dev, phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->out_rough_ = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    c->g_col = create_buffer(dev, phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->g_dep = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->g_mv  = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->g_mvhi = create_buffer(dev, phys, size_t(oW) * oH * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    c->expo = create_buffer(dev, phys, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (mode == "sr") {
        uint16_t exp_init[8] = { 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0, 0, 0, 0 };
        std::memcpy(c->expo.mapped, exp_init, 16);
    } else {
        float exp_init[4] = { 1.0f, 1.0f, 1.0f, 0.0f };
        std::memcpy(c->expo.mapped, exp_init, 16);
    }

    c->prev_mvec[0] = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->prev_mvec[1] = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->mv_ero = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->mv_nondil = create_buffer(dev, phys, size_t(H) * mvpitch * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->hidden[0] = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->hidden[1] = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->gates = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->factors = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->aniso = create_buffer(dev, phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (mode == "rr") {
        uint8_t init_gates[4] = { 58, 223, 207, 0 };
        uint8_t* g_ptr = static_cast<uint8_t*>(c->gates.mapped);
        if (g_ptr) {
            for (size_t i = 0; i < size_t(W) * H; ++i) std::memcpy(g_ptr + i * 4, init_gates, 4);
        }
        uint8_t init_factors[4] = { 41, 193, 19, 2 };
        uint8_t* f_ptr = static_cast<uint8_t*>(c->factors.mapped);
        if (f_ptr) {
            for (size_t i = 0; i < size_t(W) * H; ++i) std::memcpy(f_ptr + i * 4, init_factors, 4);
        }
        uint16_t init_aniso[4] = { 0x4420, 0x46d5, 0xba88, 0x0000 };
        uint16_t* a_ptr = static_cast<uint16_t*>(c->aniso.mapped);
        if (a_ptr) {
            for (size_t i = 0; i < size_t(W) * H; ++i) std::memcpy(a_ptr + i * 4, init_aniso, 8);
        }
    }
    c->ae_next = create_buffer(dev, phys, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->ae_exp = create_buffer(dev, phys, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->dhist = create_buffer(dev, phys, size_t(hW15) * hH15 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->full15 = create_buffer(dev, phys, size_t(W15) * H15 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->cf2 = create_buffer(dev, phys, size_t(W) * H * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->cf6 = create_buffer(dev, phys, size_t(W15) * H15 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->cfa = create_buffer(dev, phys, size_t(W15) * H15 * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->c0e = create_buffer(dev, phys, size_t((W15 + 1) / 2) * ((H15 + 1) / 2) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->bfa = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->bfe = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->bf6 = create_buffer(dev, phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->bec = create_buffer(dev, phys, size_t(W0) * H0 * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->ihist[0] = create_buffer(dev, phys, size_t(W15) * H15 * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->ihist[1] = create_buffer(dev, phys, size_t(W15) * H15 * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    size_t t0_sz = std::max(size_t(W0) * H0 * 64, size_t(L1w) * L1h * 64);
    c->t_act[0] = create_buffer(dev, phys, t0_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[1] = create_buffer(dev, phys, size_t(L1w) * L1h * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[2] = create_buffer(dev, phys, size_t(L2w) * L2h * 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[3] = create_buffer(dev, phys, size_t(L2w) * L2h * 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[4] = create_buffer(dev, phys, size_t(L3w) * L3h * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[5] = create_buffer(dev, phys, size_t(L3w) * L3h * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[6] = create_buffer(dev, phys, size_t(L3w) * L3h * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[7] = create_buffer(dev, phys, size_t(L2w) * L2h * 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->t_act[8] = create_buffer(dev, phys, size_t(L1w) * L1h * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    c->out = create_buffer(dev, phys, size_t(oW) * oH * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->wbuf = create_buffer(dev, phys, 16 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->w_enc0 = create_buffer(dev, phys, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->w_dec0 = create_buffer(dev, phys, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    if (mode == "rr") {
        // U-Net and post-input buffers at the oracle's byte sizes (dump buffers.json).  The U-Net
        // buffers hold W0*H0 >> 2*lvl pixels, so their oracle sizes scale with W0*H0; the post inputs
        // scale with W*H.  The reference sizes are the oracle's at 256x144 input (W0*H0 = 256*160 =
        // 40960, W*H = 256*144 = 36864); the scaled size is exact for every input size.
        struct UeSize { const char* name; size_t ref; bool w0h0; };
        static const UeSize kUe[] = {
            {"E0b54", 1310720, true}, {"E0b58", 655360, true}, {"E0b5c", 245760, true},
            {"E0b60", 81920, true},   {"E0b64", 25600, true},  {"E0b68", 6400, true},
            {"E0b70", 245760, true},  {"E0b74", 81920, true},  {"E0b78", 25600, true},
            {"E0b7c", 6400, true},    {"E0b80", 1310720, true}, {"E0b84", 655360, true},
            {"E0b88", 245760, true},  {"E0b8c", 81920, true},  {"E0b90", 25600, true},
            {"E0b98", 655360, true},  {"E0b9c", 245760, true}, {"E0ba0", 81920, true},
            {"E0ba4", 25600, true},   {"E0ba8", 327680, true}, {"E0bac", 81920, true},
            {"E0bb0", 20480, true},   {"E0bb4", 5120, true},   {"E0bb8", 1280, true},
            {"E0bbc", 320, true},     {"E0bc0", 81920, true},  {"E0bc4", 20480, true},
            {"E0bc8", 5120, true},    {"E0bcc", 1280, true},   {"E0bd0", 320, true},
            {"E0bd4", 327680, true},  {"E0bd8", 81920, true},  {"E0bdc", 20480, true},
            {"E0be0", 5120, true},    {"E0be4", 1280, true},
            {"E0c02", 147456, false}, {"E0c16", 147456, false}, {"E0c1a", 147456, false},
            {"E0c1e", 294912, false},
        };
        const size_t w0h0 = size_t(W0) * H0, wh = size_t(W) * H;
        for (const auto& e : kUe) {
            size_t bytes = e.w0h0 ? e.ref * w0h0 / 40960 : e.ref * wh / 36864;
            c->ue[e.name] = create_buffer(dev, phys, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            c->ue[e.name].logical = bytes;
        }
        // pooled2 output: ((hW15+1)/2) x ((hH15+1)/2) words (rrref: alloc(0xc12, E[11]*E[10]*4)).
        const size_t e0c12 = size_t(4) * ((hW15 + 1) / 2) * ((hH15 + 1) / 2);
        c->ue["E0c12"] = create_buffer(dev, phys, e0c12, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->ue["E0c12"].logical = e0c12;
        c->ue["E0d02"] = create_buffer(dev, phys, 8208, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);   // bicubic table
        c->ue["E0d02"].logical = 8208;
        c->rr_dummy = create_buffer(dev, phys, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->rr_tab = create_buffer(dev, phys, 16 * 24 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->expo.logical = 16;   // the oracle's E0c22 is 16 bytes; dumps report that size
    }

    if (!wdir.empty()) {
        if (mode == "sr") {
            std::vector<std::vector<uint8_t>> blob;
            for (int i = 0; i < 10; ++i) blob.push_back(slurp(wdir + "/stage" + std::to_string(i) + ".bin"));
            blob.push_back(slurp(wdir + "/enc0.bin"));
            blob.push_back(slurp(wdir + "/dec0.bin"));
            std::vector<size_t> woff;
            size_t tot = 0;
            for (auto& b : blob) { woff.push_back(tot); tot += (b.size() + 255) & ~size_t(255); }
            for (size_t i = 0; i < blob.size(); ++i) {
                if (!blob[i].empty() && woff[i] + blob[i].size() <= c->wbuf.size) {
                    std::memcpy(static_cast<uint8_t*>(c->wbuf.mapped) + woff[i], blob[i].data(), blob[i].size());
                }
            }
            if (!blob[10].empty() && blob[10].size() <= c->w_enc0.size) {
                std::memcpy(c->w_enc0.mapped, blob[10].data(), blob[10].size());
            }
            if (!blob[11].empty() && blob[11].size() <= c->w_dec0.size) {
                std::memcpy(c->w_dec0.mapped, blob[11].data(), blob[11].size());
            }
            auto pj_raw = slurp(wdir + "/params.json");
            if (!pj_raw.empty()) {
                std::string pj(pj_raw.begin(), pj_raw.end());
                auto s0_vec = jget(pj, "dec0_s0"); if (!s0_vec.empty()) c->dec0_s0 = float(s0_vec[0]);
                auto f0_vec = jget(pj, "dec0_f0"); if (!f0_vec.empty()) c->dec0_f0 = float(f0_vec[0]);
                auto f1_vec = jget(pj, "dec0_f1"); if (!f1_vec.empty()) c->dec0_f1 = float(f1_vec[0]);
            }
        } else {
            size_t offset = 0;
            for (int u = 0; u < 16; ++u) {
                auto b = slurp(wdir + "/blob" + std::to_string(u) + ".bin");
                if (!b.empty() && offset + b.size() <= c->wbuf.size) {
                    c->rr_blob_off[u] = offset;
                    std::memcpy(static_cast<uint8_t*>(c->wbuf.mapped) + offset, b.data(), b.size());
                    offset += (b.size() + 255) & ~size_t(255);
                }
            }
            // rr_level's item table (rr_tab, binding 13) from the packer's params.json next to the blobs.
            auto pj_units = slurp(wdir + "/params.json");
            if (!pj_units.empty() && c->rr_tab.mapped) {
                std::string pj(pj_units.begin(), pj_units.end());
                if (!rr_table_from_params(pj, static_cast<uint32_t*>(c->rr_tab.mapped)))
                    std::fprintf(stderr, "rr_level: params.json does not hold 16 units with meta and s1d2/4/6\n");
            }
            auto bc = slurp(wdir + "/bicubic.bin");
            Buffer& e0d02 = c->ue["E0d02"];
            if (!bc.empty() && bc.size() <= e0d02.size) std::memcpy(e0d02.mapped, bc.data(), bc.size());
        }
    }

    return c;
}

FrameChain::~FrameChain() {
    destroy_buffer(dev, in_col);
    destroy_buffer(dev, in_dep);
    destroy_buffer(dev, in_mv_);
    destroy_buffer(dev, in_norm);
    destroy_buffer(dev, in_rough_);
    destroy_buffer(dev, in_alb);
    destroy_buffer(dev, in_spa);
    destroy_buffer(dev, out_alb);
    destroy_buffer(dev, out_spa);
    destroy_buffer(dev, out_norm);
    destroy_buffer(dev, out_rough_);
    destroy_buffer(dev, g_col);
    destroy_buffer(dev, g_dep);
    destroy_buffer(dev, g_mv);
    destroy_buffer(dev, g_mvhi);
    destroy_buffer(dev, expo);
    destroy_buffer(dev, prev_mvec[0]);
    destroy_buffer(dev, prev_mvec[1]);
    destroy_buffer(dev, mv_ero);
    destroy_buffer(dev, mv_nondil);
    destroy_buffer(dev, hidden[0]);
    destroy_buffer(dev, hidden[1]);
    destroy_buffer(dev, gates);
    destroy_buffer(dev, factors);
    destroy_buffer(dev, aniso);
    destroy_buffer(dev, ae_next);
    destroy_buffer(dev, ae_exp);
    destroy_buffer(dev, dhist);
    destroy_buffer(dev, full15);
    destroy_buffer(dev, cf2);
    destroy_buffer(dev, cf6);
    destroy_buffer(dev, cfa);
    destroy_buffer(dev, c0e);
    destroy_buffer(dev, bfa);
    destroy_buffer(dev, bfe);
    destroy_buffer(dev, bf6);
    destroy_buffer(dev, bec);
    destroy_buffer(dev, ihist[0]);
    destroy_buffer(dev, ihist[1]);
    for (int i = 0; i < 9; ++i) destroy_buffer(dev, t_act[i]);
    destroy_buffer(dev, out);
    destroy_buffer(dev, wbuf);
    destroy_buffer(dev, w_enc0);
    destroy_buffer(dev, w_dec0);

    for (auto& kv : ue) destroy_buffer(dev, kv.second);
    destroy_buffer(dev, rr_dummy);
    destroy_buffer(dev, rr_tab);

    for (auto& pair : pipelines) {
        vkDestroyPipeline(dev, pair.second.pipeline, nullptr);
        vkDestroyPipelineLayout(dev, pair.second.layout, nullptr);
        vkDestroyDescriptorSetLayout(dev, pair.second.dsl, nullptr);
    }
    vkDestroyDescriptorPool(dev, dpool, nullptr);
}

size_t FrameChain::pipeline_count() const { return pipelines.size(); }

// ---- oracle-injection diagnosis (DLSS5_AMD_INJECT_DIR / _MASK) ------------------------------
// The dumps under build/oracle/*/dump/frame{0,1} carry every buffer NeuralHub's own kernels
// wrote, launch by launch.  With injection on, record() additionally copies those bytes over
// the chain's own results after the named dispatch points, so one run can *follow* the oracle
// from any point onward: the end-to-end PSNR of an injected run is what the whole chain
// downstream of that point contributes.  Diagnostic only; unset, record() is unchanged.

void FrameChain::set_injection(const std::string& dir, const std::string& mask) {
    inj_dir = dir;
    size_t i = 0;
    while (i < mask.size()) {
        size_t j = mask.find(',', i);
        if (j == std::string::npos) j = mask.size();
        if (j > i) inj_mask.push_back(mask.substr(i, j - i));
        i = j + 1;
    }
    if (mode == "sr") {
        inj_targets = {
            {"g_col", &g_col}, {"g_dep", &g_dep}, {"g_mv", &g_mv}, {"g_mvhi", &g_mvhi},
            {"prev_mvec0", &prev_mvec[0]}, {"prev_mvec1", &prev_mvec[1]},
            {"mv_ero", &mv_ero}, {"mv_nondil", &mv_nondil},
            {"hidden0", &hidden[0]}, {"hidden1", &hidden[1]},
            {"gates", &gates}, {"factors", &factors}, {"aniso", &aniso},
            {"dhist", &dhist}, {"full15", &full15},
            {"ihist0", &ihist[0]}, {"ihist1", &ihist[1]},
            {"t0", &t_act[0]}, {"t1", &t_act[1]}, {"t2", &t_act[2]}, {"t3", &t_act[3]},
            {"t4", &t_act[4]}, {"t5", &t_act[5]}, {"t6", &t_act[6]}, {"t7", &t_act[7]},
            {"t8", &t_act[8]},
            {"out", &out},
        };
    } else {
        inj_targets = {
            {"E0c6e", &g_col}, {"E0c72", &g_dep}, {"E0c76", &g_mv},
            {"E0c7a", &out_norm}, {"E0c7e", &out_alb}, {"E0c82", &out_spa}, {"E0c8e", &out_rough_},
            {"E0c26", &ae_next}, {"E0c2a", &ae_exp}, {"E0cee", &dhist}, {"E0c22", &expo},
            {"E0be8", &t_act[0]}, {"E0bec", &bec}, {"E0bf6", &bf6}, {"E0bf2", &bf6},
            {"E0bfa", &bfa}, {"E0bfe", &bfe},
            {"E0cf2", &cf2}, {"E0c0a", &ihist[0]}, {"E0c0e", &c0e},
            {"E0cf6", &cf6}, {"E0cfa", &cfa}, {"E0cfe", &out},
        };
        inj_targets["E0c06"] = &ihist[1];
        for (auto& kv : ue) inj_targets[kv.first] = &kv.second;
    }
}

bool FrameChain::inj_wanted(const char* point) const {
    if (inj_dir.empty() || fc >= 2) return false;
    if (inj_mask.empty()) return true;
    for (auto& m : inj_mask) if (m == point) return true;
    return false;
}

bool FrameChain::read_buffer(const std::string& name, std::vector<uint8_t>* out) {
    if (inj_targets.empty()) set_injection("", "");
    auto it = inj_targets.find(name);
    if (it == inj_targets.end() || !it->second->mapped) return false;
    const size_t n = it->second->logical ? it->second->logical : it->second->size;
    out->assign(static_cast<uint8_t*>(it->second->mapped),
                static_cast<uint8_t*>(it->second->mapped) + n);
    return true;
}

void FrameChain::inject_after(VkCommandBuffer cmd, const char* point, int launch) {
    if (!inj_wanted(point)) return;
    char lbuf[16];
    std::snprintf(lbuf, sizeof(lbuf), "%03d", launch);
    std::string base = inj_dir + "/frame" + std::to_string(fc) + "/" + lbuf + ".";
    bool any = false;
    std::string names;
    for (const auto& kv : inj_targets) {
        const std::string& name = kv.first;
        auto bytes = slurp(base + name + ".bin");
        if (bytes.empty()) continue;          // this launch did not write that buffer
        Buffer* target = kv.second;
        std::string key = std::to_string(fc) + "/" + point + "/" + name;
        auto it = inj_shadow.find(key);
        if (it == inj_shadow.end()) {
            Buffer s = create_buffer(dev, phys, std::min(bytes.size(), target->size),
                                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            inj_shadow[key] = s;
            it = inj_shadow.find(key);
        }
        Buffer& shadow = it->second;
        std::memcpy(shadow.mapped, bytes.data(), std::min(bytes.size(), shadow.size));
        VkBufferCopy region{};
        region.size = std::min({bytes.size(), shadow.size, target->size}) & ~size_t(3);
        if (region.size == 0) continue;
        vkCmdCopyBuffer(cmd, shadow.buffer, target->buffer, 1, &region);
        any = true;
        names += name + " ";
    }
    if (any) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        std::fprintf(stderr, "INJECT frame%d %s launch %d <- %s\n", fc, point, launch, names.c_str());
    }
}

// DLSS5_AMD_MAX_DISPATCH=<n>: record only the first n dispatches of a frame and skip the rest.  A
// bisect switch for finding which dispatch of a chain misbehaves; unset (the normal case, and every
// run of vk_runner) it records everything, exactly as before.
static int max_dispatch() {
    static int n = -1;
    if (n < 0) {
        const char* e = std::getenv("DLSS5_AMD_MAX_DISPATCH");
        n = (e && *e) ? std::atoi(e) : 0x7fffffff;
    }
    return n;
}
static int g_dispatch_seen = 0;

// Helper to dispatch a pipeline with buffers and push constants
void FrameChain::dispatch(VkCommandBuffer cmd, const std::string& name,
                          const std::vector<Buffer*>& bufs, const void* push,
                          size_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
    if (g_dispatch_seen++ >= max_dispatch()) return;
    auto it = pipelines.find(name);
    if (it == pipelines.end()) return;
    const auto& p = it->second;
    static const bool trace = std::getenv("DLSS5_AMD_TRACE") != nullptr;
    if (trace)
        std::fprintf(stderr, "DISPATCH frame%d %-14s %-34s grid %ux%ux%u\n", fc, name.c_str(),
                     trace_label.c_str(), gx, gy, gz);

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &p.dsl;
    VkDescriptorSet ds;
    vkAllocateDescriptorSets(dev, &dsai, &ds);

    std::vector<VkDescriptorBufferInfo> dbis(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        dbis[i].buffer = bufs[i]->buffer;
        dbis[i].offset = 0;
        dbis[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = uint32_t(i);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &dbis[i];
    }
    vkUpdateDescriptorSets(dev, uint32_t(writes.size()), writes.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &ds, 0, nullptr);
    if (push && push_size > 0) {
        vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, uint32_t(push_size), push);
    }
    if (prof_hook) prof_hook(name.c_str(), 0);
    vkCmdDispatch(cmd, gx, gy, gz);
    if (prof_hook) prof_hook(name.c_str(), 1);

    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// ---- RR U-Net (preset F): the 16 k_level launches -----------------------------------------------
// Frame 1 has them at launches 5..20 (frame 0 at 4..19).  Template ints, in_ch and the buffer roles
// are decoded from the oracle's launches.jsonl (identical in both frames) and agree with
// tools/rr/rrref/rrref.cpp's level().  Role names are oracle buffer names; "" = the argument is null.
// Args offsets: in +0x10, skip +0x18, out +0x78, out +0x80, in +0x88, in +0x90, out +0x98,
// out +0xa0, out +0xc0 / +0xc8 / +0xd0 / +0xd8.  rr_level binding 2 takes +0x78, or +0x98 when a
// unit has no +0x78 (unit 15).
enum RrRole { R_IN, R_SKIP, R_O78, R_O80, R_I88, R_I90, R_O98, R_OA0, R_OC0, R_OC8, R_OD0, R_OD8, R_N };
struct RrUnit { int C, g, mode, Cdown, Cup, n, in_ch; const char* role[R_N]; };
static const int kRrUnitLevel[16] = {0, 1, 2, 3, 4, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0};
static const RrUnit kRrUnits[16] = {
    {32, 2, 5, 0, 0, 4, 64,  // unit 0 (launch 5)
     {"E0be8", "", "E0b54", "", "E0bec", "", "E0ba8", "E0bc0", "", "", "", ""}},
    {64, 2, 1, 96, 0, 1, 32,  // unit 1 (launch 6)
     {"E0b54", "", "E0b58", "E0b70", "E0bc0", "", "E0bac", "E0bc4", "", "", "", ""}},
    {96, 4, 0, 128, 0, 1, 96,  // unit 2 (launch 7)
     {"E0b70", "", "E0b5c", "E0b74", "E0bc4", "", "E0bb0", "E0bc8", "", "", "", ""}},
    {128, 4, 0, 160, 0, 1, 128,  // unit 3 (launch 8)
     {"E0b74", "", "E0b60", "E0b78", "E0bc8", "", "E0bb4", "E0bcc", "", "", "", ""}},
    {160, 8, 0, 160, 0, 1, 160,  // unit 4 (launch 9)
     {"E0b78", "", "E0b64", "E0b7c", "E0bcc", "", "E0bb8", "E0bd0", "", "", "", ""}},
    {160, 8, 0, 0, 0, 1, 160,  // unit 5 (launch 10)
     {"E0b7c", "", "E0b68", "", "E0bd0", "", "E0bbc", "", "", "", "", ""}},
    {160, 8, 2, 0, 160, 1, 160,  // unit 6 (launch 11)
     {"E0b68", "E0b64", "E0b90", "", "", "", "", "", "", "", "", ""}},
    {160, 8, 3, 0, 0, 1, 160,  // unit 7 (launch 12)
     {"E0b90", "", "E0ba4", "", "E0bb8", "E0bbc", "E0be4", "", "", "", "", ""}},
    {128, 4, 2, 0, 160, 1, 160,  // unit 8 (launch 13)
     {"E0ba4", "E0b60", "E0b8c", "", "", "", "", "", "", "", "", ""}},
    {128, 4, 3, 0, 0, 1, 128,  // unit 9 (launch 14)
     {"E0b8c", "", "E0ba0", "", "E0bb4", "E0be4", "E0be0", "", "", "", "", ""}},
    {96, 4, 2, 0, 128, 1, 128,  // unit 10 (launch 15)
     {"E0ba0", "E0b5c", "E0b88", "", "", "", "", "", "", "", "", ""}},
    {96, 4, 3, 0, 0, 1, 96,  // unit 11 (launch 16)
     {"E0b88", "", "E0b9c", "", "E0bb0", "E0be0", "E0bdc", "", "", "", "", ""}},
    {64, 2, 2, 0, 96, 1, 96,  // unit 12 (launch 17)
     {"E0b9c", "E0b58", "E0b84", "", "", "", "", "", "", "", "", ""}},
    {64, 2, 3, 0, 0, 1, 64,  // unit 13 (launch 18)
     {"E0b84", "", "E0b98", "", "E0bac", "E0bdc", "E0bd8", "", "", "", "", ""}},
    {32, 2, 2, 0, 64, 4, 64,  // unit 14 (launch 19)
     {"E0b98", "E0b54", "E0b80", "", "", "", "", "", "", "", "", ""}},
    {32, 2, 4, 0, 0, 4, 32,  // unit 15 (launch 20)
     {"E0b80", "", "", "", "E0ba8", "E0bd8", "E0bd4", "", "E0c02", "E0c16", "E0c1a", "E0c1e"}},
};

Buffer* FrameChain::rr_buf(const std::string& n) {
    if (n.empty()) return nullptr;
    if (n == "E0be8") return &t_act[0];     // enc0 features: the same bytes as t_act[0]
    if (n == "E0bec") return &bec;          // enc0 bec, read by launch 5 at +0x88
    auto it = ue.find(n);
    return it == ue.end() ? nullptr : &it->second;
}

void FrameChain::record(VkCommandBuffer cmd, const FrameParams& fp) {
    int W = this->W, H = this->H, oW = this->oW, oH = this->oH;
    int W0 = this->W0, H0 = this->H0, L1w = this->L1w, L1h = this->L1h;
    int L2w = this->L2w, L2h = this->L2h, L3w = this->L3w, L3h = this->L3h;
    int W15 = this->W15, H15 = this->H15, hW15 = this->hW15, hH15 = this->hH15;
    int p = this->p, fc = this->fc;
    int reset = fp.reset;
    int fmt_col = fp.fmt_col, fmt_mv = fp.fmt_mv, fmt_dep = fp.fmt_dep;
    int mvhi = fp.mvhi, ldr = fp.ldr;
    float exposure_val = fp.exposure, pre_val = fp.pre_exposure;
    float jx = fp.jx, jy = fp.jy, jpx = fp.jpx, jpy = fp.jpy;
    float dec0_s0 = this->dec0_s0, dec0_f0 = this->dec0_f0, dec0_f1 = this->dec0_f1;
    const std::string& preset = this->preset;
    const std::string& mode = this->mode;

    if (reset) {
        std::memset(prev_mvec[0].mapped, 0, prev_mvec[0].size);
        std::memset(prev_mvec[1].mapped, 0, prev_mvec[1].size);
        std::memset(hidden[0].mapped, 0, hidden[0].size);
        std::memset(hidden[1].mapped, 0, hidden[1].size);
        std::memset(ihist[0].mapped, 0, ihist[0].size);
        std::memset(ihist[1].mapped, 0, ihist[1].size);
    }

    if (mode == "rr") {
        float exp_data[4] = { exposure_val, pre_val, 1.0f / std::max(exposure_val, 1e-4f), 0.0f };
        std::memcpy(expo.mapped, exp_data, 16);
    } else {
        uint16_t* expo_ptr = static_cast<uint16_t*>(expo.mapped);
        uint16_t h_exp = f32_to_f16_rne(exposure_val * pre_val);
        expo_ptr[p] = h_exp;
    }

    // The caller has begun cmd; every descriptor set below is freshly allocated.
    vkResetDescriptorPool(dev, dpool, 0);
    g_dispatch_seen = 0;

    if (mode == "sr") {
        // Dump launch numbers per dispatch point (identical in frame 0 and frame 1 dumps).
        static const int sr_l[] = {/*gather*/0, /*enc0*/2, /*stage0*/3, /*stage1*/4,
                                   /*tube0*/5, /*tube1*/6, /*tube2*/7, /*tube3*/8, /*tube4*/9,
                                   /*tube5*/10, /*stage8*/11, /*stage9*/12,
                                   /*dec0*/13, /*post*/14, /*downsample*/15};
        // 1. Gather
        {
            uint8_t push[88] = {0};
            uint32_t pcol = W * 8, pmv = W * 4, pdep = W * 4;
            std::memcpy(push + 24, &pcol, 4);
            std::memcpy(push + 28, &pmv, 4);
            std::memcpy(push + 32, &pdep, 4);
            std::memcpy(push + 36, &fmt_col, 4);
            std::memcpy(push + 40, &fmt_mv, 4);
            std::memcpy(push + 44, &fmt_dep, 4);
            std::memcpy(push + 72, &W, 4);
            std::memcpy(push + 76, &H, 4);
            std::memcpy(push + 80, &mvhi, 4);
            dispatch(cmd, "sr_gather", {&in_col, &in_mv_, &in_dep, &g_col, &g_mv, &g_dep}, push, 88, (W + 15) / 16, (H + 15) / 16, 1);
            inject_after(cmd, "gather", sr_l[0]);
        }

        if (mvhi) {
            uint8_t push[32] = {0};
            uint32_t pmv = oW * 4;
            std::memcpy(push + 8, &pmv, 4);
            std::memcpy(push + 12, &fmt_mv, 4);
            std::memcpy(push + 24, &oW, 4);
            std::memcpy(push + 28, &oH, 4);
            dispatch(cmd, "sr_gather_mv", {&in_mv_, &g_mvhi}, push, 32, (oW + 15) / 16, (oH + 15) / 16, 1);
        }

        // 2. Exposure
        {
            uint8_t push[32] = {0};
            std::memcpy(push + 8, &exposure_val, 4);
            std::memcpy(push + 12, &pre_val, 4);
            std::memcpy(push + 24, &p, 4);
            dispatch(cmd, "sr_exposure", {&expo, &expo}, push, 32, 1, 1, 1);
        }

        // 3. Enc0
        {
            uint8_t push[240] = {0};
            std::memcpy(push + 88, &W, 4); std::memcpy(push + 92, &H, 4);
            std::memcpy(push + 104, &W0, 4); std::memcpy(push + 108, &H0, 4);
            std::memcpy(push + 112, &W15, 4); std::memcpy(push + 116, &H15, 4);
            std::memcpy(push + 120, &hW15, 4); std::memcpy(push + 124, &hH15, 4);
            std::memcpy(push + 136, &p, 4);
            std::memcpy(push + 140, &jx, 4);
            std::memcpy(push + 144, &jy, 4);
            std::memcpy(push + 216, &reset, 4);
            std::memcpy(push + 220, &fc, 4);
            if (preset == "L") {
                dispatch(cmd, "sr_enc0L_in", {&g_col, &g_dep, mvhi ? &g_mvhi : &g_mv, &prev_mvec[p], &dhist, &hidden[p],
                                     &prev_mvec[p ^ 1], &mv_ero, &mv_nondil, &t_act[0], &w_enc0}, push, 240, (W * H + 63) / 64, 1, 1);
                dispatch(cmd, "sr_l0", {&wbuf, &t_act[0], &t_act[1]}, nullptr, 0, 32, 1, 1);
            } else {
                dispatch(cmd, "sr_enc0", {&g_col, &g_dep, mvhi ? &g_mvhi : &g_mv, &prev_mvec[p], &dhist, &hidden[p],
                                     &prev_mvec[p ^ 1], &mv_ero, &mv_nondil, &t_act[0], &w_enc0}, push, 240, W0 / 8, H0 / 8, 1);
            }
            inject_after(cmd, "enc0", sr_l[1]);
        }

        // 4. Stages 0..9
        static const uint32_t k_stage_w_off[10] = {
            0, 71936, 235520, 419840, 604160, 788480, 972800, 1157120, 1341440, 1505536
        };
        for (int st = 0; st < 10; ++st) {
            uint8_t push[56] = {0};
            uint32_t w_off = k_stage_w_off[st];
            std::memcpy(push + 0, &w_off, 4);
            int dimW, dimH, s0, s1;
            const int pe = (p == 0) * 2, po = (p != 0) * 2;
            if (st == 0) { dimW = L1w; dimH = L1h; s0 = pe; s1 = 2;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                // binding 4 is the mode-2 skip; the oracle passes no offset 16 in mode 0,
                // so bind the stage's own input - the shader never reads it here.
                dispatch(cmd, "sr_stage", {&wbuf, &t_act[0], &t_act[1], &t_act[2], &t_act[0]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
            } else if (st == 1) { dimW = L2w; dimH = L2h; s0 = pe; s1 = 2;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                dispatch(cmd, "sr_stage", {&wbuf, &t_act[2], &t_act[3], &t_act[4], &t_act[2]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
            } else if (st <= 7) {
                Buffer* in_b = (st == 2) ? &t_act[4] : (st & 1) ? &t_act[5] : &t_act[6];
                Buffer* out_b = (st & 1) ? &t_act[6] : &t_act[5];
                dimW = L3w; dimH = L3h;
                if ((st & 1) == 0) { s0 = pe; s1 = 2; } else { s0 = po; s1 = 0; }
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                dispatch(cmd, "sr_tube_split", {&wbuf, in_b, out_b}, push, 56, (dimW + s0 + 3) / 4, (dimH + s1 + 3) / 4, 1);
            } else if (st == 8) { dimW = L2w; dimH = L2h; s0 = po; s1 = 0;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                // Oracle launch 11: wbuf+1341440, t6 (in), t3 (skip, StageArgs+16), t7 (out).
                dispatch(cmd, "sr_stage", {&wbuf, &t_act[6], &t_act[7], &t_act[7], &t_act[3]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
            } else { dimW = L1w; dimH = L1h; s0 = po; s1 = 0;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                // Oracle launch 12: wbuf+1505536, t7 (in), t1 (skip, StageArgs+16), t8 (out).
                dispatch(cmd, "sr_stage", {&wbuf, &t_act[7], &t_act[8], &t_act[8], &t_act[1]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
            }
            static const char* st_pt[] = {"stage0", "stage1", "tube0", "tube1", "tube2",
                                          "tube3", "tube4", "tube5", "stage8", "stage9"};
            inject_after(cmd, st_pt[st], sr_l[2 + st]);
        }

        // 5. Dec0
        {
            uint8_t push[160] = {0};
            std::memcpy(push + 64, &W, 4); std::memcpy(push + 68, &H, 4);
            std::memcpy(push + 72, &W, 4); std::memcpy(push + 76, &H, 4);
            std::memcpy(push + 80, &W0, 4); std::memcpy(push + 84, &H0, 4);
            std::memcpy(push + 88, &W15, 4); std::memcpy(push + 92, &H15, 4);
            std::memcpy(push + 96, &dec0_s0, 4);
            std::memcpy(push + 100, &dec0_f0, 4);
            dispatch(cmd, "sr_dec0", {&t_act[8], &prev_mvec[p ^ 1], &hidden[p], &gates, &factors, &aniso, &hidden[p ^ 1], &w_dec0}, push, 160, W0 / 8, H0 / 8, 1);
            inject_after(cmd, "dec0", sr_l[12]);
        }

        // 6. Post
        {
            uint8_t push[256] = {0};
            std::memcpy(push + 16, &p, 4);
            std::memcpy(push + 136, &W, 4); std::memcpy(push + 140, &H, 4);
            std::memcpy(push + 152, &W15, 4); std::memcpy(push + 156, &H15, 4);
            std::memcpy(push + 160, &hW15, 4); std::memcpy(push + 164, &hH15, 4);
            std::memcpy(push + 168, &dec0_f1, 4);
            float sx15 = float(W) / float(W15);
            float sy15 = float(H) / float(H15);
            std::memcpy(push + 172, &sx15, 4);
            std::memcpy(push + 176, &sy15, 4);
            std::memcpy(push + 180, &jx, 4);
            std::memcpy(push + 184, &jy, 4);
            std::memcpy(push + 208, &reset, 4);
            std::memcpy(push + 212, &ldr, 4); std::memcpy(push + 216, &mvhi, 4);
            float mvoffx = (jpx - jx) * (float(W15) / float(W));
            float mvoffy = (jpy - jy) * (float(H15) / float(H));
            std::memcpy(push + 248, &mvoffx, 4);
            std::memcpy(push + 252, &mvoffy, 4);
            dispatch(cmd, "sr_post", {&g_col, &expo, &gates, &factors, &aniso, &ihist[p], &prev_mvec[p ^ 1], &mv_ero, &mv_nondil,
                                 &ihist[p ^ 1], &full15, &dhist}, push, 256, (W15 + 15) / 16, (hH15 + 15) / 16, 1);
            inject_after(cmd, "post", sr_l[13]);
        }

        // 7. Downsample
        {
            uint8_t push[88] = {0};
            std::memcpy(push + 24, &p, 4);
            std::memcpy(push + 40, &W, 4); std::memcpy(push + 44, &H, 4);
            std::memcpy(push + 48, &W15, 4); std::memcpy(push + 52, &H15, 4);
            std::memcpy(push + 56, &oW, 4); std::memcpy(push + 60, &oH, 4);
            float sx = float(W) / float((oW + 1) & ~1), sy = float(H) / float((oH + 1) & ~1);
            std::memcpy(push + 64, &sx, 4); std::memcpy(push + 68, &sy, 4);
            std::memcpy(push + 72, &jx, 4); std::memcpy(push + 76, &jy, 4);
            uint32_t log210 = 0x40549a78;
            std::memcpy(push + 80, &log210, 4);
            std::memcpy(push + 84, &ldr, 4);
            dispatch(cmd, "sr_downsample", {&full15, &g_col, &expo, &out}, push, 88, (oW + 15) / 16, (oH + 15) / 16, 1);
            inject_after(cmd, "downsample", sr_l[14]);
        }
    } else {
        // RR pipeline, in NeuralHub's launch order (launches.jsonl of the oracle; rrref.cpp).  Launch 1,
        // k_pooled2, runs on every frame that is not a reset (rrref: FUN_180127f10), so the dump's launch
        // numbers are one higher from there on; roff carries that shift.
        const bool pooled = (reset == 0);
        const int roff = pooled ? 1 : 0;
        // 1. Gather (launch 0)
        trace_label = "launch 0 k_rr_gather";
        {
            uint8_t push[8] = {0};
            std::memcpy(push + 0, &W, 4); std::memcpy(push + 4, &H, 4);
            dispatch(cmd, "rr_gather", {&in_col, &in_mv_, &in_dep, &in_norm, &in_alb, &in_spa, &in_rough_, &g_col, &g_dep, &g_mv, &out_norm, &out_alb, &out_spa, &out_rough_}, push, 8, (W + 15) / 16, (H + 15) / 16, 1);
            inject_after(cmd, "gather", 0);
        }
        // 1b. History pooling (launch 1, not on reset): c0e (E0c0e) -> E0c12
        if (pooled) {
            trace_label = "launch 1 k_pooled2";
            uint8_t push[32] = {0};
            const int inW = hW15, inH = hH15;                      // rrref E[8], E[9]
            const int oW2 = (hW15 + 1) / 2, oH2 = (hH15 + 1) / 2;  // rrref E[10], E[11]
            std::memcpy(push + 8, &inW, 4); std::memcpy(push + 12, &inH, 4);
            std::memcpy(push + 24, &oW2, 4); std::memcpy(push + 28, &oH2, 4);
            dispatch(cmd, "rr_pooled2", {&c0e, &ue.at("E0c12")}, push, 32, (oW2 + 15) / 16, (oH2 + 15) / 16, 1);
            inject_after(cmd, "pooled2", 1);
        }
        // 2. Auto exposure (launches 1 + roff, 2 + roff)
        {
            uint8_t push[16] = {0};
            std::memcpy(push + 8, &W, 4);
            std::memcpy(push + 12, &H, 4);
            trace_label = "launch " + std::to_string(1 + roff) + " k_ae_block";
            dispatch(cmd, "rr_ae_block", {&g_col, &dhist}, push, 16, (W + 15) / 16, (H + 15) / 16, 1);
            inject_after(cmd, "ae_block", 1 + roff);
            uint8_t push_final[8] = {0};
            std::memcpy(push_final + 0, &W, 4); std::memcpy(push_final + 4, &H, 4);
            trace_label = "launch " + std::to_string(2 + roff) + " k_ae_final";
            dispatch(cmd, "rr_ae_final", {&dhist, &expo, &ae_next, &ae_exp}, push_final, 8, 1, 1, 1);
            inject_after(cmd, "ae_final", 2 + roff);
        }
        // 3. Enc0 feat (launch 3 + roff)
        {
            uint8_t push[16] = {0};
            std::memcpy(push + 0, &W0, 4); std::memcpy(push + 4, &H0, 4);
            std::memcpy(push + 8, &W, 4); std::memcpy(push + 12, &H, 4);
            trace_label = "launch " + std::to_string(3 + roff) + " k_enc0_feat";
            dispatch(cmd, "rr_enc0_feat", {&g_col, &g_mv, &g_dep, &out_norm, &out_alb, &out_spa, &out_rough_, &expo,
                                     &t_act[0], &bec, &bf6, &bfa, &bfe}, push, 16, (W0 + 15) / 16, (H0 + 15) / 16, 1);
            inject_after(cmd, "enc0_feat", 3 + roff);
        }
        // 4. U-Net: one rr_level dispatch per oracle k_level launch (launches 4 + roff + u).
        for (int u = 0; u < 16; ++u) {
            const RrUnit& U = kRrUnits[u];
            const int uw = W0 >> kRrUnitLevel[u], uh = H0 >> kRrUnitLevel[u];
            // Args +8 / +12 (rrref's shift): s6_2 for the encoder, s4_0 for unit 5 and the dec*_0 units
            // (mode 2), s0_4 for dec*_1 and the last unit (modes 3, 4).  Indexed [class][parity].
            const int shc = (u == 5 || U.mode == 2) ? 1 : (U.mode == 3 || U.mode == 4) ? 2 : 0;
            static const int kShift[3][2][2] = {{{2, 6}, {6, 2}}, {{4, 0}, {0, 4}}, {{0, 4}, {4, 0}}};
            const int sx = kShift[shc][p][0], sy = kShift[shc][p][1];
            const int gx = (uw + sx + 7) >> 3, gy = (uh + sy + 7) >> 3;

            uint8_t push[68] = {0};
            const int ints[10] = {uw, uh, sx, sy, U.mode, U.C, U.g, U.Cdown, U.Cup, U.n};
            std::memcpy(push + 0, ints, sizeof ints);          // offsets 0..39
            const int in_ch = U.in_ch, blob_off = int(rr_blob_off[u]), unit_idx = u;
            std::memcpy(push + 56, &in_ch, 4);                  // offset 56: channels per token of binding 1
            std::memcpy(push + 60, &blob_off, 4);               // offset 60: this unit's weights in binding 0
            std::memcpy(push + 64, &unit_idx, 4);               // offset 64: this unit's row of binding 13 (rr_tab)

            Buffer* in = rr_buf(U.role[R_IN]);
            Buffer* o78 = rr_buf(U.role[R_O78]);
            Buffer* o98 = rr_buf(U.role[R_O98]);
            Buffer* primary = o78 ? o78 : o98;                  // binding 2 is the one the shader writes
            auto or_in = [&](Buffer* b) { return b ? b : in; };
            auto or_dummy = [&](Buffer* b) { return b ? b : &rr_dummy; };
            std::vector<Buffer*> bufs = {
                &wbuf, in, primary,
                or_in(rr_buf(U.role[R_SKIP])),
                or_dummy(rr_buf(U.role[R_O80])),
                or_in(rr_buf(U.role[R_I88])), or_in(rr_buf(U.role[R_I90])),
                or_dummy(o98), or_dummy(rr_buf(U.role[R_OA0])),
                or_dummy(rr_buf(U.role[R_OC0])), or_dummy(rr_buf(U.role[R_OC8])),
                or_dummy(rr_buf(U.role[R_OD0])), or_dummy(rr_buf(U.role[R_OD8])),
                &rr_tab,
            };
            char lbl[96];
            std::snprintf(lbl, sizeof lbl, "launch %d k_level<%d,%d,%d,%d,%d,%d>", u + 4 + roff,
                          U.C, U.g, U.mode, U.Cdown, U.Cup, U.n);
            trace_label = lbl;
            dispatch(cmd, "rr_level", bufs, push, 68, gx, gy, 1);
            inject_after(cmd, ("L" + std::to_string(u + 5)).c_str(), u + 4 + roff);
        }
        // 5. Post prep (launch 20 + roff): inputs are the U-Net's own outputs E0bd4, E0c16, E0c1e
        {
            uint8_t push[40] = {0};
            std::memcpy(push + 8, &W0, 4);
            std::memcpy(push + 32, &W, 4); std::memcpy(push + 36, &H, 4);
            trace_label = "launch " + std::to_string(20 + roff) + " k_post_prep";
            dispatch(cmd, "rr_post_prep", {&ue.at("E0bd4"), &ue.at("E0c16"), &ue.at("E0c1e"), &cf2}, push, 40, (W + 15) / 16, (H + 15) / 16, 1);
            inject_after(cmd, "post_prep", 20 + roff);
        }
        // 6. Post (launch 21 + roff)
        {
            uint8_t push[120] = {0};
            std::memcpy(push + 0, &W, 4);
            std::memcpy(push + 4, &H, 4);
            std::memcpy(push + 8, &W15, 4);
            std::memcpy(push + 12, &H15, 4);
            std::memcpy(push + 16, &reset, 4);
            float sx = float(W) / float(W15);
            float sy = float(H) / float(H15);
            std::memcpy(push + 20, &sx, 4);
            std::memcpy(push + 24, &sy, 4);
            std::memcpy(push + 28, &jx, 4);
            std::memcpy(push + 32, &jy, 4);
            std::memcpy(push + 36, &exposure_val, 4);
            trace_label = "launch " + std::to_string(21 + roff) + " k_post";
            dispatch(cmd, "rr_post", {&cf2, &expo, &cf6, &cfa, &ihist[0], &c0e}, push, 120, (W15 + 15) / 16, (H15 + 15) / 16, 1);
            inject_after(cmd, "post", 21 + roff);
        }
        // 7. Downsample (launch 22 + roff)
        {
            uint8_t push[48] = {0};
            std::memcpy(push + 16, &W15, 4); std::memcpy(push + 20, &H15, 4);
            std::memcpy(push + 40, &oW, 4); std::memcpy(push + 44, &oH, 4);
            trace_label = "launch " + std::to_string(22 + roff) + " k_downsample";
            dispatch(cmd, "rr_downsample", {&cf6, &cfa, &expo, &out}, push, 48, (oW + 15) / 16, (oH + 15) / 16, 1);
            inject_after(cmd, "downsample", 22 + roff);
        }
    }
}

} // namespace ngxd
