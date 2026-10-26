#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>

// Helper to load binary files
static std::vector<uint8_t> slurp(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

// Flat JSON parser
static std::vector<double> jget(const std::string& txt, const std::string& key) {
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

static int jint(const std::string& txt, const std::string& key) {
    auto v = jget(txt, key);
    return v.empty() ? 0 : int(v[0]);
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    size_t size = 0;
};

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
    bci.usage = usage | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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

struct Pipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
};

static Pipeline create_compute_pipeline(VkDevice dev, const std::vector<uint32_t>& spv, uint32_t num_bindings, uint32_t push_size) {
    Pipeline p;
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
    vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline);
    vkDestroyShaderModule(dev, mod, nullptr);
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

int main(int argc, char** argv) {
    std::string mode = "sr";
    std::string preset = "M";
    std::string wdir = "";
    std::string fdir = "";
    std::string odir = "";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (a == "--preset" && i + 1 < argc) preset = argv[++i];
        else if (a == "--weights" && i + 1 < argc) wdir = argv[++i];
        else if (a == "--in" && i + 1 < argc) fdir = argv[++i];
        else if (a == "--out" && i + 1 < argc) odir = argv[++i];
    }

    if (fdir.empty() || odir.empty()) {
        std::fprintf(stderr, "usage: vk_runner --mode sr|rr --preset M|L|F --weights <dir> --in <dir> --out <dir>\n");
        return 1;
    }

    // Read frame.json
    auto fj_raw = slurp(fdir + "/frame.json");
    if (fj_raw.empty()) {
        std::fprintf(stderr, "cannot read %s/frame.json\n", fdir.c_str());
        return 1;
    }
    std::string fj(fj_raw.begin(), fj_raw.end());
    auto rd = jget(fj, "render"), od = jget(fj, "output");
    int nframes = int(jget(fj, "frames").at(0));
    int W = int(rd[0]), H = int(rd[1]), oW = int(od[0]), oH = int(od[1]);

    // Dimensions
    int W0 = (W + 31) & ~31, H0 = (H + 31) & ~31;
    int L1w = W0 >> 1, L1h = H0 >> 1, L2w = W0 >> 2, L2h = H0 >> 2, L3w = W0 >> 3, L3h = H0 >> 3;
    int W15 = (oW * 3 + 1) >> 1, H15 = (oH * 3 + 1) >> 1;
    int hW15 = (W15 + 1) / 2, hH15 = (H15 + 1) / 2;
    int mvpitch = (W + 63) & ~63;

    // Vulkan init
    VkApplicationInfo app_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app_info.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app_info;
    VkInstance inst;
    vkCreateInstance(&ici, nullptr, &inst);

    uint32_t num_phys = 0;
    vkEnumeratePhysicalDevices(inst, &num_phys, nullptr);
    std::vector<VkPhysicalDevice> phys(num_phys);
    vkEnumeratePhysicalDevices(inst, &num_phys, phys.data());

    VkPhysicalDevice picked_phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties picked_props{};
    for (auto p : phys) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(p, &pr);
        if (std::string(pr.deviceName).find("9070") != std::string::npos ||
            pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            picked_phys = p;
            picked_props = pr;
            break;
        }
    }
    if (!picked_phys && !phys.empty()) {
        picked_phys = phys[0];
        vkGetPhysicalDeviceProperties(picked_phys, &picked_props);
    }

    std::printf("VK_DEVICE: %s\n", picked_props.deviceName);
    std::fflush(stdout);

    float qprio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &qprio;

    VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDevice16BitStorageFeatures s16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDevice8BitStorageFeaturesKHR s8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR};
    VkPhysicalDeviceFloat16Int8FeaturesKHR f16i8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};

    feats2.pNext = &s16;
    s16.pNext = &s8;
    s8.pNext = &f16i8;
    s16.storageBuffer16BitAccess = VK_TRUE;
    s8.storageBuffer8BitAccess = VK_TRUE;
    f16i8.shaderFloat16 = VK_TRUE;
    f16i8.shaderInt8 = VK_TRUE;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &feats2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkDevice dev;
    vkCreateDevice(picked_phys, &dci, nullptr, &dev);

    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool;
    vkCreateCommandPool(dev, &cpci, nullptr, &cpool);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev, &cbai, &cmd);

    // Query pool for GPU timestamps
    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpci.queryCount = 2;
    VkQueryPool qpool;
    vkCreateQueryPool(dev, &qpci, nullptr, &qpool);

    // Load Shaders and Create Pipelines
    // Locate shaders directory: build/shaders or shaders
    std::string sdir = "build/shaders";
    auto load_spv = [&](const std::string& name) -> std::vector<uint32_t> {
        auto raw = slurp(sdir + "/" + name + ".spv");
        if (raw.empty()) raw = slurp("shaders/" + name + ".spv");
        if (raw.empty()) return {};
        std::vector<uint32_t> spv(raw.size() / 4);
        std::memcpy(spv.data(), raw.data(), raw.size());
        return spv;
    };

    std::map<std::string, Pipeline> pipelines;
    auto add_pipe = [&](const std::string& name, uint32_t bindings, uint32_t push_size) {
        auto spv = load_spv(name);
        if (spv.empty()) {
            std::fprintf(stderr, "Missing shader %s.spv\n", name.c_str());
            return;
        }
        pipelines[name] = create_compute_pipeline(dev, spv, bindings, push_size);
    };

    if (mode == "sr") {
        add_pipe("sr_gather", 6, 88);
        add_pipe("sr_gather_mv", 2, 32);
        add_pipe("sr_exposure", 1, 32);
        add_pipe("sr_ae_luma", 2, 48);
        add_pipe("sr_ae_reduce", 2, 32);
        add_pipe("sr_ae_copy", 2, 24);
        add_pipe("sr_enc0", 11, 240);
        add_pipe("sr_stage", 4, 56);
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
        add_pipe("rr_ae_final", 2, 8);
        add_pipe("rr_enc0_feat", 13, 16);
        add_pipe("rr_level", 3, 16);
        add_pipe("rr_post_prep", 4, 40);
        add_pipe("rr_post", 6, 120);
        add_pipe("rr_downsample", 4, 48);
        add_pipe("rr_pooled2", 2, 32);
        add_pipe("rr_avg2x2", 2, 8);
        add_pipe("rr_ds_t", 2, 8);
        add_pipe("rr_ds_fused", 2, 8);
    }

    std::printf("PIPELINES_CREATED: %zu\n", pipelines.size());
    std::fflush(stdout);

    // Descriptor pool
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 256;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &dps;
    VkDescriptorPool dpool;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &dpool);

    // Allocate Buffers
    Buffer in_col = create_buffer(dev, picked_phys, size_t(W) * H * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer in_dep = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer in_mv  = create_buffer(dev, picked_phys, size_t(std::max(W, oW)) * std::max(H, oH) * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer in_norm = create_buffer(dev, picked_phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer in_rough = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer in_alb = create_buffer(dev, picked_phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer in_spa = create_buffer(dev, picked_phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer out_alb = create_buffer(dev, picked_phys, size_t(W) * H * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer out_spa = create_buffer(dev, picked_phys, size_t(W) * H * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer out_norm = create_buffer(dev, picked_phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer out_rough = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    Buffer g_col = create_buffer(dev, picked_phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer g_dep = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer g_mv  = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer g_mvhi = create_buffer(dev, picked_phys, size_t(oW) * oH * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    Buffer expo = create_buffer(dev, picked_phys, 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (mode == "sr") {
        uint16_t exp_init[8] = { 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0, 0, 0, 0 };
        std::memcpy(expo.mapped, exp_init, 16);
    } else {
        float exp_init[4] = { 1.0f, 1.0f, 1.0f, 0.0f };
        std::memcpy(expo.mapped, exp_init, 16);
    }

    Buffer prev_mvec[2] = {
        create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    };
    Buffer mv_ero = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer mv_nondil = create_buffer(dev, picked_phys, size_t(H) * mvpitch * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer hidden[2] = {
        create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    };
    Buffer gates = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer factors = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer aniso = create_buffer(dev, picked_phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (mode == "rr") {
        uint8_t init_gates[4] = { 58, 223, 207, 0 };
        uint8_t* g_ptr = static_cast<uint8_t*>(gates.mapped);
        if (g_ptr) {
            for (size_t i = 0; i < size_t(W) * H; ++i) std::memcpy(g_ptr + i * 4, init_gates, 4);
        }
        uint8_t init_factors[4] = { 41, 193, 19, 2 };
        uint8_t* f_ptr = static_cast<uint8_t*>(factors.mapped);
        if (f_ptr) {
            for (size_t i = 0; i < size_t(W) * H; ++i) std::memcpy(f_ptr + i * 4, init_factors, 4);
        }
        uint16_t init_aniso[4] = { 0x4420, 0x46d5, 0xba88, 0x0000 };
        uint16_t* a_ptr = static_cast<uint16_t*>(aniso.mapped);
        if (a_ptr) {
            for (size_t i = 0; i < size_t(W) * H; ++i) std::memcpy(a_ptr + i * 4, init_aniso, 8);
        }
    }
    Buffer dhist = create_buffer(dev, picked_phys, size_t(hW15) * hH15 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer full15 = create_buffer(dev, picked_phys, size_t(W15) * H15 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer cf2 = create_buffer(dev, picked_phys, size_t(W) * H * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer cf6 = create_buffer(dev, picked_phys, size_t(W15) * H15 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer cfa = create_buffer(dev, picked_phys, size_t(W15) * H15 * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer c0e = create_buffer(dev, picked_phys, size_t((W15 + 1) / 2) * ((H15 + 1) / 2) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer bfa = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer bfe = create_buffer(dev, picked_phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer bf6 = create_buffer(dev, picked_phys, size_t(W) * H * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer bec = create_buffer(dev, picked_phys, size_t(W0) * H0 * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer ihist[2] = {
        create_buffer(dev, picked_phys, size_t(W15) * H15 * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(W15) * H15 * 8, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    };

    size_t t0_sz = std::max(size_t(W0) * H0 * 64, size_t(L1w) * L1h * 64);
    Buffer t_act[9] = {
        create_buffer(dev, picked_phys, t0_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L1w) * L1h * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L2w) * L2h * 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L2w) * L2h * 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L3w) * L3h * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L3w) * L3h * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L3w) * L3h * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L2w) * L2h * 96, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        create_buffer(dev, picked_phys, size_t(L1w) * L1h * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
    };

    Buffer out = create_buffer(dev, picked_phys, size_t(oW) * oH * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer wbuf = create_buffer(dev, picked_phys, 16 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer w_enc0 = create_buffer(dev, picked_phys, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    Buffer w_dec0 = create_buffer(dev, picked_phys, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    float dec0_s0 = 8.0f, dec0_f0 = 0.078633f, dec0_f1 = 0.0f;
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
                if (!blob[i].empty() && woff[i] + blob[i].size() <= wbuf.size) {
                    std::memcpy(static_cast<uint8_t*>(wbuf.mapped) + woff[i], blob[i].data(), blob[i].size());
                }
            }
            if (!blob[10].empty() && blob[10].size() <= w_enc0.size) {
                std::memcpy(w_enc0.mapped, blob[10].data(), blob[10].size());
            }
            if (!blob[11].empty() && blob[11].size() <= w_dec0.size) {
                std::memcpy(w_dec0.mapped, blob[11].data(), blob[11].size());
            }
            auto pj_raw = slurp(wdir + "/params.json");
            if (!pj_raw.empty()) {
                std::string pj(pj_raw.begin(), pj_raw.end());
                auto s0_vec = jget(pj, "dec0_s0"); if (!s0_vec.empty()) dec0_s0 = float(s0_vec[0]);
                auto f0_vec = jget(pj, "dec0_f0"); if (!f0_vec.empty()) dec0_f0 = float(f0_vec[0]);
                auto f1_vec = jget(pj, "dec0_f1"); if (!f1_vec.empty()) dec0_f1 = float(f1_vec[0]);
            }
        } else {
            size_t offset = 0;
            for (int u = 0; u < 16; ++u) {
                auto b = slurp(wdir + "/blob" + std::to_string(u) + ".bin");
                if (!b.empty() && offset + b.size() <= wbuf.size) {
                    std::memcpy(static_cast<uint8_t*>(wbuf.mapped) + offset, b.data(), b.size());
                    offset += (b.size() + 255) & ~size_t(255);
                }
            }
        }
    }

    // Helper to dispatch a pipeline with buffers and push constants
    auto dispatch = [&](const std::string& name, const std::vector<Buffer*>& bufs, const void* push, size_t push_size, uint32_t gx, uint32_t gy, uint32_t gz) {
        auto it = pipelines.find(name);
        if (it == pipelines.end()) return;
        const auto& p = it->second;

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
        vkCmdDispatch(cmd, gx, gy, gz);

        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    // Frame execution loop
    int p = 0, fc = 0;
    for (int fi = 0; fi < nframes; ++fi) {
        std::string sfx = "_" + std::to_string(fi);
        auto pf_raw = slurp(fdir + "/params" + sfx + ".json");
        std::string pf(pf_raw.begin(), pf_raw.end());

        int fmt_col = jint(pf, "fmt_col"), fmt_mv = jint(pf, "fmt_mv"), fmt_dep = jint(pf, "fmt_dep");
        int mvhi = jint(pf, "mvhi"), ldr = jint(pf, "ldr");
        float exposure_val = float(jget(pf, "exposure").empty() ? 1.0 : jget(pf, "exposure")[0]);
        float pre_val = float(jget(pf, "pre_exposure").empty() ? 1.0 : jget(pf, "pre_exposure")[0]);
        int reset = jint(pf, "reset");
        if (reset) {
            std::memset(prev_mvec[0].mapped, 0, prev_mvec[0].size);
            std::memset(prev_mvec[1].mapped, 0, prev_mvec[1].size);
            std::memset(hidden[0].mapped, 0, hidden[0].size);
            std::memset(hidden[1].mapped, 0, hidden[1].size);
            std::memset(ihist[0].mapped, 0, ihist[0].size);
            std::memset(ihist[1].mapped, 0, ihist[1].size);
        }
        auto jit = jget(pf, "jitter");
        float jx = float(jit.size() > 0 ? jit[0] : 0.0);
        float jy = float(jit.size() > 1 ? jit[1] : 0.0);
        auto jitp = jget(pf, "jitter_prev");
        float jpx = float(jitp.size() > 0 ? jitp[0] : 0.0);
        float jpy = float(jitp.size() > 1 ? jitp[1] : 0.0);

        // Load frame textures
        std::string c_ext = (fmt_col == 0) ? ".f16" : (".c" + std::to_string(fmt_col));
        std::string m_ext = (fmt_mv == 0) ? ".f16" : (".c" + std::to_string(fmt_mv));
        std::string d_ext = (fmt_dep == 0) ? ".f32" : (".c" + std::to_string(fmt_dep));

        auto c_raw = slurp(fdir + "/col" + sfx + c_ext);
        auto m_raw = slurp(fdir + "/mv" + sfx + m_ext);
        auto d_raw = slurp(fdir + "/dep" + sfx + d_ext);

        if (!c_raw.empty()) std::memcpy(in_col.mapped, c_raw.data(), std::min(in_col.size, c_raw.size()));
        if (!m_raw.empty()) std::memcpy(in_mv.mapped, m_raw.data(), std::min(in_mv.size, m_raw.size()));
        if (!d_raw.empty()) std::memcpy(in_dep.mapped, d_raw.data(), std::min(in_dep.size, d_raw.size()));

        if (mode == "rr") {
            auto n_raw = slurp(fdir + "/nrm" + sfx + ".f16");
            if (n_raw.empty()) n_raw = slurp(fdir + "/norm" + sfx + ".f16");
            auto r_raw = slurp(fdir + "/rgh" + sfx + ".f32");
            auto a_raw = slurp(fdir + "/alb" + sfx + ".f16");
            auto s_raw = slurp(fdir + "/spa" + sfx + ".f16");
            if (!n_raw.empty()) std::memcpy(in_norm.mapped, n_raw.data(), std::min(in_norm.size, n_raw.size()));
            if (!r_raw.empty()) std::memcpy(in_rough.mapped, r_raw.data(), std::min(in_rough.size, r_raw.size()));
            if (!a_raw.empty()) std::memcpy(in_alb.mapped, a_raw.data(), std::min(in_alb.size, a_raw.size()));
            if (!s_raw.empty()) std::memcpy(in_spa.mapped, s_raw.data(), std::min(in_spa.size, s_raw.size()));

            float exp_data[4] = { exposure_val, pre_val, 1.0f / std::max(exposure_val, 1e-4f), 0.0f };
            std::memcpy(expo.mapped, exp_data, 16);
        } else {
            uint16_t* expo_ptr = static_cast<uint16_t*>(expo.mapped);
            uint16_t h_exp = f32_to_f16_rne(exposure_val * pre_val);
            expo_ptr[p] = h_exp;
        }

        // Record commands
        vkResetCommandBuffer(cmd, 0);
        vkResetDescriptorPool(dev, dpool, 0);

        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &cbbi);

        vkCmdResetQueryPool(cmd, qpool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);

        if (mode == "sr") {
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
                dispatch("sr_gather", {&in_col, &in_mv, &in_dep, &g_col, &g_mv, &g_dep}, push, 88, (W + 15) / 16, (H + 15) / 16, 1);
            }

            if (mvhi) {
                uint8_t push[32] = {0};
                uint32_t pmv = oW * 4;
                std::memcpy(push + 8, &pmv, 4);
                std::memcpy(push + 12, &fmt_mv, 4);
                std::memcpy(push + 24, &oW, 4);
                std::memcpy(push + 28, &oH, 4);
                dispatch("sr_gather_mv", {&in_mv, &g_mvhi}, push, 32, (oW + 15) / 16, (oH + 15) / 16, 1);
            }

            // 2. Exposure
            {
                uint8_t push[32] = {0};
                std::memcpy(push + 8, &exposure_val, 4);
                std::memcpy(push + 12, &pre_val, 4);
                std::memcpy(push + 24, &p, 4);
                dispatch("sr_exposure", {&expo, &expo}, push, 32, 1, 1, 1);
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
                    dispatch("sr_enc0L_in", {&g_col, &g_dep, mvhi ? &g_mvhi : &g_mv, &prev_mvec[p], &dhist, &hidden[p],
                                             &prev_mvec[p ^ 1], &mv_ero, &mv_nondil, &t_act[0], &w_enc0}, push, 240, (W * H + 63) / 64, 1, 1);
                    dispatch("sr_l0", {&wbuf, &t_act[0], &t_act[1]}, nullptr, 0, 32, 1, 1);
                } else {
                    dispatch("sr_enc0", {&g_col, &g_dep, mvhi ? &g_mvhi : &g_mv, &prev_mvec[p], &dhist, &hidden[p],
                                         &prev_mvec[p ^ 1], &mv_ero, &mv_nondil, &t_act[0], &w_enc0}, push, 240, W0 / 8, H0 / 8, 1);
                }
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
                    dispatch("sr_stage", {&wbuf, &t_act[0], &t_act[1], &t_act[2]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                } else if (st == 1) { dimW = L2w; dimH = L2h; s0 = pe; s1 = 2;
                    std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                    std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                    dispatch("sr_stage", {&wbuf, &t_act[2], &t_act[3], &t_act[4]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                } else if (st <= 7) {
                    Buffer* in_b = (st == 2) ? &t_act[4] : (st & 1) ? &t_act[5] : &t_act[6];
                    Buffer* out_b = (st & 1) ? &t_act[6] : &t_act[5];
                    dimW = L3w; dimH = L3h;
                    if ((st & 1) == 0) { s0 = pe; s1 = 2; } else { s0 = po; s1 = 0; }
                    std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                    std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                    dispatch("sr_tube_split", {&wbuf, in_b, out_b}, push, 56, (dimW + s0 + 3) / 4, (dimH + s1 + 3) / 4, 1);
                } else if (st == 8) { dimW = L2w; dimH = L2h; s0 = po; s1 = 0;
                    std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                    std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                    dispatch("sr_stage", {&wbuf, &t_act[6], &t_act[7], &t_act[7]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                } else { dimW = L1w; dimH = L1h; s0 = po; s1 = 0;
                    std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                    std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                    dispatch("sr_stage", {&wbuf, &t_act[7], &t_act[8], &t_act[8]}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                }
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
                dispatch("sr_dec0", {&t_act[8], &prev_mvec[p ^ 1], &hidden[p], &gates, &factors, &aniso, &hidden[p ^ 1], &w_dec0}, push, 160, W0 / 8, H0 / 8, 1);
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
                dispatch("sr_post", {&g_col, &expo, &gates, &factors, &aniso, &ihist[p], &prev_mvec[p ^ 1], &mv_ero, &mv_nondil,
                                     &ihist[p ^ 1], &full15, &dhist}, push, 256, (W15 + 15) / 16, (hH15 + 15) / 16, 1);
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
                dispatch("sr_downsample", {&full15, &g_col, &expo, &out}, push, 88, (oW + 15) / 16, (oH + 15) / 16, 1);
            }
        } else {
            // RR pipeline
            // 1. Gather
            {
                uint8_t push[8] = {0};
                std::memcpy(push + 0, &W, 4); std::memcpy(push + 4, &H, 4);
                dispatch("rr_gather", {&in_col, &in_mv, &in_dep, &in_norm, &in_alb, &in_spa, &in_rough, &g_col, &g_dep, &g_mv, &out_norm, &out_alb, &out_spa, &out_rough}, push, 8, (W + 15) / 16, (H + 15) / 16, 1);
            }
            // 2. Auto exposure
            {
                uint8_t push[16] = {0};
                std::memcpy(push + 8, &W, 4); std::memcpy(push + 12, &H, 4);
                dispatch("rr_ae_block", {&g_col, &dhist}, push, 16, (W + 15) / 16, (H + 15) / 16, 1);
                uint8_t push_final[8] = {0};
                std::memcpy(push_final + 0, &W, 4); std::memcpy(push_final + 4, &H, 4);
                dispatch("rr_ae_final", {&dhist, &expo}, push_final, 8, 1, 1, 1);
            }
            // 3. Enc0 feat
            {
                uint8_t push[16] = {0};
                std::memcpy(push + 0, &W0, 4); std::memcpy(push + 4, &H0, 4);
                std::memcpy(push + 8, &W, 4); std::memcpy(push + 12, &H, 4);
                dispatch("rr_enc0_feat", {&g_col, &g_mv, &g_dep, &out_norm, &out_alb, &out_spa, &out_rough, &expo,
                                         &t_act[0], &bec, &bf6, &bfa, &bfe}, push, 16, (W0 + 15) / 16, (H0 + 15) / 16, 1);
            }
            // 4. Levels (U-Net)
            {
                uint8_t push[16] = {0};
                std::memcpy(push + 0, &W, 4); std::memcpy(push + 4, &H, 4);
                dispatch("rr_level", {&wbuf, &t_act[0], &t_act[1]}, push, 16, (W + 15) / 16, (H + 15) / 16, 1);
            }
            // 5. Post prep
            {
                uint8_t push[40] = {0};
                std::memcpy(push + 8, &W0, 4);
                std::memcpy(push + 32, &W, 4); std::memcpy(push + 36, &H, 4);
                dispatch("rr_post_prep", {&bec, &gates, &aniso, &cf2}, push, 40, (W + 15) / 16, (H + 15) / 16, 1);
            }
            // 6. Post
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
                dispatch("rr_post", {&cf2, &expo, &cf6, &cfa, &ihist[0], &c0e}, push, 120, (W15 + 15) / 16, (H15 + 15) / 16, 1);
            }
            // 7. Downsample
            {
                uint8_t push[48] = {0};
                std::memcpy(push + 16, &W15, 4); std::memcpy(push + 20, &H15, 4);
                std::memcpy(push + 40, &oW, 4); std::memcpy(push + 44, &oH, 4);
                dispatch("rr_downsample", {&cf6, &cfa, &expo, &out}, push, 48, (oW + 15) / 16, (oH + 15) / 16, 1);
            }
        }

        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        uint64_t ts[2] = {0, 0};
        vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        float ms = float(ts[1] - ts[0]) * picked_props.limits.timestampPeriod / 1e6f;

        // Write output frame
        std::string out_path = odir + "/out_" + std::to_string(fi) + ".f32";
        std::ofstream fout(out_path, std::ios::binary);
        if (fout) {
            fout.write(static_cast<const char*>(out.mapped), size_t(oW) * oH * 16);
        }

        std::printf("frame %d done (parity %d)\n", fi, p);
        std::printf("GPU_FRAME_MS %d %.4f\n", fi, ms);
        std::fflush(stdout);

        p ^= 1;
        fc = (fc + 1) & 7;
    }

    // Cleanup
    vkDeviceWaitIdle(dev);
    destroy_buffer(dev, in_col);
    destroy_buffer(dev, in_dep);
    destroy_buffer(dev, in_mv);
    destroy_buffer(dev, in_norm);
    destroy_buffer(dev, in_rough);
    destroy_buffer(dev, in_alb);
    destroy_buffer(dev, in_spa);
    destroy_buffer(dev, out_alb);
    destroy_buffer(dev, out_spa);
    destroy_buffer(dev, out_norm);
    destroy_buffer(dev, out_rough);
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
    destroy_buffer(dev, dhist);
    destroy_buffer(dev, full15);
    destroy_buffer(dev, cf2);
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

    for (auto& pair : pipelines) {
        vkDestroyPipeline(dev, pair.second.pipeline, nullptr);
        vkDestroyPipelineLayout(dev, pair.second.layout, nullptr);
        vkDestroyDescriptorSetLayout(dev, pair.second.dsl, nullptr);
    }

    vkDestroyDescriptorPool(dev, dpool, nullptr);
    vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);

    return 0;
}
