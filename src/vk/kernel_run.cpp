#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

struct ImageSpec {
    bool set = false;                 // the binding is an image, not a buffer
    std::string format = "r16g16b16a16_sfloat";
    uint32_t width = 0, height = 0;
    bool sampled = false;             // sampled (linear filter) vs storage image
};

struct BindingSpec {
    uint32_t binding = 0;
    std::string file;
    size_t bytes = 0;
    std::string output;
    std::string type = "u8";
    ImageSpec img;
};

struct LaunchConfig {
    uint32_t groups[3] = {1, 1, 1};
    std::vector<uint8_t> push_data;
    std::vector<BindingSpec> bindings;
};

// Simple JSON parser for launch.json
static void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++;
}

static std::string parse_string(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"') return "";
    i++;
    std::string res;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            i++;
            res += s[i];
        } else {
            res += s[i];
        }
        i++;
    }
    if (i < s.size() && s[i] == '"') i++;
    return res;
}

static uint64_t parse_uint(const std::string& s, size_t& i) {
    skip_ws(s, i);
    uint64_t v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + (s[i] - '0');
        i++;
    }
    return v;
}

static LaunchConfig parse_launch_json(const std::string& json_str) {
    LaunchConfig cfg;
    size_t i = 0;
    while (i < json_str.size()) {
        skip_ws(json_str, i);
        if (i >= json_str.size()) break;
        if (json_str[i] == '"') {
            std::string key = parse_string(json_str, i);
            skip_ws(json_str, i);
            if (i < json_str.size() && json_str[i] == ':') i++;
            skip_ws(json_str, i);
            if (key == "groups") {
                if (json_str[i] == '[') {
                    i++;
                    cfg.groups[0] = (uint32_t)parse_uint(json_str, i);
                    skip_ws(json_str, i); if (i < json_str.size() && json_str[i] == ',') i++;
                    cfg.groups[1] = (uint32_t)parse_uint(json_str, i);
                    skip_ws(json_str, i); if (i < json_str.size() && json_str[i] == ',') i++;
                    cfg.groups[2] = (uint32_t)parse_uint(json_str, i);
                    while (i < json_str.size() && json_str[i] != ']') i++;
                    if (i < json_str.size() && json_str[i] == ']') i++;
                }
            } else if (key == "push") {
                std::string hex = parse_string(json_str, i);
                cfg.push_data.clear();
                for (size_t h = 0; h + 1 < hex.size(); h += 2) {
                    char buf[3] = { hex[h], hex[h+1], 0 };
                    cfg.push_data.push_back((uint8_t)strtoul(buf, nullptr, 16));
                }
            } else if (key == "bindings") {
                if (json_str[i] == '[') {
                    i++;
                    while (i < json_str.size() && json_str[i] != ']') {
                        skip_ws(json_str, i);
                        if (json_str[i] == '{') {
                            i++;
                            BindingSpec b;
                            while (i < json_str.size() && json_str[i] != '}') {
                                skip_ws(json_str, i);
                                if (json_str[i] == '"') {
                                    std::string bkey = parse_string(json_str, i);
                                    skip_ws(json_str, i);
                                    if (i < json_str.size() && json_str[i] == ':') i++;
                                    skip_ws(json_str, i);
                                    if (bkey == "binding") b.binding = (uint32_t)parse_uint(json_str, i);
                                    else if (bkey == "file") b.file = parse_string(json_str, i);
                                    else if (bkey == "bytes") b.bytes = (size_t)parse_uint(json_str, i);
                                    else if (bkey == "output") b.output = parse_string(json_str, i);
                                    else if (bkey == "type") b.type = parse_string(json_str, i);
                                    else if (bkey == "image") {
                                        // {"format": "...", "width": W, "height": H, "file": "...",
                                        //  "sampled": true|false, "output": "name"} - an image
                                        // binding instead of a buffer one. "file" seeds it (upload)
                                        // and "output" names the readback, both as raw texel bytes.
                                        b.img.set = true;
                                        if (json_str[i] == '{') {
                                            i++;
                                            while (i < json_str.size() && json_str[i] != '}') {
                                                skip_ws(json_str, i);
                                                if (json_str[i] == '"') {
                                                    std::string ikey = parse_string(json_str, i);
                                                    skip_ws(json_str, i);
                                                    if (i < json_str.size() && json_str[i] == ':') i++;
                                                    skip_ws(json_str, i);
                                                    if (ikey == "format") b.img.format = parse_string(json_str, i);
                                                    else if (ikey == "width") b.img.width = (uint32_t)parse_uint(json_str, i);
                                                    else if (ikey == "height") b.img.height = (uint32_t)parse_uint(json_str, i);
                                                    else if (ikey == "file") b.file = parse_string(json_str, i);
                                                    else if (ikey == "output") b.output = parse_string(json_str, i);
                                                    else if (ikey == "sampled") {
                                                        if (json_str.compare(i, 4, "true") == 0) { b.img.sampled = true; i += 4; }
                                                        else if (json_str.compare(i, 5, "false") == 0) { i += 5; }
                                                        else parse_uint(json_str, i);
                                                    } else if (json_str[i] == '"') parse_string(json_str, i);
                                                    else if (json_str[i] >= '0' && json_str[i] <= '9') parse_uint(json_str, i);
                                                    else i++;
                                                } else {
                                                    i++;
                                                }
                                                skip_ws(json_str, i);
                                                if (i < json_str.size() && json_str[i] == ',') i++;
                                            }
                                            if (i < json_str.size() && json_str[i] == '}') i++;
                                        }
                                    }
                                    else {
                                        // skip value
                                        if (json_str[i] == '"') parse_string(json_str, i);
                                        else parse_uint(json_str, i);
                                    }
                                } else {
                                    i++;
                                }
                                skip_ws(json_str, i);
                                if (i < json_str.size() && json_str[i] == ',') i++;
                            }
                            if (i < json_str.size() && json_str[i] == '}') i++;
                            cfg.bindings.push_back(b);
                        } else {
                            i++;
                        }
                        skip_ws(json_str, i);
                        if (i < json_str.size() && json_str[i] == ',') i++;
                    }
                    if (i < json_str.size() && json_str[i] == ']') i++;
                }
            } else {
                // skip unknown value
                if (json_str[i] == '"') parse_string(json_str, i);
                else if (json_str[i] >= '0' && json_str[i] <= '9') parse_uint(json_str, i);
                else i++;
            }
        } else {
            i++;
        }
    }
    return cfg;
}

static VkFormat format_from_string(const std::string& s) {
    if (s == "r16g16b16a16_sfloat") return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (s == "r16g16b16a16_uint") return VK_FORMAT_R16G16B16A16_UINT;
    if (s == "r32g32b32a32_sfloat") return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (s == "r32g32_sfloat") return VK_FORMAT_R32G32_SFLOAT;
    if (s == "r8g8b8a8_unorm") return VK_FORMAT_R8G8B8A8_UNORM;
    return VK_FORMAT_UNDEFINED;
}

static uint32_t format_texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    case VK_FORMAT_R16G16B16A16_UINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R8G8B8A8_UNORM: return 4;
    default: return 0;
    }
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            return i;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string spv_path, launch_path, out_dir;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--spv") && i + 1 < argc) spv_path = argv[++i];
        else if (!strcmp(argv[i], "--launch") && i + 1 < argc) launch_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out_dir = argv[++i];
    }
    if (spv_path.empty() || launch_path.empty() || out_dir.empty()) {
        std::fprintf(stderr, "Usage: kernel_run --spv <spv> --launch <launch.json> --out <outdir>\n");
        return 1;
    }

    std::string launch_dir = ".";
    size_t last_slash = launch_path.find_last_of('/');
    if (last_slash != std::string::npos) launch_dir = launch_path.substr(0, last_slash);

    std::ifstream jf(launch_path);
    if (!jf.is_open()) {
        std::fprintf(stderr, "Failed to open launch file: %s\n", launch_path.c_str());
        return 1;
    }
    std::string json_str((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());
    LaunchConfig cfg = parse_launch_json(json_str);

    // Read SPV
    std::ifstream sf(spv_path, std::ios::binary | std::ios::ate);
    if (!sf.is_open()) {
        std::fprintf(stderr, "Failed to open spv file: %s\n", spv_path.c_str());
        return 1;
    }
    size_t spv_bytes = (size_t)sf.tellg();
    sf.seekg(0, std::ios::beg);
    std::vector<uint32_t> spv(spv_bytes / 4);
    sf.read(reinterpret_cast<char*>(spv.data()), spv_bytes);

    // Vulkan Instance
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "kernel_run";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateInstance failed\n");
        return 1;
    }

    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(instance, &dev_count, nullptr);
    if (dev_count == 0) {
        std::fprintf(stderr, "No Vulkan devices found\n");
        return 1;
    }
    std::vector<VkPhysicalDevice> devices(dev_count);
    vkEnumeratePhysicalDevices(instance, &dev_count, devices.data());

    VkPhysicalDevice phys = devices[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    for (uint32_t i = 0; i < dev_count; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devices[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            phys = devices[i];
            props = p;
            break;
        }
    }
    std::printf("VK_DEVICE: %s\n", props.deviceName);

    // Does this device support pinning a subgroup size at all? (Needed for sr_stage, whose
    // cooperative-matrix path computes garbage at subgroup size 64.) Query it here so the feature
    // below is only enabled when it exists - enabling it blindly fails vkCreateDevice on drivers
    // without it (that is how the offline check's lavapipe run used to segfault).
    VkPhysicalDeviceVulkan13Features q13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 qf2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    qf2.pNext = &q13;
    vkGetPhysicalDeviceFeatures2(phys, &qf2);
    const bool have_subgroup_size_control = q13.subgroupSizeControl == VK_TRUE;

    // Extensions & features
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> avail_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_count, avail_exts.data());
    auto has_ext = [&](const char* name) {
        for (const auto& e : avail_exts) if (!strcmp(e.extensionName, name)) return true;
        return false;
    };

    std::vector<const char*> enabled_exts;
    if (has_ext(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) enabled_exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    if (has_ext(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) enabled_exts.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    if (has_ext(VK_KHR_16BIT_STORAGE_EXTENSION_NAME)) enabled_exts.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
    if (has_ext(VK_KHR_8BIT_STORAGE_EXTENSION_NAME)) enabled_exts.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
    if (has_ext(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME)) enabled_exts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    if (has_ext(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) enabled_exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    if (has_ext(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME)) enabled_exts.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceShaderAtomicInt64Features atom64{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

    void** next_ptr = &feat2.pNext;
    if (has_ext(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
        coop.cooperativeMatrix = VK_TRUE;
        *next_ptr = &coop; next_ptr = &coop.pNext;
    }
    if (has_ext(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME)) {
        fp8.shaderFloat8 = VK_TRUE;
        fp8.shaderFloat8CooperativeMatrix = VK_TRUE;
        *next_ptr = &fp8; next_ptr = &fp8.pNext;
    }
    if (has_ext(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME)) {
        atom64.shaderBufferInt64Atomics = VK_TRUE;
        *next_ptr = &atom64; next_ptr = &atom64.pNext;
    }
    f11.storageBuffer16BitAccess = VK_TRUE;
    *next_ptr = &f11; next_ptr = &f11.pNext;

    f12.storageBuffer8BitAccess = VK_TRUE;
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.vulkanMemoryModel = VK_TRUE;
    *next_ptr = &f12; next_ptr = &f12.pNext;

    f13.synchronization2 = VK_TRUE;
    f13.subgroupSizeControl = have_subgroup_size_control ? VK_TRUE : VK_FALSE;
    *next_ptr = &f13; next_ptr = &f13.pNext;

    uint32_t qfamily = 0;
    uint32_t qcount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qcount, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(qcount);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qcount, qprops.data());
    for (uint32_t i = 0; i < qcount; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfamily = i; break; }
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qfamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &feat2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enabled_exts.size();
    dci.ppEnabledExtensionNames = enabled_exts.data();

    VkDevice device;
    if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
        // Retry with basic features if extended features fail (e.g. lavapipe without float8)
        feat2.pNext = nullptr;
        dci.pNext = &feat2;
        dci.enabledExtensionCount = 0;
        if (vkCreateDevice(phys, &dci, nullptr, &device) != VK_SUCCESS) {
            std::fprintf(stderr, "vkCreateDevice failed\n");
            return 1;
        }
    }

    VkQueue queue;
    vkGetDeviceQueue(device, qfamily, 0, &queue);

    // Shader Module
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv_bytes;
    smci.pCode = spv.data();
    VkShaderModule shader;
    if (vkCreateShaderModule(device, &smci, nullptr, &shader) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateShaderModule failed\n");
        return 1;
    }

    // Descriptor set layout
    std::vector<VkDescriptorSetLayoutBinding> dsl_bindings;
    for (const auto& b : cfg.bindings) {
        VkDescriptorSetLayoutBinding dslb{};
        dslb.binding = b.binding;
        if (b.img.set) {
            dslb.descriptorType = b.img.sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                                : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } else {
            dslb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        dslb.descriptorCount = 1;
        dslb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        dsl_bindings.push_back(dslb);
    }

    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = (uint32_t)dsl_bindings.size();
    dslci.pBindings = dsl_bindings.data();
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl);

    // Pipeline Layout
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    uint32_t push_size = (uint32_t)cfg.push_data.size();
    if (push_size % 4 != 0) push_size += 4 - (push_size % 4);
    if (push_size < 256) push_size = 256;
    pcr.size = push_size;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeline_layout;
    vkCreatePipelineLayout(device, &plci, nullptr, &pipeline_layout);

    // Pipeline
    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.layout = pipeline_layout;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = "main";
    // Pin the subgroup size to 32 whenever the workgroup has at least 32 threads: this driver's
    // cooperative-matrix path computes garbage at subgroup size 64 (k_sr_stage; measured both ways,
    // build/agents/stage/progress.md), and the reference runtime pins wave32 for every pipeline.
    uint32_t lsx = 0;
    for (size_t i = 0; i + 5 < spv.size(); ++i) {
        // OpExecutionMode (16), word count 6, mode LocalSize (17): {6, op, 16, entry, 17, x, y, z}
        if ((spv[i] >> 16) == 6u && (spv[i] & 0xffffu) == 16u && spv[i + 2] == 17u) { lsx = spv[i + 3]; break; }
    }
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (lsx >= 32u) {
        if (!have_subgroup_size_control) {
            std::fprintf(stderr,
                "kernel_run: this shader needs subgroup size 32 (its cooperative-matrix path is "
                "wrong at 64) but the device does not support subgroupSizeControl - refusing to run it\n");
            return 1;
        }
        req.requiredSubgroupSize = 32;
        cpci.stage.pNext = &req;
    }

    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS) {
        std::fprintf(stderr, "vkCreateComputePipelines failed\n");
        return 1;
    }

    // Descriptor pool
    VkDescriptorPoolSize pool_sizes[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, std::max(1u, (uint32_t)cfg.bindings.size())},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, std::max(1u, (uint32_t)cfg.bindings.size())},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, std::max(1u, (uint32_t)cfg.bindings.size())},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = pool_sizes;
    VkDescriptorPool pool;
    vkCreateDescriptorPool(device, &dpci, nullptr, &pool);

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = pool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts = &dsl;
    VkDescriptorSet desc_set;
    vkAllocateDescriptorSets(device, &dsai, &desc_set);

    struct BufferResource {
        VkBuffer buffer;
        VkDeviceMemory memory;
        void* mapped;
        size_t size;
        BindingSpec spec;
    };
    std::vector<BufferResource> buffers;

    for (const auto& b : cfg.bindings) {
        if (b.img.set) continue;                    // images are handled below
        std::vector<uint8_t> file_bytes;
        if (!b.file.empty()) {
            std::string path = (!b.file.empty() && b.file[0] == '/') ? b.file : (launch_dir + "/" + b.file);
            std::ifstream bf(path, std::ios::binary);
            if (bf.is_open()) {
                file_bytes.assign((std::istreambuf_iterator<char>(bf)), std::istreambuf_iterator<char>());
            } else {
                std::fprintf(stderr, "Warning: failed to open input binding file: %s\n", path.c_str());
            }
        }
        size_t alloc_size = std::max(b.bytes, file_bytes.size());
        if (alloc_size == 0) alloc_size = 256;
        if (alloc_size % 256 != 0) alloc_size += 256 - (alloc_size % 256);

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = alloc_size;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buf;
        vkCreateBuffer(device, &bci, nullptr, &buf);

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device, buf, &memReq);

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = memReq.size;
        mai.memoryTypeIndex = find_memory_type(phys, memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory mem;
        vkAllocateMemory(device, &mai, nullptr, &mem);
        vkBindBufferMemory(device, buf, mem, 0);

        void* ptr = nullptr;
        vkMapMemory(device, mem, 0, alloc_size, 0, &ptr);
        std::memset(ptr, 0, alloc_size);
        if (!file_bytes.empty()) {
            std::memcpy(ptr, file_bytes.data(), file_bytes.size());
        }

        BufferResource res;
        res.buffer = buf;
        res.memory = mem;
        res.mapped = ptr;
        res.size = alloc_size;
        res.spec = b;
        buffers.push_back(res);

        VkDescriptorBufferInfo dbi{};
        dbi.buffer = buf;
        dbi.offset = 0;
        dbi.range = alloc_size;

        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wds.dstSet = desc_set;
        wds.dstBinding = b.binding;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);
    }

    struct ImageResource {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        VkBuffer up = VK_NULL_HANDLE;            // upload staging (host visible)
        VkDeviceMemory up_mem = VK_NULL_HANDLE;
        void* up_ptr = nullptr;
        size_t up_bytes = 0;
        VkBuffer dn = VK_NULL_HANDLE;            // readback staging
        VkDeviceMemory dn_mem = VK_NULL_HANDLE;
        void* dn_ptr = nullptr;
        size_t dn_bytes = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        uint32_t texel = 0;
        BindingSpec spec;
    };
    std::vector<ImageResource> images;

    for (const auto& b : cfg.bindings) {
        if (!b.img.set) continue;
        ImageResource im;
        im.spec = b;
        im.format = format_from_string(b.img.format);
        im.texel = format_texel_bytes(im.format);
        if (im.format == VK_FORMAT_UNDEFINED || im.texel == 0 || b.img.width == 0 || b.img.height == 0) {
            std::fprintf(stderr, "kernel_run: image binding %u has an unsupported format/size (%s, %ux%u)\n",
                         b.binding, b.img.format.c_str(), b.img.width, b.img.height);
            return 1;
        }
        const size_t nbytes = size_t(b.img.width) * b.img.height * im.texel;

        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = im.format;
        ici.extent = {b.img.width, b.img.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &ici, nullptr, &im.image) != VK_SUCCESS) {
            std::fprintf(stderr, "kernel_run: vkCreateImage failed for binding %u\n", b.binding);
            return 1;
        }
        VkMemoryRequirements mreq;
        vkGetImageMemoryRequirements(device, im.image, &mreq);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mreq.size;
        mai.memoryTypeIndex = find_memory_type(phys, mreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &mai, nullptr, &im.memory) != VK_SUCCESS) {
            std::fprintf(stderr, "kernel_run: image memory allocation failed for binding %u\n", b.binding);
            return 1;
        }
        vkBindImageMemory(device, im.image, im.memory, 0);

        VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        ivci.image = im.image;
        ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ivci.format = im.format;
        ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCreateImageView(device, &ivci, nullptr, &im.view);

        if (b.img.sampled) {
            VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            // integer formats are not filterable: NEAREST is the only legal choice (and the only
            // one this kernel uses - its CR bilinear is computed in the shader)
            const bool ifmt = (im.format == VK_FORMAT_R16G16B16A16_UINT ||
                               im.format == VK_FORMAT_R32_UINT || im.format == VK_FORMAT_R32_SINT);
            sci.magFilter = ifmt ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            sci.minFilter = ifmt ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.minLod = 0.0f;
            sci.maxLod = 0.0f;
            vkCreateSampler(device, &sci, nullptr, &im.sampler);
        }

        // staging: upload (if a file seeds the image) and/or readback (if it is an output)
        auto make_stage = [&](size_t sz, VkBuffer& buf, VkDeviceMemory& mem, void** ptr) {
            VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bci.size = sz;
            bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            vkCreateBuffer(device, &bci, nullptr, &buf);
            VkMemoryRequirements r;
            vkGetBufferMemoryRequirements(device, buf, &r);
            VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            a.allocationSize = r.size;
            a.memoryTypeIndex = find_memory_type(phys, r.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            vkAllocateMemory(device, &a, nullptr, &mem);
            vkBindBufferMemory(device, buf, mem, 0);
            vkMapMemory(device, mem, 0, sz, 0, ptr);
            std::memset(*ptr, 0, sz);
        };
        if (!b.file.empty()) {
            std::string path = (b.file[0] == '/') ? b.file : (launch_dir + "/" + b.file);
            std::ifstream bf(path, std::ios::binary);
            std::vector<uint8_t> fb;
            if (bf.is_open()) fb.assign((std::istreambuf_iterator<char>(bf)), std::istreambuf_iterator<char>());
            if (fb.empty()) std::fprintf(stderr, "kernel_run: image binding %u: cannot read %s\n", b.binding, path.c_str());
            if (fb.size() != nbytes)
                std::fprintf(stderr, "kernel_run: image binding %u: %s is %zu bytes, image is %zu\n",
                             b.binding, path.c_str(), fb.size(), nbytes);
            im.up_bytes = nbytes;
            make_stage(nbytes, im.up, im.up_mem, &im.up_ptr);
            if (!fb.empty()) std::memcpy(im.up_ptr, fb.data(), std::min(fb.size(), nbytes));
        }
        if (!b.output.empty()) {
            im.dn_bytes = nbytes;
            make_stage(nbytes, im.dn, im.dn_mem, &im.dn_ptr);
        }

        VkDescriptorImageInfo dii{};
        dii.imageView = im.view;
        dii.imageLayout = b.img.sampled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                        : VK_IMAGE_LAYOUT_GENERAL;
        dii.sampler = im.sampler;
        VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wds.dstSet = desc_set;
        wds.dstBinding = b.binding;
        wds.descriptorCount = 1;
        wds.descriptorType = b.img.sampled ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                                           : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        wds.pImageInfo = &dii;
        vkUpdateDescriptorSets(device, 1, &wds, 0, nullptr);

        images.push_back(im);
    }

    // Command pool and buffer
    VkCommandPoolCreateInfo cpci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci2.queueFamilyIndex = qfamily;
    VkCommandPool cmd_pool;
    vkCreateCommandPool(device, &cpci2, nullptr, &cmd_pool);

    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmd_pool;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cbai, &cmd);

    VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &cbbi);

    // Image setup: UNDEFINED -> TRANSFER_DST, upload, then -> the layout the shader uses. Images
    // with no seed file still need the transition out of UNDEFINED.
    for (auto& im : images) {
        const bool samp = im.spec.img.sampled;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = 0;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (im.up) {
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
            VkBufferImageCopy cic{};
            cic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            cic.imageExtent = {im.spec.img.width, im.spec.img.height, 1};
            vkCmdCopyBufferToImage(cmd, im.up, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cic);
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = samp ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = samp ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        } else {
            b.dstAccessMask = samp ? VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = samp ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        }
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1, &desc_set, 0, nullptr);
    if (!cfg.push_data.empty()) {
        vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)cfg.push_data.size(), cfg.push_data.data());
    }
    vkCmdDispatch(cmd, cfg.groups[0], cfg.groups[1], cfg.groups[2]);

    // Storage-image readback: GENERAL -> TRANSFER_SRC, copy out
    for (auto& im : images) {
        if (!im.dn) continue;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im.image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy cic{};
        cic.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        cic.imageExtent = {im.spec.img.width, im.spec.img.height, 1};
        vkCmdCopyImageToBuffer(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im.dn, 1, &cic);
    }
    vkEndCommandBuffer(cmd);

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(device, &fci, nullptr, &fence);

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);

    // Write outputs
    mkdir(out_dir.c_str(), 0777);
    for (auto& im : images) {                       // storage-image readbacks: raw texel bytes
        if (im.spec.output.empty() || !im.dn_ptr) continue;
        std::string out_path = out_dir + "/" + im.spec.output + ".bin";
        std::ofstream out_f(out_path, std::ios::binary);
        if (!out_f.is_open()) {
            std::fprintf(stderr, "Failed to open output file: %s\n", out_path.c_str());
        } else {
            out_f.write(reinterpret_cast<const char*>(im.dn_ptr), (std::streamsize)im.dn_bytes);
        }
    }
    for (const auto& res : buffers) {
        if (!res.spec.output.empty()) {
            std::string out_path = out_dir + "/" + res.spec.output + ".bin";
            std::ofstream out_f(out_path, std::ios::binary);
            if (!out_f.is_open()) {
                std::fprintf(stderr, "Failed to open output file: %s\n", out_path.c_str());
            } else {
                size_t write_bytes = res.spec.bytes > 0 ? res.spec.bytes : res.size;
                out_f.write(reinterpret_cast<const char*>(res.mapped), write_bytes);
            }
        }
        vkUnmapMemory(device, res.memory);
        vkDestroyBuffer(device, res.buffer, nullptr);
        vkFreeMemory(device, res.memory, nullptr);
    }

    for (auto& im : images) {
        if (im.up_ptr) vkUnmapMemory(device, im.up_mem);
        if (im.dn_ptr) vkUnmapMemory(device, im.dn_mem);
        if (im.up) vkDestroyBuffer(device, im.up, nullptr);
        if (im.dn) vkDestroyBuffer(device, im.dn, nullptr);
        if (im.up_mem) vkFreeMemory(device, im.up_mem, nullptr);
        if (im.dn_mem) vkFreeMemory(device, im.dn_mem, nullptr);
        if (im.sampler) vkDestroySampler(device, im.sampler, nullptr);
        if (im.view) vkDestroyImageView(device, im.view, nullptr);
        if (im.image) vkDestroyImage(device, im.image, nullptr);
        if (im.memory) vkFreeMemory(device, im.memory, nullptr);
    }

    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmd_pool, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    vkDestroyShaderModule(device, shader, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
