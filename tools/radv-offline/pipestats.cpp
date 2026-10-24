// Compile compute SPIR-V with RADV/ACO for a GPU that is not present and print ACO's statistics.
//
//   RADV_FORCE_FAMILY=gfx1201 VK_ICD_FILENAMES=<mesa>/share/vulkan/icd.d/radeon_icd.x86_64.json \
//       pipestats [--isa DIR] a.spv b.spv ...
//
// RADV_FORCE_FAMILY makes RADV create a null device: pipelines compile exactly as on the real card
// (same NIR, same ACO), nothing can execute. Output, one line per file:
//   <file> vgprs <n> sgprs <n> spill_vgprs <n> spill_sgprs <n> lds <bytes> waves <n> insts <n>
// --isa DIR also writes each pipeline's disassembly to DIR/<file>.s.
//
// Pipelines are created the way the reference runtime creates them (linux/src/core/nrvk.hpp): wave32
// required, one descriptor set, one push-constant range. The set layout is read from the SPIR-V itself.
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::fprintf(stderr, "%s:%d %s = %d\n", __FILE__, __LINE__, #x, r_); std::exit(1); } } while (0)

static std::vector<uint32_t> read_spirv(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path); std::exit(1); }
    std::vector<uint32_t> w(size_t(f.tellg()) / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(w.data()), w.size() * 4);
    return w;
}

// Descriptor bindings from OpDecorate Binding / DescriptorSet and the variables' types.
static std::vector<VkDescriptorSetLayoutBinding> reflect(const std::vector<uint32_t>& w, bool& has_push) {
    enum { OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27, OpTypeArray = 28,
           OpTypeRuntimeArray = 29, OpTypePointer = 32, OpTypeAccelerationStructure = 5341,
           OpVariable = 59, OpDecorate = 71 };
    std::map<uint32_t, uint32_t> binding, set, ptr_to, type_op, image_sampled;
    std::map<uint32_t, uint32_t> array_of;
    std::map<uint32_t, bool> block, buffer_block;
    std::vector<std::pair<uint32_t, uint32_t>> vars;  // (id, storage class)
    std::map<uint32_t, uint32_t> var_type;
    for (size_t i = 5; i < w.size();) {
        const uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
        if (n == 0) break;
        const uint32_t* a = &w[i + 1];
        switch (op) {
        case OpDecorate:
            if (a[1] == 33) binding[a[0]] = a[2];
            if (a[1] == 34) set[a[0]] = a[2];
            if (a[1] == 2) block[a[0]] = true;
            if (a[1] == 3) buffer_block[a[0]] = true;
            break;
        case OpTypeImage: type_op[a[0]] = op; image_sampled[a[0]] = a[6]; break;
        case OpTypeSampler: case OpTypeSampledImage: case OpTypeAccelerationStructure:
            type_op[a[0]] = op; break;
        case OpTypeArray: case OpTypeRuntimeArray: type_op[a[0]] = op; array_of[a[0]] = a[1]; break;
        case OpTypePointer: ptr_to[a[0]] = a[2]; break;
        case OpVariable: vars.push_back({a[1], a[2]}); var_type[a[1]] = a[0]; break;
        }
        i += n;
    }
    std::vector<VkDescriptorSetLayoutBinding> out;
    has_push = false;
    for (auto [id, sc] : vars) {
        if (sc == 9) { has_push = true; continue; }
        if (!binding.count(id)) continue;
        if (set.count(id) && set[id] != 0) { std::fprintf(stderr, "descriptor set %u not supported\n", set[id]); std::exit(1); }
        uint32_t t = ptr_to[var_type[id]];
        uint32_t count = 1;
        while (type_op.count(t) && (type_op[t] == OpTypeArray || type_op[t] == OpTypeRuntimeArray)) t = array_of[t];
        VkDescriptorType dt;
        if (sc == 12) dt = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        else if (sc == 2) dt = buffer_block.count(t) ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else if (type_op[t] == OpTypeSampledImage) dt = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        else if (type_op[t] == OpTypeSampler) dt = VK_DESCRIPTOR_TYPE_SAMPLER;
        else if (type_op[t] == OpTypeImage) dt = image_sampled[t] == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        else continue;
        // Several variables may alias one binding (the reference binds an arena under several types).
        bool dup = false;
        for (auto& b : out) dup |= b.binding == binding[id];
        if (!dup) out.push_back({binding[id], dt, count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    }
    return out;
}

int main(int argc, char** argv) {
    const char* isa_dir = nullptr;
    std::vector<const char*> files;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--isa") && i + 1 < argc) isa_dir = argv[++i];
        else files.push_back(argv[i]);
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_4;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst;
    CHECK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
    std::vector<VkPhysicalDevice> pds(n);
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds.data()));
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    for (auto p : pds) {
        vkGetPhysicalDeviceProperties(p, &props);
        if (props.vendorID == 0x1002) { pd = p; break; }
    }
    if (!pd) { std::fprintf(stderr, "no AMD device (set RADV_FORCE_FAMILY and VK_ICD_FILENAMES)\n"); return 1; }
    std::fprintf(stderr, "device: %s (%s)\n", props.deviceName, props.driverVersion ? "radv" : "?");

    // Enable everything the device has: the shaders need FP8, cooperative matrices, 8/16-bit storage.
    VkPhysicalDeviceShaderFloat8FeaturesEXT f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR, &f8};
    VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pe{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, &cm};
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR, &pe};
    VkPhysicalDeviceVulkan14Features v14{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES, &bf};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &v14};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &v13};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &v12};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &v11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!cm.cooperativeMatrix || !f8.shaderFloat8CooperativeMatrix)
        std::fprintf(stderr, "warning: cooperativeMatrix %d shaderFloat8CooperativeMatrix %d\n",
                     cm.cooperativeMatrix, f8.shaderFloat8CooperativeMatrix);
    const char* exts[] = {VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME,
                          VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME};
    float prio = 1;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 4; dci.ppEnabledExtensionNames = exts;
    VkDevice dev;
    CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
    auto getStats = reinterpret_cast<PFN_vkGetPipelineExecutableStatisticsKHR>(vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableStatisticsKHR"));
    auto getIR = reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(vkGetDeviceProcAddr(dev, "vkGetPipelineExecutableInternalRepresentationsKHR"));

    int failed = 0;
    for (const char* path : files) {
        const auto code = read_spirv(path);
        bool has_push = false;
        auto lb = reflect(code, has_push);
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = uint32_t(lb.size()); li.pBindings = lb.data();
        VkDescriptorSetLayout dsl;
        CHECK(vkCreateDescriptorSetLayout(dev, &li, nullptr, &dsl));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, props.limits.maxPushConstantsSize};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &dsl;
        pli.pushConstantRangeCount = has_push ? 1 : 0; pli.pPushConstantRanges = &pcr;
        VkPipelineLayout layout;
        CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &layout));
        VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smi.codeSize = code.size() * 4; smi.pCode = code.data();
        VkShaderModule mod;
        CHECK(vkCreateShaderModule(dev, &smi, nullptr, &mod));
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
        req.requiredSubgroupSize = 32;
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                    (isa_dir ? VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR : 0);
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, &req};
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = mod; cpi.stage.pName = "main";
        cpi.layout = layout;
        VkPipeline pipe;
        VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe);
        const char* base = std::strrchr(path, '/') ? std::strrchr(path, '/') + 1 : path;
        if (r != VK_SUCCESS) { std::printf("%s FAILED %d\n", base, r); ++failed; continue; }
        VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR};
        ei.pipeline = pipe; ei.executableIndex = 0;
        uint32_t ns = 0;
        getStats(dev, &ei, &ns, nullptr);
        std::vector<VkPipelineExecutableStatisticKHR> st(ns, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
        getStats(dev, &ei, &ns, st.data());
        std::printf("%s", base);
        for (auto& s : st) {
            std::string name = s.name;
            for (auto& c : name) c = c == ' ' ? '_' : char(std::tolower(c));
            if (s.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) std::printf(" %s %llu", name.c_str(), (unsigned long long)s.value.u64);
        }
        std::printf("\n");
        if (isa_dir) {
            uint32_t nr = 0;
            getIR(dev, &ei, &nr, nullptr);
            std::vector<VkPipelineExecutableInternalRepresentationKHR> ir(nr, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR});
            getIR(dev, &ei, &nr, ir.data());
            std::vector<std::vector<char>> bufs(nr);
            for (uint32_t i = 0; i < nr; ++i) { bufs[i].resize(ir[i].dataSize); ir[i].pData = bufs[i].data(); }
            getIR(dev, &ei, &nr, ir.data());
            for (uint32_t i = 0; i < nr; ++i)
                if (std::strstr(ir[i].name, "Assembly")) {
                    std::ofstream o(std::string(isa_dir) + "/" + base + ".s");
                    o.write(bufs[i].data(), std::strlen(bufs[i].data()));
                }
        }
        vkDestroyPipeline(dev, pipe, nullptr);
        vkDestroyShaderModule(dev, mod, nullptr);
        vkDestroyPipelineLayout(dev, layout, nullptr);
        vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
    }
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return failed ? 1 : 0;
}
