// hwprobe: what the card and its driver actually report, on the machine the port targets.
//
// The cloud container the port was written in has no GPU; this is the first time the device gate of
// docs/inventory/KERNELS.md section 3.5 is answered by a real RDNA4 card instead of a drm-shim.
// It prints the device, the driver, the subgroup sizes, the feature bits the network needs, and
// every cooperative-matrix configuration the driver reports (the FP8 16x16x16 subgroup entry is the
// one the whole port depends on).
//
// Build: tools/hwtest/build_probe.sh   (podman container, binary runs on the host)

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { std::printf("FAILED %s -> %d\n", #x, int(r_)); return 1; } } while (0)

static std::string component_name(VkComponentTypeKHR t) {
    switch (t) {
        case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
        case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
        case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
        case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
        case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
        case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
        case VK_COMPONENT_TYPE_SINT64_KHR: return "s64";
        case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
        case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
        case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
        case VK_COMPONENT_TYPE_UINT64_KHR: return "u64";
        case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT: return "e4m3";
        case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT: return "e5m2";
        // VK_COMPONENT_TYPE_BFLOAT16_KHR (VK_KHR_shader_bfloat16); not in every header revision.
        case static_cast<VkComponentTypeKHR>(1000141000): return "bf16";
        default: break;
    }
    char buf[32];
    std::snprintf(buf, sizeof buf, "raw(%u)", unsigned(t));
    return buf;
}

static const char* scope_name(VkScopeKHR s) {
    switch (s) {
        case VK_SCOPE_DEVICE_KHR: return "device";
        case VK_SCOPE_WORKGROUP_KHR: return "workgroup";
        case VK_SCOPE_SUBGROUP_KHR: return "subgroup";
        case VK_SCOPE_QUEUE_FAMILY_KHR: return "queue-family";
        default: return "?";
    }
}

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance instance = VK_NULL_HANDLE;
    CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t n = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
    std::vector<VkPhysicalDevice> devs(n);
    CHECK(vkEnumeratePhysicalDevices(instance, &n, devs.data()));
    std::printf("devices: %u\n", n);
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        std::printf("  [%u] %s  vendor %04x device %04x  api %u.%u.%u  driver %u.%u.%u\n", i, p.deviceName,
                    p.vendorID, p.deviceID, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                    VK_API_VERSION_PATCH(p.apiVersion), VK_API_VERSION_MAJOR(p.driverVersion),
                    VK_API_VERSION_MINOR(p.driverVersion), VK_API_VERSION_PATCH(p.driverVersion));
    }
    if (!n) return 1;

    // The card the port targets: Navi 48, 1002:7550 (RX 9070 / 9070 XT / 9070 GRE).
    VkPhysicalDevice phys = devs[0];
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == 0x1002 && p.deviceID == 0x7550) { phys = devs[i]; break; }
    }

    // Extensions the network's device gate names.
    uint32_t ext_n = 0;
    CHECK(vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_n, nullptr));
    std::vector<VkExtensionProperties> exts(ext_n);
    CHECK(vkEnumerateDeviceExtensionProperties(phys, nullptr, &ext_n, exts.data()));
    const char* want[] = {"VK_KHR_cooperative_matrix", "VK_EXT_shader_float8", "VK_KHR_shader_float16_int8",
                          "VK_EXT_subgroup_size_control", "VK_KHR_8bit_storage", "VK_KHR_16bit_storage",
                          "VK_KHR_workgroup_memory_explicit_layout", "VK_EXT_shader_subgroup_ballot"};
    for (const char* w : want) {
        bool have = false;
        for (const auto& e : exts) if (!std::strcmp(e.extensionName, w)) have = true;
        std::printf("extension %-46s %s\n", w, have ? "yes" : "NO");
    }

    // Features, through the 1.1/1.2/1.3 structs the runtime enables.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat8FeaturesEXT fp8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT};
    VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR wml{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &coop; coop.pNext = &fp8; fp8.pNext = &f11; f11.pNext = &f12; f12.pNext = &f13; f13.pNext = &wml;
    vkGetPhysicalDeviceFeatures2(phys, &f2);
    struct { const char* name; bool have; } feats[] = {
        {"cooperativeMatrix", bool(coop.cooperativeMatrix)},
        {"cooperativeMatrixRobustBufferAccess", bool(coop.cooperativeMatrixRobustBufferAccess)},
        {"shaderFloat8", bool(fp8.shaderFloat8)},
        {"shaderFloat8CooperativeMatrix", bool(fp8.shaderFloat8CooperativeMatrix)},
        {"storageBuffer16BitAccess", bool(f11.storageBuffer16BitAccess)},
        {"storageBuffer8BitAccess", bool(f12.storageBuffer8BitAccess)},
        {"shaderFloat16", bool(f12.shaderFloat16)},
        {"shaderInt8", bool(f12.shaderInt8)},
        {"vulkanMemoryModel", bool(f12.vulkanMemoryModel)},
        {"subgroupSizeControl", bool(f13.subgroupSizeControl)},
        {"synchronization2", bool(f13.synchronization2)},
        {"workgroupMemoryExplicitLayout", bool(wml.workgroupMemoryExplicitLayout)},
        {"workgroupMemoryExplicitLayout8BitAccess", bool(wml.workgroupMemoryExplicitLayout8BitAccess)},
        {"workgroupMemoryExplicitLayout16BitAccess", bool(wml.workgroupMemoryExplicitLayout16BitAccess)},
    };
    for (const auto& f : feats) std::printf("feature   %-46s %s\n", f.name, f.have ? "yes" : "NO");

    VkPhysicalDeviceSubgroupSizeControlProperties sg{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties sgp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceShaderCorePropertiesAMD cores{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD};
    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &sg; sg.pNext = &sgp; sgp.pNext = &drv; drv.pNext = &cores;
    vkGetPhysicalDeviceProperties2(phys, &p2);
    std::printf("driver    %s / %s (conformance %u.%u.%u)\n", drv.driverName, drv.driverInfo,
                drv.conformanceVersion.major, drv.conformanceVersion.minor, drv.conformanceVersion.subminor);
    std::printf("subgroup  default %u, min %u, max %u, requiredSubgroupSizeStages %scompute\n",
                sgp.subgroupSize, sg.minSubgroupSize, sg.maxSubgroupSize,
                (sg.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) ? "" : "no ");
    std::printf("subgroup  ops: basic %d shuffle %d ballot %d vote %d arithmetic %d\n",
                bool(sgp.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT),
                bool(sgp.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT),
                bool(sgp.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT),
                bool(sgp.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT),
                bool(sgp.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT));
    std::printf("shaderCore: engines %u, arrays/engine %u, CU/array %u, SIMD/CU %u, wavefronts/SIMD %u, wave size %u\n",
                cores.shaderEngineCount, cores.shaderArraysPerEngineCount, cores.computeUnitsPerShaderArray,
                cores.simdPerComputeUnit, cores.wavefrontsPerSimd, cores.wavefrontSize);
    std::printf("limits: LDS %u B, workgroup invocations %u, timestampPeriod %.1f ns\n",
                p2.properties.limits.maxComputeSharedMemorySize,
                p2.properties.limits.maxComputeWorkGroupInvocations,
                double(p2.properties.limits.timestampPeriod));

    // Every cooperative matrix configuration. The one the network needs is M=N=K=16, subgroup scope,
    // e4m3 x e4m3 -> f32, non-saturating.
    auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!query) {
        std::printf("cooperative matrix: vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR absent\n");
        return 1;
    }
    uint32_t cn = 0;
    CHECK(query(phys, &cn, nullptr));
    std::vector<VkCooperativeMatrixPropertiesKHR> props(cn, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    CHECK(query(phys, &cn, props.data()));
    std::printf("cooperative matrix configurations: %u\n", cn);
    bool have_fp8_16 = false;
    for (const auto& m : props) {
        std::printf("  %2ux%2ux%-2u %-11s A %-5s B %-5s C %-5s R %-5s sat %d\n", m.MSize, m.NSize, m.KSize,
                    scope_name(m.scope), component_name(m.AType).c_str(), component_name(m.BType).c_str(),
                    component_name(m.CType).c_str(), component_name(m.ResultType).c_str(), int(m.saturatingAccumulation));
        if (m.MSize == 16 && m.NSize == 16 && m.KSize == 16 && m.scope == VK_SCOPE_SUBGROUP_KHR &&
            m.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT && m.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
            m.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && m.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            !m.saturatingAccumulation)
            have_fp8_16 = true;
    }
    std::printf("device gate: %s\n", have_fp8_16 ? "e4m3 x e4m3 -> f32 16x16x16 subgroup present" :
                                                  "MISSING the e4m3 16x16x16 subgroup configuration");
    vkDestroyInstance(instance, nullptr);
    return have_fp8_16 ? 0 : 1;
}
