// vkprobe: the referee's own look at the Vulkan devices the loader offers under the environment the
// port runs in (run_all.sh V7), independent of anything the port prints.
//
//   vkprobe            one line per physical device:
//                      VK_PROBE_DEVICE: <name> | driverID <n> | <driverName> <driverInfo> | type <n> | fp8 <0|1> | coopmat <0|1>
// Exit 0 when the list is not empty.
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vkprobe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkInstance inst;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) { std::fprintf(stderr, "vkCreateInstance failed\n"); return 1; }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pd(n);
    vkEnumeratePhysicalDevices(inst, &n, pd.data());
    for (VkPhysicalDevice d : pd) {
        VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
        vkGetPhysicalDeviceProperties2(d, &p2);
        uint32_t ne = 0;
        vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> ext(ne);
        vkEnumerateDeviceExtensionProperties(d, nullptr, &ne, ext.data());
        auto has = [&](const char* e) {
            for (auto& x : ext) if (!std::strcmp(x.extensionName, e)) return 1;
            return 0;
        };
        std::printf("VK_PROBE_DEVICE: %s | driverID %d | %s %s | type %d | fp8 %d | coopmat %d\n", p2.properties.deviceName,
                    int(drv.driverID), drv.driverName, drv.driverInfo, int(p2.properties.deviceType),
                    has("VK_EXT_shader_float8"), has("VK_KHR_cooperative_matrix"));
    }
    vkDestroyInstance(inst, nullptr);
    return n ? 0 : 1;
}
