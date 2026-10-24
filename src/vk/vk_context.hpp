#pragma once

#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(expr) do { VkResult res_ = (expr); if (res_ != VK_SUCCESS) { \
    char buf_[256]; std::snprintf(buf_, sizeof(buf_), "%s:%d %s failed with VkResult=%d", __FILE__, __LINE__, #expr, res_); \
    throw std::runtime_error(buf_); } } while(0)

namespace vk {

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

class Context {
public:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkQueryPool query_pool = VK_NULL_HANDLE;
    float timestamp_period = 1.0f;
    std::string device_name;

    Context() {
        init_instance();
        select_gpu();
        init_device();
        init_commands();
    }

    ~Context() {
        if (device != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(device);
            if (query_pool) vkDestroyQueryPool(device, query_pool, nullptr);
            if (fence) vkDestroyFence(device, fence, nullptr);
            if (cmd_pool) vkDestroyCommandPool(device, cmd_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    Buffer create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props) {
        Buffer b;
        b.size = size;

        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = usage;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device, &bci, nullptr, &b.handle));

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device, b.handle, &req);

        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = find_memory_type(req.memoryTypeBits, props);
        VK_CHECK(vkAllocateMemory(device, &mai, nullptr, &b.memory));
        VK_CHECK(vkBindBufferMemory(device, b.handle, b.memory, 0));

        if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            VK_CHECK(vkMapMemory(device, b.memory, 0, size, 0, &b.mapped));
        }
        return b;
    }

    void destroy_buffer(Buffer& b) {
        if (b.mapped) {
            vkUnmapMemory(device, b.memory);
            b.mapped = nullptr;
        }
        if (b.handle) {
            vkDestroyBuffer(device, b.handle, nullptr);
            b.handle = VK_NULL_HANDLE;
        }
        if (b.memory) {
            vkFreeMemory(device, b.memory, nullptr);
            b.memory = VK_NULL_HANDLE;
        }
    }

    VkShaderModule load_shader(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("Cannot open shader file: " + path);
        size_t sz = size_t(f.tellg());
        f.seekg(0);
        std::vector<uint32_t> code(sz / 4);
        f.read(reinterpret_cast<char*>(code.data()), sz);

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = sz;
        smci.pCode = code.data();
        VkShaderModule sm = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &sm));
        return sm;
    }

    void begin_cmd() {
        VK_CHECK(vkResetCommandBuffer(cmd_buf, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_buf, &bi));
    }

    void end_and_submit() {
        VK_CHECK(vkEndCommandBuffer(cmd_buf));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_buf;
        VK_CHECK(vkResetFences(device, 1, &fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    }

private:
    uint32_t find_memory_type(uint32_t filter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties mp{};
        vkGetPhysicalDeviceMemoryProperties(physical, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
            if ((filter & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type");
    }

    void init_instance() {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.pApplicationName = "vk_runner";
        ai.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &ai;
        VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
    }

    void select_gpu() {
        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
        if (count == 0) throw std::runtime_error("No Vulkan physical devices found");
        std::vector<VkPhysicalDevice> devs(count);
        VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devs.data()));

        // Prefer discrete GPU
        physical = devs[0];
        for (auto d : devs) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                physical = d;
                break;
            }
        }
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physical, &props);
        device_name = props.deviceName;
        timestamp_period = props.limits.timestampPeriod;
        std::cout << "[vk] Selected GPU: " << device_name << " (driver version 0x" << std::hex << props.driverVersion << std::dec << ")" << std::endl;
    }

    void init_device() {
        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &qcount, qprops.data());
        queue_family = qcount;
        for (uint32_t i = 0; i < qcount; ++i) {
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                queue_family = i;
                break;
            }
        }
        if (queue_family == qcount) throw std::runtime_error("No compute queue family found");

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        // Query and enable required features
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
        coop.cooperativeMatrix = VK_TRUE;

        VkPhysicalDeviceShaderFloat16Int8Features f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        f16.shaderFloat16 = VK_TRUE;
        coop.pNext = &f16;

        VkPhysicalDeviceSubgroupSizeControlFeatures ssc{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES};
        ssc.subgroupSizeControl = VK_TRUE;
        ssc.computeFullSubgroups = VK_TRUE;
        f16.pNext = &ssc;

        VkPhysicalDevice16BitStorageFeatures st16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
        st16.storageBuffer16BitAccess = VK_TRUE;
        ssc.pNext = &st16;

        VkPhysicalDevice8BitStorageFeaturesKHR st8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR};
        st8.storageBuffer8BitAccess = VK_TRUE;
        st16.pNext = &st8;

        VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feat2.pNext = &coop;

        const char* exts[] = {
            VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
            VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME,
            VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME,
            VK_KHR_16BIT_STORAGE_EXTENSION_NAME,
            VK_KHR_8BIT_STORAGE_EXTENSION_NAME,
        };

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &feat2;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = sizeof(exts) / sizeof(exts[0]);
        dci.ppEnabledExtensionNames = exts;

        VK_CHECK(vkCreateDevice(physical, &dci, nullptr, &device));
        vkGetDeviceQueue(device, queue_family, 0, &queue);
    }

    void init_commands() {
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = queue_family;
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK(vkCreateCommandPool(device, &cpci, nullptr, &cmd_pool));

        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cmd_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd_buf));

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));

        VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpci.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(device, &qpci, nullptr, &query_pool));
    }
};

} // namespace vk
