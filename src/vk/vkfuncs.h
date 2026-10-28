// src/vk/vkfuncs.h -- the Vulkan entry points the frame chain calls.
//
// vk_runner links libvulkan, so the loader resolves them.  _nvngx.dll is a Windows
// DLL loaded by a game (or by ngx_client under Proton): it links no Vulkan of its
// own, so the same names are *function pointers* there, filled from the
// gipa/gdpa that NVSDK_NGX_VULKAN_Init is handed, or from the device vkd3d-proton
// hands over.  Compiling the DLL with VK_NO_PROTOTYPES keeps vulkan.h from
// declaring them as functions, and this header declares the pointers instead;
// without that define nothing at all is declared here and the real functions are
// used.
#pragma once

#include <vulkan/vulkan.h>

#define NGXD_VK_FUNCS(X)                 \
    X(vkAllocateCommandBuffers)          \
    X(vkAllocateDescriptorSets)          \
    X(vkAllocateMemory)                  \
    X(vkBeginCommandBuffer)              \
    X(vkBindBufferMemory)                \
    X(vkBindImageMemory)                 \
    X(vkCmdBindDescriptorSets)           \
    X(vkCmdBindPipeline)                 \
    X(vkCmdClearColorImage)              \
    X(vkCmdCopyBuffer)                   \
    X(vkCmdCopyBufferToImage)            \
    X(vkCmdCopyImageToBuffer)            \
    X(vkCmdDispatch)                     \
    X(vkCmdPipelineBarrier)              \
    X(vkCmdPushConstants)                \
    X(vkCreateBuffer)                    \
    X(vkCreateCommandPool)               \
    X(vkCreateComputePipelines)          \
    X(vkCreateDescriptorPool)            \
    X(vkCreateDescriptorSetLayout)       \
    X(vkCreateImage)                     \
    X(vkCreateImageView)                 \
    X(vkCreatePipelineLayout)            \
    X(vkCreateSampler)                   \
    X(vkCreateShaderModule)              \
    X(vkDestroyBuffer)                   \
    X(vkDestroyCommandPool)              \
    X(vkDestroyDescriptorPool)           \
    X(vkDestroyDescriptorSetLayout)      \
    X(vkDestroyImage)                    \
    X(vkDestroyImageView)                \
    X(vkDestroyPipeline)                 \
    X(vkDestroyPipelineLayout)           \
    X(vkDestroySampler)                  \
    X(vkDestroyShaderModule)             \
    X(vkDeviceWaitIdle)                  \
    X(vkEndCommandBuffer)                \
    X(vkFreeMemory)                      \
    X(vkGetBufferMemoryRequirements)     \
    X(vkGetDeviceQueue)                  \
    X(vkGetImageMemoryRequirements)      \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkMapMemory)                       \
    X(vkQueueSubmit)                     \
    X(vkQueueWaitIdle)                   \
    X(vkResetDescriptorPool)             \
    X(vkUnmapMemory)                     \
    X(vkUpdateDescriptorSets)

#ifdef VK_NO_PROTOTYPES
#define NGXD_VK_DECL(f) extern PFN_##f f;
NGXD_VK_FUNCS(NGXD_VK_DECL)
#undef NGXD_VK_DECL
#endif
