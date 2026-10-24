#pragma once

#include "vk_context.hpp"

namespace vk {

struct ComputePipeline {
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;

    void destroy(VkDevice device) {
        if (handle) vkDestroyPipeline(device, handle, nullptr);
        if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
        if (desc_layout) vkDestroyDescriptorSetLayout(device, desc_layout, nullptr);
        handle = VK_NULL_HANDLE;
        layout = VK_NULL_HANDLE;
        desc_layout = VK_NULL_HANDLE;
    }
};

inline ComputePipeline create_compute_pipeline(
    Context& ctx,
    VkShaderModule shader,
    uint32_t num_storage_buffers,
    uint32_t push_constant_size,
    const char* entry_point = "main")
{
    ComputePipeline p;

    // Build descriptor set layout with N storage buffers
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_buffers);
    for (uint32_t i = 0; i < num_storage_buffers; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo dlci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = uint32_t(bindings.size());
    dlci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(ctx.device, &dlci, nullptr, &p.desc_layout));

    // Pipeline layout with push constants
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = push_constant_size;

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &p.desc_layout;
    if (push_constant_size > 0) {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges = &pcr;
    }
    VK_CHECK(vkCreatePipelineLayout(ctx.device, &plci, nullptr, &p.layout));

    // Compute pipeline creation pinned to Wave32
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo ssc{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    ssc.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.layout = p.layout;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.pNext = &ssc;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = shader;
    cpci.stage.pName = entry_point;

    VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.handle));
    return p;
}

} // namespace vk
