// src/vk/runner.cpp -- the CLI around the frame chain: it reads a frame directory, feeds
// the chain's host-visible staging buffers and writes out_<i>.f32.  All of the dispatch
// work lives in src/vk/pipeline.cpp so that the NGX bridge can reuse it.

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

#include "pipeline.h"

using namespace ngxd;

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

    // Per-dispatch GPU timestamps (Step D, DLSS5_AMD_PROFILE=1).  Two queries per dispatch
    // bracket the vkCmdDispatch itself, so the memory barrier after each dispatch is charged
    // to neither neighbour and the sum of dispatches can be compared against the frame total.
    // The hook lives here rather than in the chain so that the NGX path, which never installs
    // one, records exactly what it always did and the DLL gains no Vulkan entry point.
    const bool prof_on = std::getenv("DLSS5_AMD_PROFILE") != nullptr;
    constexpr uint32_t kProfMax = 64;
    VkQueryPool prof_pool = VK_NULL_HANDLE;
    std::vector<std::string> prof_names;
    uint32_t prof_n = 0;                                   // dispatches bracketed this frame
    float prof_period = picked_props.limits.timestampPeriod;   // ns per timestamp tick

    // Load shaders, create pipelines and allocate the chain's buffers
    std::string chain_err;
    const char* shader_env = std::getenv("DLSS5_AMD_SHADER_DIR");
    const std::string shader_dir = (shader_env && *shader_env) ? shader_env : "build/shaders";
    FrameChain* chain = FrameChain::create(dev, picked_phys, mode, preset, W, H, oW, oH,
                                           wdir, shader_dir, &chain_err);
    if (!chain) {
        std::fprintf(stderr, "%s\n", chain_err.c_str());
        return 1;
    }
    // Diagnosis: DLSS5_AMD_INJECT_DIR=<dump root> makes the chain copy the oracle's own
    // buffer contents over its results after the dispatch points DLSS5_AMD_INJECT_MASK
    // names (empty = all), so the end-to-end PSNR of the run measures only what runs
    // downstream of those points.  Unset, nothing changes.
    if (const char* inj = std::getenv("DLSS5_AMD_INJECT_DIR")) {
        const char* m = std::getenv("DLSS5_AMD_INJECT_MASK");
        chain->set_injection(inj, m ? m : "");
    }

    std::printf("PIPELINES_CREATED: %zu\n", chain->pipeline_count());
    std::fflush(stdout);

    if (prof_on) {
        VkQueryPoolCreateInfo pq{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        pq.queryType = VK_QUERY_TYPE_TIMESTAMP;
        pq.queryCount = 2 * kProfMax;
        vkCreateQueryPool(dev, &pq, nullptr, &prof_pool);
        chain->set_prof_hook([&](const char* name, int phase) {
            if (prof_n >= kProfMax) return;
            if (phase == 0) {
                if (prof_names.size() <= prof_n) prof_names.resize(prof_n + 1);
                prof_names[prof_n] = name;
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, prof_pool, 2 * prof_n);
            } else {
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, prof_pool, 2 * prof_n + 1);
                ++prof_n;
            }
        });
        std::fprintf(stderr, "PROFILE: per-dispatch GPU timestamps on (%.3f ns/tick, cap %u)\n",
                     prof_period, kProfMax);
    }

    // Frame execution loop
    for (int fi = 0; fi < nframes; ++fi) {
        std::string sfx = "_" + std::to_string(fi);
        auto pf_raw = slurp(fdir + "/params" + sfx + ".json");
        std::string pf(pf_raw.begin(), pf_raw.end());

        FrameParams fp;
        fp.fmt_col = jint(pf, "fmt_col");
        fp.fmt_mv = jint(pf, "fmt_mv");
        fp.fmt_dep = jint(pf, "fmt_dep");
        fp.mvhi = jint(pf, "mvhi");
        fp.ldr = jint(pf, "ldr");
        fp.exposure = float(jget(pf, "exposure").empty() ? 1.0 : jget(pf, "exposure")[0]);
        fp.pre_exposure = float(jget(pf, "pre_exposure").empty() ? 1.0 : jget(pf, "pre_exposure")[0]);
        fp.reset = jint(pf, "reset");
        auto jit = jget(pf, "jitter");
        fp.jx = float(jit.size() > 0 ? jit[0] : 0.0);
        fp.jy = float(jit.size() > 1 ? jit[1] : 0.0);
        auto jitp = jget(pf, "jitter_prev");
        fp.jpx = float(jitp.size() > 0 ? jitp[0] : 0.0);
        fp.jpy = float(jitp.size() > 1 ? jitp[1] : 0.0);
        fp.frame_index = fi;      // monotonic sequence index (injection dump dir; pipeline.h)

        // Load frame textures
        std::string c_ext = (fp.fmt_col == 0) ? ".f16" : (".c" + std::to_string(fp.fmt_col));
        std::string m_ext = (fp.fmt_mv == 0) ? ".f16" : (".c" + std::to_string(fp.fmt_mv));
        std::string d_ext = (fp.fmt_dep == 0) ? ".f32" : (".c" + std::to_string(fp.fmt_dep));

        auto c_raw = slurp(fdir + "/col" + sfx + c_ext);
        auto m_raw = slurp(fdir + "/mv" + sfx + m_ext);
        auto d_raw = slurp(fdir + "/dep" + sfx + d_ext);

        chain->fill_color(c_raw.data(), c_raw.size());
        chain->fill_mv(m_raw.data(), m_raw.size());
        chain->fill_depth(d_raw.data(), d_raw.size());

        if (mode == "rr") {
            auto n_raw = slurp(fdir + "/nrm" + sfx + ".f16");
            if (n_raw.empty()) n_raw = slurp(fdir + "/norm" + sfx + ".f16");
            auto r_raw = slurp(fdir + "/rgh" + sfx + ".f32");
            auto a_raw = slurp(fdir + "/alb" + sfx + ".f16");
            auto s_raw = slurp(fdir + "/spa" + sfx + ".f16");
            chain->fill_normal(n_raw.data(), n_raw.size());
            chain->fill_rough(r_raw.data(), r_raw.size());
            chain->fill_albedo(a_raw.data(), a_raw.size());
            chain->fill_specular(s_raw.data(), s_raw.size());
        }

        // Record commands
        vkResetCommandBuffer(cmd, 0);
        prof_n = 0;
        // Host-side reset: safe here because the previous frame's queue wait has completed.
        if (prof_on) vkResetQueryPool(dev, prof_pool, 0, 2 * kProfMax);

        VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &cbbi);

        vkCmdResetQueryPool(cmd, qpool, 0, 2);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);

        chain->record(cmd, fp);

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

        // Per-dispatch breakdown (Step D).  GPU_FRAME_MS above brackets chain->record() only -
        // texture slurps and the host-side fill_* memcpy happen before it, and the D2H readback
        // and file write after - so this is render work with setup and uploads already excluded.
        if (prof_on && prof_n) {
            std::vector<uint64_t> q(2 * prof_n);
            vkGetQueryPoolResults(dev, prof_pool, 0, 2 * prof_n, q.size() * sizeof(uint64_t),
                                  q.data(), sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            double sum = 0.0;
            for (uint32_t i = 0; i < prof_n; ++i) {
                double d = double(q[2 * i + 1] - q[2 * i]) * prof_period / 1e6;
                if (q[2 * i + 1] < q[2 * i]) d = 0.0;   // queries not written: dispatch skipped
                sum += d;
                std::printf("DISPATCH_MS %d %u %s %.4f\n", fi, i,
                            i < prof_names.size() ? prof_names[i].c_str() : "?", d);
            }
            std::printf("DISPATCH_SUM_MS %d %u %.4f FRAME_MS %.4f OTHER_MS %.4f\n",
                        fi, prof_n, sum, ms, ms - sum);
        }

        // Write output frame
        std::string out_path = odir + "/out_" + std::to_string(fi) + ".f32";
        std::ofstream fout(out_path, std::ios::binary);
        if (fout) {
            fout.write(static_cast<const char*>(chain->out_color()), size_t(oW) * oH * 16);
        }

        std::printf("frame %d done (parity %d)\n", fi, chain->parity());
        std::printf("GPU_FRAME_MS %d %.4f\n", fi, ms);
        std::fflush(stdout);

        // Diagnosis: DLSS5_AMD_DUMP_BUF=name:PATH writes that chain buffer's bytes (same
        // names as DLSS5_AMD_INJECT_DIR) after the frame's GPU work completes.  Unset: no-op.
        if (const char* db = std::getenv("DLSS5_AMD_DUMP_BUF")) {
            const std::string all(db);
            for (size_t pos = 0; pos < all.size();) {
                size_t comma = all.find(',', pos);
                if (comma == std::string::npos) comma = all.size();
                const std::string spec = all.substr(pos, comma - pos);
                pos = comma + 1;
                size_t c = spec.find(':');
                if (c == std::string::npos) continue;
                std::vector<uint8_t> bytes;
                if (chain->read_buffer(spec.substr(0, c), &bytes)) {
                    std::ofstream f(spec.substr(c + 1), std::ios::binary);
                    f.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
                    std::fprintf(stderr, "DUMP_BUF %s -> %s (%zu bytes)\n", spec.substr(0, c).c_str(),
                                 spec.substr(c + 1).c_str(), bytes.size());
                } else {
                    std::fprintf(stderr, "DUMP_BUF: unknown buffer %s\n", spec.substr(0, c).c_str());
                }
            }
        }

        chain->advance();
    }

    // Cleanup
    vkDeviceWaitIdle(dev);
    delete chain;
    vkDestroyQueryPool(dev, qpool, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);

    return 0;
}
