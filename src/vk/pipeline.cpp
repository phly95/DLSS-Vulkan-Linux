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
// (-1 = absent), s1d2 / s1d4 / s1d6 / s1d0 as u16 bits (21..24), and unit 15's dec0_scalars (Args +0xe8 /
// +0xec, f32 bits, 25 / 26; 0 for the other units). 32 uint32 per unit (rr_level binding 13).
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
            tab[k * 32 + j] = uint32_t(int32_t(v));
            i = size_t(e - pj.c_str());
        }
        // the unit's own keys follow its meta and precede the next unit's meta
        const size_t base = i;
        const size_t next = (k < 15) ? pj.find("\"meta\"", base) : std::string::npos;
        static const char* const keys[4] = {"\"s1d2\"", "\"s1d4\"", "\"s1d6\"", "\"s1d0\""};
        static const int slot[4] = {21, 22, 23, 24};
        for (int s = 0; s < 4; ++s) {
            size_t q = pj.find(keys[s], base);
            if (q == std::string::npos) return false;
            size_t c = pj.find(':', q);
            if (c == std::string::npos) return false;
            ++c;
            while (c < pj.size() && (pj[c] == ' ' || pj[c] == '\n' || pj[c] == '\t' || pj[c] == '\r')) ++c;
            char* e = nullptr;
            long v = std::strtol(pj.c_str() + c, &e, 10);
            if (e == pj.c_str() + c) return false;
            tab[k * 32 + slot[s]] = uint32_t(v) & 0xFFFFu;
        }

        float dec0[2] = {0.0f, 0.0f};
        size_t dq = pj.find("\"dec0_scalars\"", base);
        if (dq != std::string::npos && (next == std::string::npos || dq < next)) {
            size_t c = pj.find(':', dq);
            if (c != std::string::npos) {
                ++c;
                while (c < pj.size() && (pj[c] == ' ' || pj[c] == '\n' || pj[c] == '\t' || pj[c] == '\r')) ++c;
                if (c < pj.size() && pj[c] == '[') {
                    char* e = nullptr;
                    dec0[0] = std::strtof(pj.c_str() + c + 1, &e);
                    size_t comma = pj.find(',', size_t(e - pj.c_str()));
                    if (comma != std::string::npos) dec0[1] = std::strtof(pj.c_str() + comma + 1, nullptr);
                }
            }
        }
        std::memcpy(&tab[k * 32 + 25], &dec0[0], 4);
        std::memcpy(&tab[k * 32 + 26], &dec0[1], 4);
        cur = base;
    }
    return true;
}

// ---- rr_swin repack (mu_rrl): tiled stored-order weights + f32 vector/gram tables ----
// Mirrors tools/rr/rr_layout.py (frag order) and rr_pack.py (permutations).  The only
// consumer is shaders/rr/rr_swin.comp: every weight matrix as [rowtile][coltile][16][16]
// = 256-byte tiles (one cooperative-matrix load, stride 16), activation-channel axes in
// the activation's STORED order (byte j of a 32-group holds logical channel P(j), so the
// kernel indexes channels by stored byte with no permute).  LayerScale is folded here:
// Wo' = s1d2*Wo, v128' = s1d2*v128, W2' = s1d4*W2, b2' = s1d4*b2 (f32 scalars from tab).
namespace {
static uint32_t rrsw_frag_u(uint32_t u, uint32_t c, uint32_t cols) {
    return 16u * (u & 15u) + 256u * ((c >> 2u) & 1u) + 8u * ((c >> 3u) & 1u) + 4u * ((c >> 1u) & 1u)
         + 2u * ((c >> 4u) & 1u) + (c & 1u) + 512u * (c >> 5u) + 16u * cols * (u >> 4u);
}
static uint32_t rrsw_frag_off(uint32_t r, uint32_t c, uint32_t cols) {
    uint32_t u = ((r & 0x7FFFFFE1u) | ((r >> 3) & 2u)) + (r & 0xEu) * 2u;
    return rrsw_frag_u(u, c, cols);
}
static uint32_t rrsw_perm_P(uint32_t i) {
    return ((i >> 2u) << 1u) | (i & 1u) | (((i >> 1u) & 1u) << 4u);
}
static uint32_t rrsw_basis(uint32_t idx) { return (idx & ~31u) | rrsw_perm_P(idx & 31u); }
static uint32_t rrsw_pinv(uint32_t k) {
    return (k & 1u) | ((k & 0x10u) >> 3u) | ((k & 2u) << 1u) | ((k & 4u) << 1u) | ((k & 8u) << 1u);
}
static float rrsw_f16_to_f32(uint32_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 0x3FFu;
    if (e == 0u) {
        float v = float(m) * 5.9604644775390625e-08f;
        return (s != 0u) ? -v : v;
    }
    // Sign rides in b for normals and inf/nan alike (e+112 <= 142, no overflow).
    uint32_t b = (s << 31) | ((e == 31u ? 0xFFu : e + 112u) << 23) | (m << 13);
    float v;
    std::memcpy(&v, &b, 4);
    return v;
}
static float rrsw_e4m3_to_f32(uint32_t b) {
    uint32_t e = (b >> 3u) & 15u;
    float v = (e == 0u) ? float(b & 7u) * 0.001953125f
                        : ([&] { uint32_t bb = ((e + 120u) << 23) | ((b & 7u) << 20); float f;
                                  std::memcpy(&f, &bb, 4); return f; })();
    return ((b & 0x80u) != 0u) ? -v : v;
}
static uint8_t rrsw_f32_to_e4m3(float x) {
    if (!std::isfinite(x)) return 0x7Fu;
    uint32_t s;
    std::memcpy(&s, &x, 4);
    uint32_t sgn = (s >> 24) & 0x80u;
    float a = std::fabs(x);
    if (a > 448.0f) a = 448.0f;
    uint32_t ab;
    std::memcpy(&ab, &a, 4);
    uint32_t code;
    if (a >= 0.015625f) code = ((ab + 0x407FFFFu + ((ab >> 20) & 1u)) >> 20) & 0xFFu;
    else code = uint32_t(std::nearbyint(a * 512.0f));
    return uint8_t(sgn | code);
}

// Matrix sizes (tiles of 256 B) for unit (C,G,Cdown,Cup) with the given present items.
static size_t rrsw_unit_tiles(uint32_t C, uint32_t G, uint32_t Cdown, uint32_t Cup, const uint32_t* meta) {
    const uint32_t NT = C / 16, NBLK = C / 8;
    size_t t = 0;
    t += size_t(G) * 4u * NT;                 // Q+V per head
    t += size_t(G) * NT * 2u;                 // Wo per head
    t += size_t(NBLK) * 2u * NT;              // fc1 per block
    t += size_t(NBLK) * NT * 2u;              // fc2 per block
    t += NT;                                  // m108 (always present)
    if (meta[8] != 0xFFFFFFFFu) t += size_t(Cdown / 16) * size_t(4 * C / 16);   // mA8
    if (meta[10] != 0xFFFFFFFFu) t += size_t(4) * size_t(128 / 16);             // mC8
    if (meta[12] != 0xFFFFFFFFu) t += size_t(4 * C / 16) * size_t(Cup / 16);    // mE8
    if (meta[16] != 0xFFFFFFFFu) t += size_t(4) * size_t(64 / 16);              // m1d8
    if (meta[18] != 0xFFFFFFFFu) t += size_t(2) * size_t(64 / 16);              // m1f8
    return t;
}
static size_t rrsw_unit_tab(uint32_t C, uint32_t G) { return size_t(6 * C + 20 + G * 4096 + 962); }

// Repacks unit (C,G,Cdown,Cup): blob (wblob, size bytes, meta = 21 item offsets,
// s1d2/s1d4 f32) -> tiled matrices at wout, f32 table at tout.  Returns false when any
// byte read would leave the blob.
static bool rrsw_repack_unit(const uint8_t* wblob, size_t size, const uint32_t* meta,
                             uint32_t C, uint32_t G, uint32_t Cdown, uint32_t Cup,
                             float s1d2, float s1d4, uint8_t* wout, float* tout) {
    const uint32_t NT = C / 16, NBLK = C / 8;
    auto byte_ok = [&](uint32_t rel) { return size_t(rel) < size; };
    auto tile = [&](uint32_t rel, uint32_t row0, uint32_t col0, uint32_t rows, uint32_t cols,
                    uint32_t scols, size_t tiles_off, bool perm_r, bool perm_c, bool plain) {
        if (rel == 0xFFFFFFFFu) return true;
        for (uint32_t rt = 0; rt < rows / 16u; ++rt)
            for (uint32_t ct = 0; ct < cols / 16u; ++ct)
                for (uint32_t n = 0u; n < 16u; ++n)
                    for (uint32_t k = 0u; k < 16u; ++k) {
                        uint32_t r = row0 + rt * 16u + n, c = col0 + ct * 16u + k;
                        uint32_t rl = perm_r ? rrsw_basis(r) : r;
                        uint32_t cl = perm_c ? rrsw_basis(c) : c;
                        uint32_t src = plain ? rrsw_frag_u(rl, cl, scols) : rrsw_frag_off(rl, cl, scols);
                        if (!byte_ok(rel + src)) return false;
                        wout[tiles_off + (size_t(rt) * (cols / 16u) + ct) * 256u + n * 16u + k] =
                            wblob[rel + src];
                    }
        return true;
    };
    size_t cur = 0;
    const size_t rHS = size_t(4u) * NT * 256u;
    for (uint32_t h = 0u; h < G; ++h) {
        if (!tile(meta[0] + h * 64u * C, 0, 0, 32, C, C, cur, false, true, false)) return false;
        if (!tile(meta[0] + h * 64u * C, 32, 0, 32, C, C, cur + size_t(2u) * NT * 256u, false, true, false))
            return false;
        cur += rHS;
    }
    for (uint32_t h = 0u; h < G; ++h) {
        if (!tile(meta[2], 0, 32u * h, C, 32, 32u * G, cur, true, false, false)) return false;
        cur += size_t(NT) * 2u * 256u;
    }
    for (uint32_t j = 0u; j < NBLK; ++j) {
        if (!tile(meta[4], 32u * j, 0, 32, C, C, cur, false, true, false)) return false;
        cur += size_t(2u) * NT * 256u;
    }
    for (uint32_t j = 0u; j < NBLK; ++j) {
        if (!tile(meta[6], 0, 32u * j, C, 32, 4u * C, cur, true, false, false)) return false;
        cur += size_t(NT) * 2u * 256u;
    }
    if (!tile(meta[14], 0, 0, 16, C, C, cur, false, true, true)) return false;
    cur += size_t(NT) * 256u;
    if (!tile(meta[8], 0, 0, Cdown, 4 * C, 4u * C, cur, false, false, false)) return false;
    if (meta[8] != 0xFFFFFFFFu) cur += size_t(Cdown / 16) * size_t(4 * C / 16) * 256u;
    if (!tile(meta[10], 0, 0, 64, 128, 128, cur, true, true, false)) return false;
    if (meta[10] != 0xFFFFFFFFu) cur += size_t(4) * size_t(8) * 256u;
    if (!tile(meta[12], 0, 0, 4 * C, Cup, Cup, cur, true, true, false)) return false;
    if (meta[12] != 0xFFFFFFFFu) cur += size_t(4 * C / 16) * size_t(Cup / 16) * 256u;
    if (!tile(meta[16], 0, 0, 64, 64, 64, cur, false, true, false)) return false;
    if (meta[16] != 0xFFFFFFFFu) cur += size_t(4) * size_t(4) * 256u;
    if (!tile(meta[18], 0, 0, 32, 64, 64, cur, true, false, false)) return false;
    (void)cur;
    // NOTE: no LayerScale fold on Wo/W2 (it re-quantises the weights and costs ~6 dB):
    // the shader scales ctx by s1d2 and h by s1d4 instead.  v128'/b2' above stay folded
    // (f32 vectors: exact).
    // f32 table: v128' [C], b2' [C], v140 [4C hidden, logical order], v1b8 [16],
    // s1d0, s1d6, s1d2, s1d4, gram [G][64][64], then per-mode extras (uniform offsets):
    // v218[64] (u0 hidden, logical), v230[32] (u0, stored), v188[64] (u1, stored),
    // v1a0[640] (mode2, stored per-quad), v170[160] (down, stored), dec0_s0/f0 (u15).
    // (LayerScale is folded into the v128'/b2' vectors only; Wo/W2 stay exact and the
    // shader scales ctx/h by s1d2/s1d4 from slots 18/19.)
    auto f16at = [&](uint32_t rel, uint32_t i) -> float {
        if (!byte_ok(rel + 2u * i + 1u)) return 0.0f;
        return rrsw_f16_to_f32(uint32_t(wblob[rel + 2u * i]) | (uint32_t(wblob[rel + 2u * i + 1u]) << 8));
    };
    if (meta[3] != 0xFFFFFFFFu)
        for (uint32_t j = 0u; j < C; ++j) {
            uint32_t bj = rrsw_basis(j);
            uint32_t st = (bj & ~31u) | rrsw_pinv(bj & 31u);
            tout[j] = s1d2 * f16at(meta[3], st);
        }
    if (meta[7] != 0xFFFFFFFFu)
        for (uint32_t j = 0u; j < C; ++j) {
            uint32_t bj = rrsw_basis(j);
            uint32_t st = (bj & ~31u) | rrsw_pinv(bj & 31u);
            tout[C + j] = s1d4 * f16at(meta[7], st);
        }
    if (meta[5] != 0xFFFFFFFFu)
        for (uint32_t j = 0u; j < 4 * C; ++j) tout[2 * C + j] = f16at(meta[5], (j & ~31u) | rrsw_pinv(j & 31u));
    if (meta[15] != 0xFFFFFFFFu)
        for (uint32_t j = 0u; j < 16; ++j) tout[6 * C + j] = f16at(meta[15], j);
    tout[6 * C + 16] = rrsw_f16_to_f32(meta[24] & 0xFFFFu);   // s1d0
    tout[6 * C + 17] = rrsw_f16_to_f32(meta[23] & 0xFFFFu);   // s1d6
    tout[6 * C + 18] = rrsw_f16_to_f32(meta[21] & 0xFFFFu);   // s1d2 (unfolded scale)
    tout[6 * C + 19] = rrsw_f16_to_f32(meta[22] & 0xFFFFu);   // s1d4 (unfolded scale)
    if (meta[1] != 0xFFFFFFFFu)
        for (uint32_t j = 0u; j < G * 4096u; ++j) tout[6 * C + 20 + j] = f16at(meta[1], j);
    // Per-mode extras (uniform offsets from the gram end; activation axes stored
    // raw = blob order, hidden axes de-permuted to logical order).
    {
        const size_t VH = size_t(6 * C + 20 + G * 4096u);
        if (meta[17] != 0xFFFFFFFFu)
            for (uint32_t j = 0u; j < 64u; ++j)
                tout[VH + j] = f16at(meta[17], (j & ~31u) | rrsw_pinv(j & 31u));   // v218 hidden
        if (meta[19] != 0xFFFFFFFFu)
            for (uint32_t j = 0u; j < 32u; ++j) tout[VH + 64u + j] = f16at(meta[19], j);  // v230
        if (meta[11] != 0xFFFFFFFFu)
            for (uint32_t j = 0u; j < 64u; ++j) tout[VH + 96u + j] = f16at(meta[11], j);  // v188
        if (meta[13] != 0xFFFFFFFFu)
            for (uint32_t j = 0u; j < 4u * C; ++j) tout[VH + 160u + j] = f16at(meta[13], j); // v1a0
        if (meta[9] != 0xFFFFFFFFu)
            for (uint32_t j = 0u; j < Cdown; ++j) tout[VH + 800u + j] = f16at(meta[9], (j & ~31u) | rrsw_pinv(j & 31u));  // v170 logical
    }
    return true;
}
} // namespace

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
    // VRAM first: host-visible device-local memory (ReBAR / the visible VRAM window). Asking only for
    // HOST_VISIBLE | HOST_COHERENT picks the first such type, which on RADV is system RAM (GTT), so
    // every kernel read and wrote across PCIe. System RAM stays the fallback for a device without a
    // host-visible VRAM type.
    mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    {
        VkPhysicalDeviceMemoryProperties mp;
        vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((mp.memoryTypes[mai.memoryTypeIndex].propertyFlags & want) != want)
            mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
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

// subgroup_size != 0 pins the stage's subgroup size (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo,
// needs shaderSubgroupSizeControl). k_sr_stage needs it: this driver's cooperative-matrix path computes
// garbage at subgroup size 64 (measured: the same shader logic gives sane values with 32-lane
// workgroups and garbage with 64-lane ones - see build/agents/stage/progress.md), and the reference
// runtime pins wave32 the same way for every pipeline (reference-nr-vulkan/linux/src/core/nrvk.hpp).
static ChainPipeline create_compute_pipeline(VkDevice dev, const std::vector<uint32_t>& spv, uint32_t num_bindings, uint32_t push_size, uint32_t subgroup_size = 0,
                                             const std::vector<VkDescriptorType>& types = {}) {
    ChainPipeline p;
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_bindings);
    for (uint32_t i = 0; i < num_bindings; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = (i < types.size()) ? types[i] : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    p.dt.resize(num_bindings, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    for (uint32_t i = 0; i < num_bindings && i < types.size(); ++i) p.dt[i] = types[i];
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
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo req{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (subgroup_size) {
        req.requiredSubgroupSize = subgroup_size;
        cpci.stage.pNext = &req;
    }
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &p.pipeline);
    if ((r != VK_SUCCESS || p.pipeline == VK_NULL_HANDLE) && subgroup_size) {
        // This kernel REQUIRES the pinned subgroup size: at subgroup size 64 its cooperative-matrix
        // path computes garbage (measured - build/agents/stage/progress.md), so a silent retry without
        // the pin would be corruption in a game. The device has to support (and have enabled)
        // subgroupSizeControl - say so and fail the chain instead of running wrong.
        g_pipeline_error = std::string("required subgroup size ") + std::to_string(subgroup_size) +
            " rejected (VkResult " + std::to_string(int(r)) +
            "): sr_stage computes incorrectly at subgroup size 64 and must not run unpinned - the "
            "VkDevice needs VkPhysicalDeviceVulkan13Features::subgroupSizeControl";
        p.pipeline = VK_NULL_HANDLE;
    }
    vkDestroyShaderModule(dev, mod, nullptr);
    if ((r != VK_SUCCESS || p.pipeline == VK_NULL_HANDLE) && g_pipeline_error.empty()) {
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

// SR weight repack for the cooperative-matrix stage kernels. NeuralHub stores every fp8 matrix of a stage
// blob in its per-lane WMMA fragment order (frag_off, the same as tools/sr/sr_pack.py frag_offsets); a KHR
// coopMatLoad needs rows, so wrep holds each matrix as [N][K] row-major (K contiguous) at the same offset.
// The vectors (f16 gains, biases) and the relpos tables are copied unchanged. The walk is StageWeights'
// (tools/sr/sr_stage_model.py) and ends exactly at the blob size, which is checked.
namespace {
struct SrStageShape { uint32_t C, g, Cn, mode; };
// tools/sr/sr_pack.py LAYERS: stage0..9 = enc1, enc2, tube0..5, dec2, dec1.
const SrStageShape kSrStageShape[10] = {
    {64, 2, 96, 0}, {96, 4, 128, 0}, {128, 4, 0, 1}, {128, 4, 0, 1}, {128, 4, 0, 1},
    {128, 4, 0, 1}, {128, 4, 0, 1}, {128, 4, 0, 1}, {96, 4, 128, 2}, {64, 2, 96, 2}};
// Preset L's level-0 layers (perf mu_l0): enc0L (w_enc0+6272, 24192 B, mode 0) and dec0L
// (w_dec0+0, 24320 B, mode 2), both C=32, g=2, Cn=64. New array (kSrStageShape is shared).
const SrStageShape kSrL0Shape[2] = {{32, 2, 64, 0}, {32, 2, 64, 2}};

uint32_t sr_frag_off(uint32_t r, uint32_t c, uint32_t cols) {
    uint32_t u1 = ((r & 0x7FFFFFE1u) | ((r >> 3) & 2u)) + (r & 0xEu) * 2u;
    uint32_t u5 = 2u * c;
    uint32_t u9 = 16u * c;
    uint32_t low = (u5 & 4u) | (c & 1u) | ((c >> 3) & 2u);
    uint32_t high = ((u5 & 0x10u) << 4) + (u9 & ~0x1FFu) + (cols << 4) * (u1 >> 4);
    return (low | high) + ((u1 & 0xFu) + (u5 & 8u) * 2u) * 8u;
}

// Brief item (b): sr_swin's per-stage f32 vector table. Byte layout (stage-relative, 16384
// stride): gain1 [C] @0, v270 @4C, gain2 @8C, fc2_bias @12C, b1 [NBLK*32] @16C, rel^T [g*256]
// @32C - every vector in LOGICAL channel order, each tile 16 contiguous f32 so a lane's 8 values
// are one 32-byte run (the shader's LANE8). Fills the table for stage `si`; false if the blob is
// too small.
static bool sr_fill_vec_table(const uint8_t* blob, size_t blob_size, uint8_t* tab, const SrStageShape& s) {
    auto f16f = [](const uint8_t* p) {
        uint32_t h = uint32_t(p[0]) | (uint32_t(p[1]) << 8);
        uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
        uint32_t e = (exp == 0) ? 0u : (exp + 112u);          // weights are normal or zero
        uint32_t b = (sign << 31) | (e << 23) | (man << 13);
        float v; std::memcpy(&v, &b, 4); return v;
    };
    auto perm_invP = [](uint32_t c) { return (((c >> 1u) & 7u) << 2u) | (c & 1u) | (((c >> 4u) & 1u) << 1u); };
    auto vidx = [&](uint32_t c) { return (c & ~31u) | perm_invP(c); };
    auto perm_P = [](uint32_t i) { return ((i >> 2u) << 1u) | (i & 1u) | (((i >> 1u) & 1u) << 4u); };
    // chan: the vector is indexed by C channels -> store it in the SAME stored basis as the tile
    // repack (slot j = logical perm_P(j)); permuted: the blob's own P-permuted vector storage.
    auto vec = [&](uint32_t base, uint32_t n, uint32_t tab_off, uint32_t permuted, uint32_t chan) {
        if (size_t(base) + 2u * n > blob_size) return false;
        for (uint32_t c = 0; c < n; ++c) {
            uint32_t lc = chan ? ((c & ~31u) | perm_P(c & 31u)) : c;
            float v = f16f(blob + base + 2u * (permuted ? vidx(lc) : lc));
            std::memcpy(tab + 4u * (tab_off + c), &v, 4);
        }
        return true;
    };
    const uint32_t C = s.C, g = s.g, Cn = s.Cn;
    uint32_t base = (s.mode == 2) ? 4u * C * Cn + 8u * C : 0u;
    if (!vec(base, C, 0, 1, 1)) return false;                                // gain1 (channel)
    uint32_t v270 = base + 2u * C + g * (96u * C + 512u);
    if (!vec(v270, C, C, 1, 1)) return false;                                // v270 (channel)
    if (!vec(v270 + 2u * C, C, 2u * C, 1, 1)) return false;                  // gain2 (channel)
    uint32_t ffn = base + 6u * C + g * (96u * C + 512u);
    for (uint32_t j = 0; j < C / 8u; ++j) {
        uint32_t b1o = ffn + j * (64u * C + 64u) + ((j == 0u) ? 0u : 2u * C) + 32u * C;
        if (!vec(b1o, 32, 4u * C + 32u * j, 1, 0)) return false;             // b1: HIDDEN dim, no perm
    }
    if (!vec(ffn + 64u * C + 64u, C, 3u * C, 1, 1)) return false;            // fc2_bias (channel)
    for (uint32_t h = 0; h < g; ++h) {                                       // rel^T: TOKEN dim
        uint32_t rlo = base + 2u * C + h * (96u * C + 512u) + 64u * C;
        if (!vec(rlo, 256, 8u * C + 256u * h, 0, 0)) return false;
    }
    return true;
}

// Brief item (a): sr_swin's tile-contiguous weight repack. Per stage, in compact order:
//   Wq [g][32][C], Wk [g][32][C], Wo [g][C][32], W1 [NBLK][32][C], W2 [NBLK][C][32]
// each matrix as [rowtile][coltile][16][16] = 256-byte tiles, so a cooperative-matrix load is
// one tile with stride 16 (a literal) and the tile index folds into the immediate offset -
// the review counted 47 address VALU per MLP block against this.
static bool sr_repack_tiles(const uint8_t* wbuf, uint8_t* out, size_t woff, const SrStageShape& s) {
    const uint32_t C = s.C, g = s.g, CT = C / 16u, NBLK = C / 8u;
    // Brief item (c): channel axes go into the activation's STORED basis - byte j of a 32-group
    // holds logical channel perm_P(j), so the kernel can index channels by stored byte and the
    // input/output move as whole bytes. perm_c maps a BASIS channel index to the LOGICAL one the
    // blob (and sr_frag_off) speaks; group order is untouched (groups of 32 stay aligned).
    auto perm_P = [](uint32_t i) { return ((i >> 2u) << 1u) | (i & 1u) | (((i >> 1u) & 1u) << 4u); };
    auto basis = [&](uint32_t idx) { return (idx & ~31u) | perm_P(idx & 31u); };
    auto tile = [&](size_t rel, uint32_t rows, uint32_t cols, size_t tiles_off, bool perm_r, bool perm_c) {
        for (uint32_t rt = 0; rt < rows / 16u; ++rt)
            for (uint32_t ct = 0; ct < cols / 16u; ++ct)
                for (uint32_t n = 0; n < 16u; ++n)
                    for (uint32_t k = 0; k < 16u; ++k) {
                        uint32_t r = rt * 16u + n, c = ct * 16u + k;
                        uint32_t rl = perm_r ? basis(r) : r;
                        uint32_t cl = perm_c ? basis(c) : c;
                        out[tiles_off + (size_t(rt) * (cols / 16u) + ct) * 256u + n * 16u + k] =
                            wbuf[woff + rel + sr_frag_off(rl, cl, cols)];
                    }
    };
    size_t cur = 0;
    // Region STARTS (the first version used sizes as bases, so every region after Wq was written
    // one slot late - Wk's data at 8192 while the shader read at 4096; Wq alone agreed).
    const size_t rWq = 256u * (size_t)(2u * g * CT);             // one head-unit of Wq tiles
    const size_t wqT = 0;
    const size_t wkT = rWq;                                      // = 256 * (2*g*CT)
    const size_t woT = wkT + rWq;
    const size_t w1T = woT + rWq;
    const size_t w2T = w1T + 256u * (size_t)(2u * NBLK * CT);
    if (s.mode == 2) cur += size_t(4) * C * s.Cn + 8u * C;      // up_W + up_b (not tiled: the up
    cur += 2u * C;                                              // kernel reads wrep directly)
    for (uint32_t h = 0; h < g; ++h) {
        tile(cur, 32, C, 256u * (size_t)(2u * h * CT), false, true);           // Wq: K = C
        tile(cur + 32u * C, 32, C, wkT + 256u * (size_t)(2u * h * CT), false, true);
        cur += size_t(64u) * C + 512u;                          // + relpos (not tiled)
        tile(cur, C, 32, woT + 256u * (size_t)(2u * h * CT), true, false);     // Wo: rows = C
        cur += size_t(32u) * C;
    }
    cur += 4u * C;                                              // v270, gain2 (table only)
    for (uint32_t j = 0; j < NBLK; ++j) {
        size_t b = cur + ((j == 0u) ? 0u : 2u * C);
        tile(b, 32, C, w1T + 256u * (size_t)(2u * j * CT), false, true);       // W1: K = C
        tile(b + 32u * C + 64u, C, 32, w2T + 256u * (size_t)(2u * j * CT), true, false);  // W2: rows = C
        cur += size_t(64u) * C + 64u;
    }
    return true;
}

// Returns false when the walk does not end at the blob size (the stage is then left unrepacked).
bool sr_repack_stage(const uint8_t* wbuf, uint8_t* wrep, size_t woff, size_t blob_size, const SrStageShape& s) {
    size_t cur = 0;
    auto mat = [&](size_t rel, uint32_t rows, uint32_t cols) {
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t c = 0; c < cols; ++c)
                wrep[woff + rel + size_t(r) * cols + c] = wbuf[woff + rel + sr_frag_off(r, c, cols)];
    };
    const uint32_t C = s.C, Cn = s.Cn;
    if (s.mode == 2) { mat(cur, 4 * C, Cn); cur += size_t(4) * C * Cn + 8 * C; }
    cur += 2 * C;                                             // gain1
    for (uint32_t h = 0; h < s.g; ++h) {
        mat(cur, 32, C);                                      // Wq
        mat(cur + 32 * C, 32, C);                             // Wk
        cur += size_t(64) * C + 512;                          // + relpos (16 x 16 f16)
        mat(cur, C, 32);                                      // Wo
        cur += size_t(32) * C;
    }
    cur += 4 * C;                                             // v270, gain2
    for (uint32_t j = 0; j < C / 8; ++j) {
        mat(cur, 32, C);                                      // fc1
        cur += size_t(32) * C + 64;                           // + b1
        mat(cur, C, 32);                                      // fc2
        cur += size_t(32) * C;
        if (j == 0) cur += 2 * C;                             // fc2_bias
    }
    if (s.mode == 0) { mat(cur, Cn, 4 * C); cur += size_t(4) * C * Cn + 2 * Cn; }   // dn_W, dn_b
    return cur == blob_size;
}

// sr_dec0's weights, repacked for the kernel (PERF 2026-10-28). dec0.bin = frag(128 x 64) fp8 at
// sr_frag_off positions + 128 P-permuted f16 biases at byte 8192; the shader walked frag_off() and
// converted an fp8 byte per weight - 1152 indexed byte loads AND 1152 fp8->f32 conversions per
// render pixel, ~16k VALU instructions of the kernel's 22k per wave. The repack turns it into
//   dst[0 .. 16383]  = [q][n][k] f16 weights, q = output sub-pixel 0..3, n = channel 0..31, k = 0..63
//   dst[16384 ..]    = [q][n] f16 biases (the same 128 biases, de-P-permuted)
// every fp8 value is exactly representable in f16, so this is a layout change only, and the shader
// then loads f16vec2 (two k at a time) with a linear index: half the loads, no conversion.
bool sr_repack_dec0(const uint8_t* src, uint8_t* dst, size_t blob_size) {
    if (blob_size < 8192 + 256) return false;
    for (uint32_t q = 0; q < 4; ++q)
        for (uint32_t n = 0; n < 32; ++n) {
            uint16_t* row = reinterpret_cast<uint16_t*>(dst) + (size_t(q) * 32 + n) * 64;
            for (uint32_t k = 0; k < 64; ++k) {
                uint32_t j = q * 32 + n;
                // e4m3 -> f32: the model's LUT (tools/sr/sr_dec0_model.py e4m3_to_f32)
                uint8_t b = src[sr_frag_off(j, k, 64)];
                uint32_t e = (b >> 3) & 15u;
                uint32_t bits = (e == 0u) ? 0u : ((e + 120u) << 23u) | ((b & 7u) << 20u);
                float v; std::memcpy(&v, &bits, 4);
                if (e == 0u) v = float(b & 7u) * 0.001953125f;
                if (b & 0x80u) v = -v;
                row[k] = f32_to_f16_rne(v);
            }
            // bias of output channel j = q*32 + n: byte 8192 + 2*((j & ~31) | perm_invP(n))
            uint32_t invP = (((n >> 1) & 7u) << 2) | (n & 1u) | (((n >> 4) & 1u) << 1);
            size_t bo = 8192 + 2u * ((q * 32u) | invP);
            uint16_t* bias = reinterpret_cast<uint16_t*>(dst) + 8192 + (size_t(q) * 32 + n);
            *bias = uint16_t(src[bo]) | (uint16_t(src[bo + 1]) << 8);
        }
    return true;
}

// sr_dec0's embed matrix for the cooperative-matrix path: dec0.bin's one 128x64 fp8 tile, at
// frag(128,64) offsets, rewritten [N][K] row-major (enc0's sr_repack_enc0 pattern). Binding 7's
// f16 expansion keeps only its biases now.
void sr_repack_dec0f8(const uint8_t* blob, size_t blob_size, uint8_t* dst) {
    // dec0.bin = 8192 fp8 frag bytes + 128 f16 biases = 8448 B (the old `< 16384` guard here
    // silently left the buffer zeroed and cost preset M the whole guard)
    if (blob_size < 8448) return;
    for (uint32_t r = 0; r < 128; ++r)
        for (uint32_t c = 0; c < 64; ++c)
            dst[size_t(r) * 64 + c] = blob[sr_frag_off(r, c, 64)];
}

// sr_enc0's embed matrix: the one 64x128 fp8 tile of enc0.bin, at blob offset 4096 in NeuralHub's
// per-lane fragment order (frag_off, cols = 128), rewritten [N][K] row-major for the cooperative
// matrix kernel's B operand (K contiguous). The 64 f16 biases at blob +12288 stay in w_enc0: the
// shader reads them there with its own P-un permutation.
void sr_repack_enc0(const uint8_t* blob, uint8_t* dst) {
    for (uint32_t r = 0; r < 64; ++r)
        for (uint32_t c = 0; c < 128; ++c)
            dst[size_t(r) * 128 + c] = blob[4096 + sr_frag_off(r, c, 128)];
}

// Preset L's enc0L embedding matrices for sr_l0emb (perf mu_l0): W1s/W2s [32][32] row-major in
// the activation's STORED basis on both axes (Ws[s][t] = frag(perm_P(s), perm_P(t))), so the
// kernel's B loads are K-contiguous and no per-lane permutation is needed.
void sr_repack_l0emb(const uint8_t* blob, uint8_t* dst) {
    auto perm_P = [](uint32_t i) { return ((i >> 2u) << 1u) | (i & 1u) | (((i >> 1u) & 1u) << 4u); };
    for (uint32_t s = 0; s < 32; ++s)
        for (uint32_t t = 0; t < 32; ++t) {
            dst[size_t(s) * 32 + t] = blob[4096 + sr_frag_off(perm_P(s), perm_P(t), 32)];
            dst[1024 + size_t(s) * 32 + t] = blob[5184 + sr_frag_off(perm_P(s), perm_P(t), 32)];
        }
}

// rr_level phase 3a: item 0's per-head Q/V matrices are in NeuralHub's frag_rr order (the byte map
// tools/rr/rr_layout.py checks: FRAG(r, c) with permute_rows); the WMMA B operand wants
// [head][out][k] row-major at the same offsets, so element (r, c) is copied to r*C + c - the
// sr_repack_stage rule.  rr_frag_off mirrors rr_level.comp's frag_off exactly.
static uint32_t rr_frag_off(uint32_t r, uint32_t c, uint32_t cols) {
    uint32_t u = ((r & 0x7FFFFFE1u) | ((r >> 3u) & 2u)) + (r & 0xEu) * 2u;
    return 16u * (u & 15u) + 256u * ((c >> 2u) & 1u) + 8u * ((c >> 3u) & 1u) + 4u * ((c >> 1u) & 1u)
         + 2u * ((c >> 4u) & 1u) + (c & 1u) + 512u * (c >> 5u) + 16u * cols * (u >> 4u);
}
static void rr_repack_mat(const uint8_t* src, uint8_t* dst, size_t base, size_t extent, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || base + size_t(rows) * size_t(cols) > extent) return;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            dst[base + size_t(r) * cols + c] = src[base + rr_frag_off(uint32_t(r), uint32_t(c), uint32_t(cols))];
}
// item 0: G per-head [64 x C] Q/V blocks (each head block carries its own fragment layout).
static void rr_repack_qkv(const uint8_t* src, uint8_t* dst, size_t base, size_t extent, int C, int G) {
    if (C <= 0 || G <= 0 || base + size_t(G) * 64u * size_t(C) > extent) return;
    for (int h = 0; h < G; ++h)
        rr_repack_mat(src, dst, base + size_t(h) * 64u * size_t(C), extent, 64, C);
}
} // namespace

// ---- RR U-Net (preset F): the 16 k_level launches -----------------------------------------------
// Frame 1 has them at launches 5..20 (frame 0 at 4..19).  Template ints, in_ch and the buffer roles
// are decoded from the oracle's launches.jsonl (identical in both frames) and agree with
// tools/rr/rrref/rrref.cpp's level().  Role names are oracle buffer names; "" = the argument is null.
// Args offsets: in +0x10, skip +0x18, out +0x78, out +0x80, in +0x88, in +0x90, out +0x98,
// out +0xa0, out +0xc0 / +0xc8 / +0xd0 / +0xd8.  rr_level binding 2 takes +0x78 (fp8 out; a dummy for
// unit 15, which has none); the heads go to bindings 7..12 (+0x98, +0xa0, +0xc0..+0xd8).
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

    auto add_pipe = [&](const std::string& name, uint32_t bindings, uint32_t push_size, uint32_t subgroup_size = 0,
                        const std::vector<VkDescriptorType>& types = {}) {
        auto spv = load_spv(name);
        if (spv.empty()) {
            std::fprintf(stderr, "Missing shader %s.spv\n", name.c_str());
            return;
        }
        c->pipelines[name] = create_compute_pipeline(dev, spv, bindings, push_size, subgroup_size, types);
    };

    if (mode == "sr") {
        add_pipe("sr_gather", 6, 88);
        add_pipe("sr_gather_mv", 2, 32);
        add_pipe("sr_exposure", 1, 32);
        add_pipe("sr_ae_luma", 2, 48);
        add_pipe("sr_ae_reduce", 2, 32);
        add_pipe("sr_ae_copy", 2, 24);
        add_pipe("sr_enc0", 13, 256);
        add_pipe("sr_stage", 6, 56, 32);   // 64 threads = 2 x wave32: pin the subgroup size (coopmat)
        // Per-(C, mode) specializations of the same source: C and mode as compile-time constants
        // (NR_SR_* defines) unroll every K/N loop and fold the tile-branching - measured 9.42 ->
        // 6.96 ms on the four launches. The chain dispatches these; the dynamic "sr_stage" above
        // stays as the generic artifact (offline O5/O7 keep building and running it).
        add_pipe("sr_stage_c64m0", 6, 56, 32);
        add_pipe("sr_stage_c96m0", 6, 56, 32);
        add_pipe("sr_stage_c96m2", 6, 56, 32);
        add_pipe("sr_stage_c64m2", 6, 56, 32);
        // sr_stage split into the tube-shaped pieces (optimisation phase 2026-10-28): the Swin block
        // generalised from sr_tube_split.comp (32 threads = one wave a 4x4 window, NO subgroup pin -
        // wave32 follows from the workgroup size), plus the mode-0 patch projection and the mode-2
        // up+skip as their own small GEMM dispatches. sr_stage.comp stays for DLSS5_AMD_STAGE_LEGACY=1.
        add_pipe("sr_swin64", 6, 56);   // + wvec (table) + wrepT (tiled weights)
        add_pipe("sr_swin96", 6, 56);
        // Fused builds of the same source (Opus proposal 1): binding 6 is the mode wire -
        // mode 0 writes the projection's low grid there, mode 2 reads the encoder skip.
        add_pipe("sr_swin64m0", 7, 56);
        add_pipe("sr_swin96m0", 7, 56);
        add_pipe("sr_swin64m2", 7, 56);
        add_pipe("sr_swin96m2", 7, 56);
        add_pipe("sr_swin_proj", 4, 56);
        add_pipe("sr_swin_up", 5, 56);
        // per-(C) specializations of the same sources (SR_C define -> constant K/N loops)
        add_pipe("sr_swin_proj64", 4, 56);
        add_pipe("sr_swin_proj96", 4, 56);
        add_pipe("sr_swin_up64", 5, 56);
        add_pipe("sr_swin_up96", 5, 56);
        add_pipe("sr_tube_split", 4, 56);
        // sr_dec0 gained binding 8: t8 bound again as 16-byte vectors for the staging loop
        // (one vector load a half-pixel-plane instead of a byte load a channel).
        // sr_dec0 pinned wave32 (mu_srm 2-wave split, local_size 64): the chain's unpinned
        // pipeline let RADV pick subgroup size 64, where this driver's cooperative-matrix path
        // computes garbage (same failure mode as sr_stage; kernel_run pins 32, which is why
        // fixtures passed while the chain failed - build/agents/mu_srm/progress.md).
        add_pipe("sr_dec0", 10, 160, 32);
        add_pipe("sr_post", 14, 256, 0,
                 // 0-4 SSBO, 5 history SAMPLED image (linear filter, the 5-tap CR's bilinear), 6-8
                 // SSBO, 9 history STORAGE image write, 10-13 SSBO (12 is the unused slot that keeps
                 // the packed views at their pass-2 indices)
                 {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                  VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
        add_pipe("sr_downsample", 4, 88);
        add_pipe("sr_unorm8_table", 1, 0);
        if (preset == "L") {
            // sr_enc0L_in: 12 bindings (11 = expo, as sr_enc0), push 240 B (the shader reads the
            // Enc0Args words up to +0xe8).
            add_pipe("sr_enc0L_in", 12, 240);
            // k_sr_l0: 7 bindings (see sr_l0.comp), push = L0Args' own offsets 0x48..0x67 plus
            // the port's mode word at 0x68 (0 = k_sr_l0<0,2>, 1 = k_sr_l0<2,2> - the template
            // parameter has no slot in the oracle's 104-byte args).
            add_pipe("sr_l0", 7, 112);
            // Preset L's WMMA rewrite (perf mu_l0): both sr_l0 launches are single dispatches
            // now. sr_l0blk_pad (l0_in -> l0_skip + t0: fused embedding + step (a)'s dn
            // projection); sr_l0blk (t8 + l0_skip -> B AND the heads, steps b1+b2). sr_l0dn /
            // sr_l0up / sr_l0heads are undispatched (pipelines kept so O5 stays green; delete
            // when srl rebases).
            add_pipe("sr_l0blk_pad", 7, 64);
            add_pipe("sr_l0blk", 11, 64);         // +6 l0_skip, +7..10 heads outputs (steps b1/b2)
            add_pipe("sr_l0dn", 4, 56);           // undispatched since step (a) (kept for O5)
            add_pipe("sr_l0up", 5, 56);           // undispatched since step (b1)
            add_pipe("sr_l0heads", 5, 24);        // undispatched since step (b2)
        }
    } else {
        // Bytes per texel of each RR gather format code (tools/verify/texfmt.py RR_BPP).
        // rr_gather's row pitch is width * bpp of the ACTUAL format (rrref.cpp); the old
        // code passed only W/H and the shader read every slot as its default format.
        add_pipe("rr_gather", 14, 68);
        add_pipe("rr_ae_block", 2, 16);
        // rr_ae_final.comp declares bindings 0..3 (Accumulator, PreviousAE, NextAE, Exposure); with
        // only 2 in the set layout, bindings 2/3 had no descriptor and the shader addressed garbage
        // - undefined behaviour that is harmless on a native vkCreateDevice and a GPUVM fault on the
        // device the NGX host hands over.  The push range has to cover the shader's block, which
        // starts at offset 24 (scale) and ends at 31 (reset) - 8 bytes left both reading garbage.
        add_pipe("rr_ae_final", 4, 32);
        // rr_enc0_feat: 17 bindings (13 = wbuf, the dq() dither table at blob0's meta[20] offset;
        // 14 = E0c0e history, 15 = E0c12 history, 16 = E0c02 k_level hidden for the c22-family),
        // push 68 B = the oracle's E0Args words 0x00..0x1c (W0/H0/W/H, fc, rst, dither offset,
        // SC0) plus the history words +0x20/+0x24 pj, +0x28/+0x2c bev, +0x30/+0x34 W15/H15,
        // +0x38 SC1, +0x3c/+0x40 pj_prev (the c22-family E0c02 texel offset).
        add_pipe("rr_enc0_feat", 17, 68);
        add_pipe("rr_level", 15, 80, 32);   // +binding 14 = wrep (item 0 row-major); pin wave32:
                                             // 4 waves of the 128-thread WMMA path, and ACO may pick
                                             // wave64 at this VGPR pressure (sr_stage's lesson)
        add_pipe("rr_swin_u5", 14, 80, 32);   // mu_rrl: C=160 G=8 mode-0 rewrite (wave32 pin)
        add_pipe("rr_swin_u0", 14, 80, 32);   // mu_rrl: C=32 G=2 mode-5 (integrated mu_h2 entry + mu_h3 heads)
        add_pipe("rr_swin_u1", 14, 80, 32);   // mu_rrl: C=64 G=2 mode-1 + enc head + down
        add_pipe("rr_swin_u2", 14, 80, 32);   // mu_rrl: C=96 G=4 mode-0 + enc head + down
        add_pipe("rr_swin_u3", 14, 80, 32);   // mu_rrl: C=128 G=4 mode-0 + enc head + down
        add_pipe("rr_swin_u4", 14, 80, 32);   // mu_rrl: C=160 G=8 mode-0 + enc head + down
        add_pipe("rr_swin_u14", 14, 80, 32);  // mu_rrl: C=32 G=2 mode-2 up (tube entry + mu_h2 raw-v1a0)
        add_pipe("rr_swin_u6", 14, 80, 32);   // mu_rrl: C=160 G=8 mode-2 up (tube + mu_h2 raw-v1a0)
        add_pipe("rr_swin_u8", 14, 80, 32);   // mu_rrl: C=128 G=4 mode-2 up (tube + mu_h2 raw-v1a0)
        add_pipe("rr_swin_u10", 14, 80, 32);  // mu_rrl: C=96 G=4 mode-2 up (tube + mu_h2 raw-v1a0)
        add_pipe("rr_swin_u15", 14, 80, 32);  // mu_rrl: C=32 G=2 mode-4 + Recipe A (mu_h1 RR_PRECISE)
        add_pipe("rr_swin_u12", 14, 80, 32);  // mu_rrl: C=64 G=2 mode-2 up (tube entry + mu_h2 raw-v1a0)
        add_pipe("rr_swin_u7", 14, 80, 32);   // mu_rrl: C=160 G=8 mode-3 + dec head (mu_h3)
        add_pipe("rr_swin_u9", 14, 80, 32);   // mu_rrl: C=128 G=4 mode-3 + dec head (mu_h3)
        add_pipe("rr_swin_u11", 14, 80, 32);  // mu_rrl: C=96 G=4 mode-3 + dec head (mu_h3)
        add_pipe("rr_swin_u13", 14, 80, 32);  // mu_rrl: C=64 G=2 mode-3 + dec head (mu_h3)
        add_pipe("rr_post_prep", 4, 40);
        add_pipe("rr_post", 12, 184);
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

    // Descriptor pool. sr_post binds one sampled and one storage image per dispatch (its temporal
    // history), so the pool needs those types too - a set that asks for a type the pool has no
    // descriptors of fails to allocate and the whole dispatch binds nothing.
    VkDescriptorPoolSize dps[3] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
    };
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpci.maxSets = 256;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes = dps;
    vkCreateDescriptorPool(dev, &dpci, nullptr, &c->dpool);

    // Allocate Buffers
    c->in_col = create_buffer(dev, phys, size_t(W) * H * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->in_dep = create_buffer(dev, phys, size_t(W) * H * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // 16 B/px: RR's code 11 (RGBA32F) motion vectors need the full texel (rrref slots).
    c->in_mv_ = create_buffer(dev, phys, size_t(std::max(W, oW)) * std::max(H, oH) * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
    // The two AE ping-pong slots (oracle E0c26 / E0c2a) start at f16 1.0: rrref seeds both on
    // every reset frame before k_ae_block (FUN_180127f10), and frame 0 is a reset.  Unseeded
    // they read as 0, which made ae_final's `exposure[1]` (the previous state) 0 instead of 1.0.
    {
        const uint16_t ae_one = 0x3c00;
        std::memcpy(c->ae_next.mapped, &ae_one, 2);
        std::memcpy(c->ae_exp.mapped, &ae_one, 2);
    }
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

    // sr_post's history as images (PERF pass 4): two W15 x H15 rgba16f images, one sampled through a
    // linear clamp-to-edge sampler (five texture instructions replace twelve buffer loads and the
    // manual bilinear of the 5-tap Catmull-Rom) and one written as a storage image. Both stay in
    // VK_IMAGE_LAYOUT_GENERAL for their whole life; the chain's per-dispatch memory barrier covers
    // every hazard, and they are cleared ONCE here the way srref zeroes the history at allocation
    // (no per-reset clear - see the note above record()).
    if (mode == "sr") {
        for (int i = 0; i < 2; ++i) {
            VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = VK_FORMAT_R16G16B16A16_UINT;
            ici.extent = {uint32_t(W15), uint32_t(H15), 1};
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            vkCreateImage(dev, &ici, nullptr, &c->sr_hist_img[i]);
            VkMemoryRequirements req;
            vkGetImageMemoryRequirements(dev, c->sr_hist_img[i], &req);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (vkAllocateMemory(dev, &mai, nullptr, &c->sr_hist_mem[i]) != VK_SUCCESS) {
                // fall back to any host-visible type rather than leaving the image unbound
                mai.memoryTypeIndex = find_memory_type(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
                vkAllocateMemory(dev, &mai, nullptr, &c->sr_hist_mem[i]);
            }
            vkBindImageMemory(dev, c->sr_hist_img[i], c->sr_hist_mem[i], 0);
            VkImageViewCreateInfo ivci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            ivci.image = c->sr_hist_img[i];
            ivci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ivci.format = VK_FORMAT_R16G16B16A16_UINT;
            ivci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCreateImageView(dev, &ivci, nullptr, &c->sr_hist_view[i]);
        }
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        // The history image is an integer format (16-bit fixed-point tm), which Vulkan only allows
        // with NEAREST filtering - and this kernel never filters through the sampler anyway: the
        // CR's bilinear is computed here from the gathered texels (see sr_post.comp).
        sci.magFilter = VK_FILTER_NEAREST;
        sci.minFilter = VK_FILTER_NEAREST;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.minLod = 0.0f;
        sci.maxLod = 0.0f;
        vkCreateSampler(dev, &sci, nullptr, &c->sr_hist_sampler);

        uint32_t qf = 0;
        {
            uint32_t n = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, nullptr);
            std::vector<VkQueueFamilyProperties> qp(n);
            vkGetPhysicalDeviceQueueFamilyProperties(phys, &n, qp.data());
            for (uint32_t q = 0; q < n; ++q)
                if (qp[q].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = q; break; }
        }
        VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.queueFamilyIndex = qf;
        pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        VkCommandPool pool = VK_NULL_HANDLE;
        vkCreateCommandPool(dev, &pci, nullptr, &pool);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        VkCommandBuffer cbuf = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(dev, &cai, &cbuf);
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cbuf, &bi);
        for (int i = 0; i < 2; ++i) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = 0;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = c->sr_hist_img[i];
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
            VkClearColorValue zero{};
            VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cbuf, c->sr_hist_img[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &rr);
            b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 0, nullptr, 0, nullptr, 1, &b);
        }
        vkEndCommandBuffer(cbuf);
        VkQueue q = VK_NULL_HANDLE;
        vkGetDeviceQueue(dev, qf, 0, &q);
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cbuf;
        vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
        vkQueueWaitIdle(q);
        vkDestroyCommandPool(dev, pool, nullptr);
    }

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

    c->out = create_buffer(dev, phys, size_t(oW) * oH * DLSS5_AMD_OUT_BPP, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->wbuf = create_buffer(dev, phys, 16 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->w_enc0 = create_buffer(dev, phys, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // enc0's 64x128 tile repacked [N][K] row-major for sr_enc0's fp8 WMMA (8192 B).
    c->w_enc0r = create_buffer(dev, phys, 8192, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (mode == "sr" || mode == "rr") c->wrep = create_buffer(dev, phys, 16 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (mode == "sr") c->wvec = create_buffer(dev, phys, 10 * 16384, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (mode == "sr") c->wrepT = create_buffer(dev, phys, 10 * 131072, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->w_dec0 = create_buffer(dev, phys, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    c->w_dec0f8 = create_buffer(dev, phys, 8192, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    // sr_dec0's own repacked weights (see sr_repack_dec0): [4][32][64] f16 + 4x32 f16 biases = 16640 B.
    c->w_dec0rep = create_buffer(dev, phys, 65536, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Preset L's two level-0 tensors (oracle sr_L dump: l0_in 1419264 B = 264x168x32 at 256x144
    // input = (W0+8)*(H0+8)*32, l0_skip 1310720 B = W0*H0*32).  k_sr_enc0L_in writes l0_in (its
    // oracle pointer slot 0x118, which the 280-byte Enc0Args cannot push, so the port's binding 9
    // carries it), k_sr_l0<0,2> reads it and rewrites t0 (t_act[0], stage0's input) plus l0_skip,
    // and k_sr_l0<2,2> reads l0_skip.  They live in the `ue` map (RR's, unused in SR) so no header
    // changes are needed; the destructor already destroys ue.
    if (mode == "sr" && preset == "L") {
        c->ue["l0_in"] = create_buffer(dev, phys, size_t(W0 + 8) * (H0 + 8) * 32,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->ue["l0_skip"] = create_buffer(dev, phys, size_t(W0) * H0 * 32,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        // Perf mu_l0's sr_l0 intermediates: U (mode-2 block input, fp8; the E tensor is
        // fused away), B (mode-2 block output), plus the repacked L0 weights.
        c->ue["l0_up"] = create_buffer(dev, phys, size_t(W0) * H0 * 32,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->ue["l0_blk"] = create_buffer(dev, phys, size_t(W0) * H0 * 32,
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->ue["wL0tab"] = create_buffer(dev, phys, 2 * 16384, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->ue["wL0T"] = create_buffer(dev, phys, 2 * 131072, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->ue["wL0repE"] = create_buffer(dev, phys, 32768, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->ue["wL0repD"] = create_buffer(dev, phys, 24320, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

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
        c->rr_tab = create_buffer(dev, phys, 16 * 32 * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        c->rr_swin_w = create_buffer(dev, phys, 16 * 1024 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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
            // wrep: the same bytes, then every stage's fp8 matrices rewritten row-major (sr_repack_stage).
            std::memcpy(c->wrep.mapped, c->wbuf.mapped, c->wbuf.size);
            for (int i = 0; i < 10; ++i) {
                if (blob[i].empty() || woff[i] + blob[i].size() > c->wbuf.size) continue;
                if (!sr_repack_stage(static_cast<const uint8_t*>(c->wbuf.mapped), static_cast<uint8_t*>(c->wrep.mapped),
                                     woff[i], blob[i].size(), kSrStageShape[i])) {
                    std::fprintf(stderr, "SR stage %d: blob size %zu does not match its layer walk\n", i, blob[i].size());
                }
                if (!sr_fill_vec_table(blob[i].data(), blob[i].size(),
                                       static_cast<uint8_t*>(c->wvec.mapped) + 16384u * i, kSrStageShape[i])) {
                    std::fprintf(stderr, "SR stage %d: vector table walk failed (blob %zu)\n", i, blob[i].size());
                }
                // blob[i] is the stage's own slice (0-based) - unlike wbuf.mapped, which is what
                // sr_repack_stage gets with woff[i]. Passing woff here read ~1.4 MB past the blob
                // (SIGSEGV in FrameChain::create, found with the -O0 -g runner under gdb).
                sr_repack_tiles(blob[i].data(), static_cast<uint8_t*>(c->wrepT.mapped) + 131072u * i, 0, kSrStageShape[i]);
            }
            if (!blob[10].empty() && blob[10].size() <= c->w_enc0.size) {
                std::memcpy(c->w_enc0.mapped, blob[10].data(), blob[10].size());
            }
            if (!blob[10].empty() && blob[10].size() >= 4096 + 8192) {
                sr_repack_enc0(blob[10].data(), static_cast<uint8_t*>(c->w_enc0r.mapped));
            }
            if (!blob[11].empty() && blob[11].size() <= c->w_dec0.size) {
                std::memcpy(c->w_dec0.mapped, blob[11].data(), blob[11].size());
                sr_repack_dec0f8(blob[11].data(), blob[11].size(), static_cast<uint8_t*>(c->w_dec0f8.mapped));
                // sr_dec0's own repacked weights (binding 7 of sr_dec0 only; sr_l0, preset L, keeps
                // reading the original blob out of w_dec0)
                if (!sr_repack_dec0(blob[11].data(), static_cast<uint8_t*>(c->w_dec0rep.mapped), blob[11].size()))
                    std::fprintf(stderr, "SR dec0: blob size %zu is smaller than its 8192+256 layout\n", blob[11].size());
            }
            // Preset L's level-0 repacks (perf mu_l0). blob[10] = enc0L (30464 B, layer at
            // +6272), blob[11] = dec0L (24320 B, layer at +0). Guards check the exact sizes.
            if (preset == "L" && blob[10].size() == 30464 && blob[11].size() == 24320) {
                const uint8_t* enc0L = blob[10].data() + 6272;
                const uint8_t* dec0L = blob[11].data();
                uint8_t* tab = static_cast<uint8_t*>(c->ue["wL0tab"].mapped);
                uint8_t* til = static_cast<uint8_t*>(c->ue["wL0T"].mapped);
                uint8_t* repE = static_cast<uint8_t*>(c->ue["wL0repE"].mapped);
                uint8_t* repD = static_cast<uint8_t*>(c->ue["wL0repD"].mapped);
                sr_repack_tiles(enc0L, til, 0, kSrL0Shape[0]);
                sr_repack_tiles(dec0L, til + 131072, 0, kSrL0Shape[1]);
                if (!sr_fill_vec_table(enc0L, 24192, tab, kSrL0Shape[0]))
                    std::fprintf(stderr, "SR L0: enc0L vector table walk failed\n");
                if (!sr_fill_vec_table(dec0L, 24320, tab + 16384, kSrL0Shape[1]))
                    std::fprintf(stderr, "SR L0: dec0L vector table walk failed\n");
                if (!sr_repack_stage(enc0L, repE + 6272, 0, 24192, kSrL0Shape[0]))
                    std::fprintf(stderr, "SR L0: enc0L row-major walk failed\n");
                if (!sr_repack_stage(dec0L, repD, 0, 24320, kSrL0Shape[1]))
                    std::fprintf(stderr, "SR L0: dec0L row-major walk failed\n");
                sr_repack_l0emb(blob[10].data(), repE);
            } else if (preset == "L") {
                std::fprintf(stderr, "SR L0: unexpected blob sizes %zu %zu (want 30464 24320)\n",
                             blob[10].size(), blob[11].size());
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
                // rr_enc0_feat's SC0 (oracle E0Args +0x200): the blob's f16 scalar, the low half
                // of params.json's enc0_s2598 word (rrref.cpp: E[0xb02] = enc0_s2598, SC0 =
                // f16(E[0xb02] & 0xffff)).  Stashed in unit 0's rr_tab row at slot 27, past the
                // 26 words rr_level reads (rr_level.comp: slots 0..26 only).  SC1 (E0Args
                // +0x204, the c22-family coefficient) is the HIGH half (enc0_s2598 = 0xB309B6F0:
                // low 0xB6F0 = SC0 -0.43359375, high 0xB309 = SC1 -0.2198486328125) -> slot 28.
                auto sc = jget(pj, "enc0_s2598");
                uint32_t* tab = static_cast<uint32_t*>(c->rr_tab.mapped);
                if (!sc.empty()) { tab[27] = uint32_t(sc[0]) & 0xFFFFu; tab[28] = (uint32_t(sc[0]) >> 16) & 0xFFFFu; }
                else std::fprintf(stderr, "rr_enc0_feat: params.json has no enc0_s2598 (SC0/SC1)\n");
            }
            // wrep (rr_level binding 14) = the rr blob copy with the WMMA B operands rewritten
            // [row][col] row-major at the same wbase offsets (frag_rr -> logical).  Item meanings
            // are mode-dependent, so gate on kRrUnits; rr_repack_mat's extent guard skips the -1
            // items params.json marks absent.
            if (c->wrep.mapped && offset > 0) {
                std::memcpy(c->wrep.mapped, c->wbuf.mapped, offset);
                const uint32_t* tabw = static_cast<const uint32_t*>(c->rr_tab.mapped);
                if (tabw) for (int u = 0; u < 16; ++u) {
                    const RrUnit& U = kRrUnits[u];
                    const uint8_t* src = static_cast<const uint8_t*>(c->wbuf.mapped);
                    uint8_t* dst = static_cast<uint8_t*>(c->wrep.mapped);
                    const size_t ub = c->rr_blob_off[u];
                    rr_repack_qkv(src, dst, ub + tabw[u * 32], offset, U.C, U.g);        // item 0: QKV
                    rr_repack_mat(src, dst, ub + tabw[u * 32 + 4], offset, 4 * U.C, U.C); // item 4: fc1
                    rr_repack_mat(src, dst, ub + tabw[u * 32 + 6], offset, U.C, 4 * U.C);     // item 6: fc2 W2
                    rr_repack_mat(src, dst, ub + tabw[u * 32 + 2], offset, U.C, 32 * U.g);   // item 2: Wo
                    if (U.mode == 5) {
                        rr_repack_mat(src, dst, ub + tabw[u * 32 + 16], offset, 64, 64);  // GEMM1
                        rr_repack_mat(src, dst, ub + tabw[u * 32 + 18], offset, U.C, 64); // GEMM2
                    } else if (U.mode == 1) {
                        rr_repack_mat(src, dst, ub + tabw[u * 32 + 10], offset, U.C, 4 * U.in_ch);
                    } else if (U.mode == 2) {
                        rr_repack_mat(src, dst, ub + tabw[u * 32 + 12], offset, 4 * U.C, U.Cup);
                    }
                    if ((U.mode == 0 || U.mode == 1) && U.Cdown > 0)
                        rr_repack_mat(src, dst, ub + tabw[u * 32 + 8], offset, U.Cdown, 4 * U.C);
                }
            }
            auto bc = slurp(wdir + "/bicubic.bin");
            Buffer& e0d02 = c->ue["E0d02"];
            if (!bc.empty() && bc.size() <= e0d02.size) std::memcpy(e0d02.mapped, bc.data(), bc.size());
            // rr_swin repack (mu_rrl): tiled stored-order weights + f32 tables, all 16 units.
            if (c->rr_tab.mapped && c->rr_swin_w.mapped) {
                static const struct { int C, g, mode, Cdown, Cup; } kU[16] = {
                    {32,2,5,0,0}, {64,2,1,96,0}, {96,4,0,128,0}, {128,4,0,160,0},
                    {160,8,0,160,0}, {160,8,0,0,0}, {160,8,2,0,160}, {160,8,3,0,0},
                    {128,4,2,0,160}, {128,4,3,0,0}, {96,4,2,0,128}, {96,4,3,0,0},
                    {64,2,2,0,96}, {64,2,3,0,0}, {32,2,2,0,64}, {32,2,4,0,0}};
                const uint8_t* wb = static_cast<const uint8_t*>(c->wbuf.mapped);
                uint8_t* sw = static_cast<uint8_t*>(c->rr_swin_w.mapped);
                const uint32_t* tab = static_cast<const uint32_t*>(c->rr_tab.mapped);
                size_t moff = 0;
                for (int u = 0; u < 16; ++u) {
                    moff = (moff + 255) & ~size_t(255);
                    c->rrsw_moff[u] = moff;
                    size_t toff = kRrSwinTabBase + size_t(u) * kRrSwinTabMax;
                    const uint32_t* meta = tab + u * 32;
                    float s1d2 = rrsw_f16_to_f32(meta[21] & 0xFFFFu);
                    float s1d4 = rrsw_f16_to_f32(meta[22] & 0xFFFFu);
                    // blob size upper bound: blobs are 256-aligned back to back from rr_blob_off.
                    size_t bsz = (u < 15 ? c->rr_blob_off[u + 1] : c->wbuf.size) - c->rr_blob_off[u];
                    const uint8_t* wblob = wb + c->rr_blob_off[u];
                    uint8_t* wout = sw + moff;
                    float* tout = reinterpret_cast<float*>(sw + toff);
                    size_t need = rrsw_unit_tiles(kU[u].C, kU[u].g, kU[u].Cdown, kU[u].Cup, meta) * 256u;
                    size_t tneed = rrsw_unit_tab(kU[u].C, kU[u].g) * 4u;
                    if (moff + need <= kRrSwinTabBase && toff + tneed <= c->rr_swin_w.size) {
                        if (!rrsw_repack_unit(wblob, bsz, meta, kU[u].C, kU[u].g,
                                              kU[u].Cdown, kU[u].Cup, s1d2, s1d4, wout, tout))
                            std::fprintf(stderr, "rr_swin: unit %d repack out of range\n", u);
                    } else std::fprintf(stderr, "rr_swin: unit %d does not fit (%zu/%zu)\n", u, need, tneed);
                    moff += need;
                }
            }
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
    for (int i = 0; i < 2; ++i) {
        if (sr_hist_view[i]) vkDestroyImageView(dev, sr_hist_view[i], nullptr);
        if (sr_hist_img[i]) vkDestroyImage(dev, sr_hist_img[i], nullptr);
        if (sr_hist_mem[i]) vkFreeMemory(dev, sr_hist_mem[i], nullptr);
    }
    if (sr_hist_sampler) vkDestroySampler(dev, sr_hist_sampler, nullptr);
    for (int i = 0; i < 9; ++i) destroy_buffer(dev, t_act[i]);
    destroy_buffer(dev, out);
    destroy_buffer(dev, wbuf);
    destroy_buffer(dev, w_enc0);
    destroy_buffer(dev, w_enc0r);
    destroy_buffer(dev, w_dec0);
    destroy_buffer(dev, w_dec0f8);
    destroy_buffer(dev, wrep);
    destroy_buffer(dev, wvec);
    destroy_buffer(dev, wrepT);

    for (auto& kv : ue) destroy_buffer(dev, kv.second);
    destroy_buffer(dev, rr_dummy);
    destroy_buffer(dev, rr_tab);
    destroy_buffer(dev, rr_swin_w);

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
        // Preset L's level-0 tensors: the sr_L dump writes 002.l0_in.bin at the enc0L_in point and
        // 003.l0_skip.bin at the l0<0> point; without these targets the enc0/l0 masks have nothing
        // to inject (the flat rows of the 2026-10-27 L leverage table).
        for (const char* nm : {"l0_in", "l0_skip"}) {
            auto it = ue.find(nm);
            if (it != ue.end()) inj_targets[nm] = &it->second;
        }
        // Perf mu_l0's intermediates (dump-only; the oracle has no files for them).
        for (const char* nm : {"l0_up", "l0_blk", "wL0tab", "wL0T", "wL0repE", "wL0repD"}) {
            auto it = ue.find(nm);
            if (it != ue.end()) inj_targets[nm] = &it->second;
        }
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
    // No fc >= 2 guard: injection is gated by the dump's own files (a frame without dumps simply
    // has nothing to slurp), and build/work/multidump now carries frames 0..8 (chain-leverage.md
    // "Instrument update 2026-10-27"), which makes the frames 4-8 segments measurable.
    if (inj_dir.empty()) return false;
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
    std::string base = inj_dir + "/frame" + std::to_string(inj_frame) + "/" + lbuf + ".";
    bool any = false;
    std::string names;
    for (const auto& kv : inj_targets) {
        const std::string& name = kv.first;
        auto bytes = slurp(base + name + ".bin");
        if (bytes.empty()) continue;          // this launch did not write that buffer
        Buffer* target = kv.second;
#if DLSS5_AMD_OUT_F16
        // M2: `out` is f16 now. An f32 SOURCE dump (an old 16 B/px dump or an oracle-format one)
        // arrives at twice the target's width -> convert; an f16 source dump is width-matched.
        std::vector<uint8_t> conv;
        if (target == &out && bytes.size() == size_t(target->size) * 2) {
            const uint32_t* f = reinterpret_cast<const uint32_t*>(bytes.data());
            size_t n = size_t(target->size) / 2;
            conv.resize(size_t(target->size));
            uint16_t* h = reinterpret_cast<uint16_t*>(conv.data());
            for (size_t i = 0; i < n; ++i) {
                float v; std::memcpy(&v, &f[i], 4);
                h[i] = f32_to_f16_rne(v);
            }
            bytes.swap(conv);
        }
#endif
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
        std::fprintf(stderr, "INJECT frame%d %s launch %d <- %s\n", inj_frame, point, launch, names.c_str());
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
                          size_t push_size, uint32_t gx, uint32_t gy, uint32_t gz,
                          const std::vector<DispatchImage>& images) {
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
    VkResult dsr = vkAllocateDescriptorSets(dev, &dsai, &ds);
    if (dsr != VK_SUCCESS) {
        std::fprintf(stderr, "dispatch %s: vkAllocateDescriptorSets failed (%d)\n", name.c_str(), int(dsr));
        return;
    }

    std::vector<VkDescriptorBufferInfo> dbis(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    std::vector<VkDescriptorImageInfo> iis(bufs.size());
    std::vector<size_t> live;                      // indices that actually carry a descriptor
    for (size_t i = 0; i < bufs.size(); ++i) {
        if (bufs[i]) {
            dbis[i].buffer = bufs[i]->buffer;
            dbis[i].offset = 0;
            dbis[i].range = VK_WHOLE_SIZE;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = uint32_t(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = (i < p.dt.size()) ? p.dt[i] : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            if (writes[i].descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) {
                std::fprintf(stderr, "dispatch %s: binding %zu is not a storage buffer in the layout\n",
                             name.c_str(), i);
                return;
            }
            writes[i].pBufferInfo = &dbis[i];
            live.push_back(i);
        } else {
            // a nullptr slot: the image list is keyed by binding number, not by position
            const DispatchImage* im = nullptr;
            for (const auto& cand : images)
                if (cand.binding == uint32_t(i)) { im = &cand; break; }
            if (!im || !im->view) continue;
            iis[i].sampler = im->sampler;
            iis[i].imageView = im->view;
            iis[i].imageLayout = im->layout;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = ds;
            writes[i].dstBinding = uint32_t(i);
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = im->type;
            writes[i].pImageInfo = &iis[i];
            live.push_back(i);
        }
    }
    // vkUpdateDescriptorSets has no way to skip a slot, so write the live ones individually
    for (size_t k : live) vkUpdateDescriptorSets(dev, 1, &writes[k], 0, nullptr);

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

Buffer* FrameChain::rr_buf(const std::string& n) {
    if (n.empty()) return nullptr;
    if (n == "E0be8") return &t_act[0];     // enc0 features: the same bytes as t_act[0]
    if (n == "E0bec") return &bec;          // enc0 bec, read by launch 5 at +0x88
    auto it = ue.find(n);
    return it == ue.end() ? nullptr : &it->second;
}

// f16 bits -> f32 (rr_enc0_feat's SC0 comes out of params.json as the low half of enc0_s2598).
static float f16_bits_to_f32(uint32_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 0x3FFu;
    float v;
    if (e == 0u) v = float(m) * 5.9604644775390625e-08f;
    else if (e == 31u) { uint32_t b = s | 0x7F800000u | (m << 13); std::memcpy(&v, &b, 4); return v; }
    else { uint32_t b = s | ((e + 112u) << 23) | (m << 13); std::memcpy(&v, &b, 4); return v; }
    return (h & 0x8000u) ? -v : v;
}

void FrameChain::record(VkCommandBuffer cmd, const FrameParams& fp) {
    int W = this->W, H = this->H, oW = this->oW, oH = this->oH;
    int W0 = this->W0, H0 = this->H0, L1w = this->L1w, L1h = this->L1h;
    int L2w = this->L2w, L2h = this->L2h, L3w = this->L3w, L3h = this->L3h;
    int W15 = this->W15, H15 = this->H15, hW15 = this->hW15, hH15 = this->hH15;
    int p = this->p, fc = this->fc;
    inj_frame = (fp.frame_index >= 0) ? fp.frame_index : fc;   // injection's true frame key
    int reset = fp.reset;
    int fmt_col = fp.fmt_col, fmt_mv = fp.fmt_mv, fmt_dep = fp.fmt_dep;
    int fmt_nrm = fp.fmt_nrm, fmt_alb = fp.fmt_alb, fmt_spa = fp.fmt_spa, fmt_rgh = fp.fmt_rgh;
    int mvhi = fp.mvhi, ldr = fp.ldr;
    float exposure_val = fp.exposure, pre_val = fp.pre_exposure;
    float jx = fp.jx, jy = fp.jy, jpx = fp.jpx, jpy = fp.jpy;
    float dec0_s0 = this->dec0_s0, dec0_f0 = this->dec0_f0, dec0_f1 = this->dec0_f1;
    const std::string& preset = this->preset;
    const std::string& mode = this->mode;

    // No host-side clear on reset. NeuralHub's launchers (FUN_180101f60, FUN_180116e70, FUN_180117ab0,
    // FUN_180117ee0) never clear prev_mvec / hidden / ihist; reset only reaches the kernels as an argument
    // (Enc0Args +0xd8), and srref zeroes these buffers once at allocation, as create_buffer does here.
    // Clearing them on a later reset frame (frame 4 of the oracle sequence) cost SR M 33.76 -> 53.82 dB min.

    // "hist" pseudo-point (DLSS5_AMD_INJECT_MASK=hist[,points...]): before this frame's first
    // dispatch, copy the ORACLE's temporal state - the dump's start.<name>.bin, the buffer as it
    // stood before launch 0 - over the chain's. After-point injection cannot supply oracle history
    // to a segment that writes that history itself (post writes ihist/dhist/full15), which is why
    // the frames 4-8 single-point rows were non-monotone (docs/notes/chain-leverage.md,
    // "frames 4-8 made measurable"). SR name list; RR needs its own set when its table is attacked.
    if (!inj_dir.empty() && mode == "sr" &&
        std::find(inj_mask.begin(), inj_mask.end(), std::string("hist")) != inj_mask.end()) {
        static const char* kHist[] = {"ihist0", "ihist1", "dhist", "full15",
                                     "hidden0", "hidden1", "prev_mvec0", "prev_mvec1",
                                     "mv_ero", "mv_nondil"};
        std::string hbase = inj_dir + "/frame" + std::to_string(inj_frame) + "/start.";
        bool hany = false;
        std::string hnames;
        for (const char* nm : kHist) {
            auto it = inj_targets.find(nm);
            if (it == inj_targets.end() || !it->second->mapped) continue;
            auto bytes = slurp(hbase + nm + ".bin");
            if (bytes.empty()) continue;
            Buffer* target = it->second;
            std::string key = std::to_string(inj_frame) + "/hist/" + nm;
            auto shi = inj_shadow.find(key);
            if (shi == inj_shadow.end()) {
                Buffer sh = create_buffer(dev, phys, std::min(bytes.size(), target->size),
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                inj_shadow[key] = sh;
                shi = inj_shadow.find(key);
            }
            std::memcpy(shi->second.mapped, bytes.data(),
                        std::min(bytes.size(), shi->second.size));
            VkBufferCopy region{};
            region.size = std::min({bytes.size(), shi->second.size, target->size}) & ~size_t(3);
            if (region.size == 0) continue;
            vkCmdCopyBuffer(cmd, shi->second.buffer, target->buffer, 1, &region);
            hany = true;
            hnames += nm;
            hnames += " ";
        }
        if (hany) {
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            std::fprintf(stderr, "INJECTSTART frame%d hist <- %s\n", inj_frame, hnames.c_str());
        }
    }

    if (mode == "rr") {
        float exp_data[4] = { exposure_val, pre_val, 1.0f / std::max(exposure_val, 1e-4f), 0.0f };
        std::memcpy(expo.mapped, exp_data, 16);
    } else {
        // LDR keeps expo's last value (no exposure launch, HOST-RE FUN_180116c90).
        if (!ldr) {
            uint16_t* expo_ptr = static_cast<uint16_t*>(expo.mapped);
            uint16_t h_exp = f32_to_f16_rne(exposure_val * pre_val);
            expo_ptr[p] = h_exp;
        }
    }

    // The caller has begun cmd; every descriptor set below is freshly allocated.
    vkResetDescriptorPool(dev, dpool, 0);
    g_dispatch_seen = 0;

    if (mode == "sr") {
        // Dump launch numbers per dispatch point (identical in frame 0 and frame 1 dumps).
        // Preset M: enc0 at 2, stage0..stage9 at 3..12, sr_dec0 13, post 14, downsample 15.
        // Preset L inserts k_sr_l0 at launch 3 (l0<0>, between enc0L_in and stage0) and runs
        // k_sr_l0<2,2> at 14 where M has sr_dec0, so stage0..stage9 are 4..13 and
        // dec0/post/downsample are 14/15/16 (oracle sr_L launches.jsonl; the M-based table was
        // what made the 2026-10-27 L leverage rows flat - docs/notes/chain-leverage.md).
        static const int sr_l_M[] = {/*gather*/0, /*enc0*/2, /*stage0*/3, /*stage1*/4,
                                     /*tube0*/5, /*tube1*/6, /*tube2*/7, /*tube3*/8, /*tube4*/9,
                                     /*tube5*/10, /*stage8*/11, /*stage9*/12,
                                     /*dec0*/13, /*post*/14, /*downsample*/15};
        static const int sr_l_L[] = {/*gather*/0, /*enc0L_in*/2, /*stage0*/4, /*stage1*/5,
                                     /*tube0*/6, /*tube1*/7, /*tube2*/8, /*tube3*/9, /*tube4*/10,
                                     /*tube5*/11, /*stage8*/12, /*stage9*/13,
                                     /*l0<2>*/14, /*post*/15, /*downsample*/16};
        const int* sr_l = (preset == "L") ? sr_l_L : sr_l_M;
        // 1. Gather
        {
            uint8_t push[88] = {0};
            // Row pitches are the tightly-packed texel sizes (srref: W*col_bpp etc.;
            // HOST-RE GatherArgs +0x18/+0x1c/+0x20). The old fixed W*8/W*4/W*4 was right
            // only for the canonical fmt_col 0, so every other colour format read the
            // wrong rows (sr_M_col4/6 ~18.4 dB, col5 ~16.0 dB, ldr1-3 ~18.7 dB).
            auto sr_col_bpp = [](int c) {
                static const int t[] = {8, 4, 4, 4, 4, 16, 4};
                return (c >= 0 && c < 7) ? t[c] : 4;
            };
            auto sr_mv_bpp = [](int c) { return (c == 0) ? 4 : 8; };
            uint32_t pcol = uint32_t(W) * uint32_t(sr_col_bpp(fmt_col));
            uint32_t pmv = uint32_t(W) * uint32_t(sr_mv_bpp(fmt_mv));
            uint32_t pdep = uint32_t(W) * 4u;
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
            uint32_t pmv = uint32_t(oW) * uint32_t((fmt_mv == 0) ? 4 : 8);
            std::memcpy(push + 8, &pmv, 4);
            std::memcpy(push + 12, &fmt_mv, 4);
            std::memcpy(push + 24, &oW, 4);
            std::memcpy(push + 28, &oH, 4);
            dispatch(cmd, "sr_gather_mv", {&in_mv_, &g_mvhi}, push, 32, (oW + 15) / 16, (oH + 15) / 16, 1);
        }

        // 2. Exposure (skipped for LDR colour: the host launches nothing and expo
        // keeps its last value, HOST-RE FUN_180116c90; with exposure 1 the value is
        // identical, but the skip keeps history exact when exposure varies)
        if (!ldr) {
            uint8_t push[32] = {0};
            std::memcpy(push + 8, &exposure_val, 4);
            std::memcpy(push + 12, &pre_val, 4);
            std::memcpy(push + 24, &p, 4);
            dispatch(cmd, "sr_exposure", {&expo, &expo}, push, 32, 1, 1, 1);
        }

        // 3. Enc0 (oracle launch 2, k_sr_enc0): Enc0Args at the oracle's own offsets
        //    (tools/sr/srref/srref.cpp `enc0 (FUN_180116e70, preset M)`), 256 bytes of the
        //    280-byte struct: maxPushConstantsSize is 256 on this device, so +0x100 (mvhi) and
        //    beyond cannot be pushed. The shader reads +0x58..+0xfc only; mvhi selects the g_mv
        //    vs g_mvhi buffer at bind time instead. Binding 11 is expo (+0x80), which the
        //    shader reads as enc0[parity] (1.0 when the word is 0).
        {
            uint8_t push[256] = {0};
            std::memcpy(push + 88, &W, 4); std::memcpy(push + 92, &H, 4);
            std::memcpy(push + 96, &W, 4); std::memcpy(push + 100, &H, 4);   // +0x60 render size again
            std::memcpy(push + 104, &W0, 4); std::memcpy(push + 108, &H0, 4);
            std::memcpy(push + 112, &W15, 4); std::memcpy(push + 116, &H15, 4);
            std::memcpy(push + 120, &hW15, 4); std::memcpy(push + 124, &hH15, 4);
            std::memcpy(push + 136, &p, 4);
            std::memcpy(push + 140, &jx, 4);                                 // +0x8c F20x (this frame's jitter)
            std::memcpy(push + 144, &jy, 4);                                 // +0x90 F20y
            std::memcpy(push + 148, &jpx, 4);                                // +0x94 F28x (the previous frame's)
            std::memcpy(push + 152, &jpy, 4);                                // +0x98 F28y
            // +0xbc mvoff: srref's mv_offset_kernel is 0 for unjittered MVs (as on both
            // dumped frames); the push is zero-filled, stated here for the args-parity reader.
            const float mvoff0 = 0.0f;
            std::memcpy(push + 188, &mvoff0, 4); std::memcpy(push + 192, &mvoff0, 4);
            std::memcpy(push + 216, &reset, 4);
            std::memcpy(push + 220, &fc, 4);
            // +0xe0 depinv has no FrameParams source (the sr_post +0xd0 precedent) and stays 0,
            // the oracle's value on both dumped frames; the shader reads neither it nor
            // +0xe4 mvpitch / +0xfc ldr - they are filled for Enc0Args parity.
            std::memcpy(push + 228, &mvpitch, 4);
            std::memcpy(push + 252, &ldr, 4);
            if (preset == "L") {
                // Oracle launch 2 (k_sr_enc0L_in): the port's binding 9 carries the l0_in pointer
                // (the 288-byte Enc0Args cannot push +0x118), so l0_in lands in its own buffer now
                // - t_act[0] is t0, which k_sr_l0<0,2> rewrites from l0_in and stage0 reads.
                // The oracle reads hidden0 (ptr 0x28) on both frames (the dump's own name), not
                // the M ping-pong slot.  Binding 11 = expo (+0x80), read as enc0[parity].
                // The shader computes the motion-energy feature from the preset-L args words the
                // oracle itself passed: +0xd0 SC and +0xe8 k = 2774.359619140625/H (the same
                // literals srref.cpp's `enc0` block pushes on preset L - args parity, the model
                // reads them from the dump: tools/sr/sr_enc0L_model.py load()/parse_args).
                const float enc0L_sc = 0.08225896209478378f;
                const float enc0L_k = 2774.359619140625f / (float)H;
                std::memcpy(push + 208, &enc0L_sc, 4);
                std::memcpy(push + 232, &enc0L_k, 4);
                // One 8x8 tile of the PADDED grid a workgroup (64 lanes = one wave): (W0+8)*(H0+8)
                // = 264*168 = 33x21 tiles on the oracle frames; the shader drops lanes beyond the
                // grid when a dimension is not tile-aligned.
                dispatch(cmd, "sr_enc0L_in", {&g_col, &g_dep, mvhi ? &g_mvhi : &g_mv, &prev_mvec[p], &dhist, &hidden[0],
                                     &prev_mvec[p ^ 1], &mv_ero, &mv_nondil, &ue["l0_in"], &w_enc0, &expo},
                         push, 240, ((W0 + 8) + 7) / 8, ((H0 + 8) + 7) / 8, 1);
                // Oracle launch 3 (k_sr_l0<0,2>): l0_in -> embedding (ReLU clamp [0,448]) -> the
                // C=32 block over the padded (+2) border -> mode-0 down projection -> t0
                // (t_act[0]); l0_skip = the fp8 block output.  L0Args 0x48.. : (W0,H0) level-0
                // grid, (sx,sy) window shift (f0 (2,2), f1 (0,2)), (Wr,Hr) render grid.
                // Perf mu_l0: two WMMA dispatches replace the bit-exact launch (~630 ms).
                {
                    // Step (a): block with fused embedding AND fused down projection
                    // (PAD input): l0_in -> l0_skip + t0. One wave per 2x2 window group
                    // (sr_swin64m0's shape); the dn dispatch and its l0_skip DRAM read are gone.
                    const uint32_t eIW = uint32_t(W0 + 8), eIH = uint32_t(H0 + 8);
                    const int l0sx = (p == 0) ? 2 : 0, l0sy = 2;
                    const uint32_t blo = 6272, bvo = 0;
                    uint8_t bpush[64] = {0};
                    std::memcpy(bpush + 0, &blo, 4); std::memcpy(bpush + 8, &bvo, 4);
                    std::memcpy(bpush + 40, &eIW, 4); std::memcpy(bpush + 44, &eIH, 4);
                    std::memcpy(bpush + 48, &l0sx, 4); std::memcpy(bpush + 52, &l0sy, 4);
                    std::memcpy(bpush + 56, &W0, 4); std::memcpy(bpush + 60, &H0, 4);
                    dispatch(cmd, "sr_l0blk_pad", {&w_enc0, &ue["l0_in"], &ue["l0_skip"],
                                                   &ue["wL0repE"], &ue["wL0tab"], &ue["wL0T"], &t_act[0]},
                             bpush, 64, (W0 + l0sx + 7) / 8, (H0 + l0sy + 7) / 8, 1);
                }
                inject_after(cmd, "l0", 3);
            } else {
                // One workgroup = 4 waves = 4 tiles stacked vertically (H0 is a multiple of 32,
                // so H0/32 is exact): one-wave workgroups cap residency at ~2 waves a SIMD, which
                // measured 4% VALU efficiency on the feature build.
                dispatch(cmd, "sr_enc0", {&g_col, &g_dep, mvhi ? &g_mvhi : &g_mv, &prev_mvec[p], &dhist, &hidden[p],
                                     &prev_mvec[p ^ 1], &mv_ero, &mv_nondil, &t_act[0], &w_enc0, &expo, &w_enc0r}, push, 256, W0 / 8, H0 / 8, 1);
            }
            inject_after(cmd, "enc0", sr_l[1]);
        }

        // 4. Stages 0..9
        static const uint32_t k_stage_w_off[10] = {
            0, 71936, 235520, 419840, 604160, 788480, 972800, 1157120, 1341440, 1505536
        };
        // A/B: DLSS5_AMD_STAGE_LEGACY=1 keeps the fused pinned sr_stage for these four launches.
        static const bool stage_legacy = [] {
            const char* e = std::getenv("DLSS5_AMD_STAGE_LEGACY");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        // A/B: DLSS5_AMD_SWIN_SEP=1 keeps the split chain (sr_swin + sr_swin_proj / sr_swin_up,
        // the pre-proposal-1 dispatches); default = the fused sr_swin*m0/m2 launches.
        static const bool swin_sep = [] {
            const char* e = std::getenv("DLSS5_AMD_SWIN_SEP");
            return e != nullptr && e[0] != '\0' && e[0] != '0';
        }();
        for (int st = 0; st < 10; ++st) {
            uint8_t push[56] = {0};
            uint32_t w_off = k_stage_w_off[st];
            std::memcpy(push + 0, &w_off, 4);
            uint32_t vec_off = 16384u * (uint32_t)st;   // sr_swin's vector table (brief item b)
            std::memcpy(push + 8, &vec_off, 4);
            int dimW, dimH, s0, s1;
            const int pe = (p == 0) * 2, po = (p != 0) * 2;
            if (st == 0) { dimW = L1w; dimH = L1h; s0 = pe; s1 = 2;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                // binding 4 is the mode-2 skip; the oracle passes no offset 16 in mode 0,
                // so bind the stage's own input - the shader never reads it here.
                if (stage_legacy) {
                    // DLSS5_AMD_STAGE_LEGACY=1: main 3c51da6's per-(C,mode) specialised sr_stage
                    // (the A/B bar: 6.32 ms high for the four launches).
                    dispatch(cmd, "sr_stage_c64m0", {&wbuf, &t_act[0], &t_act[1], &t_act[2], &t_act[0], &wrep}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                } else if (swin_sep) {
                    dispatch(cmd, "sr_swin64", {&wbuf, &t_act[0], &t_act[1], &wrep, &wvec, &wrepT}, push, 56,
                             (dimW + s0 + 3) / 4, (dimH + s1 + 3) / 4, 1);
                    dispatch(cmd, "sr_swin_proj64", {&wbuf, &t_act[1], &t_act[2], &wrep}, push, 56,
                             ((dimW >> 1) + 3) / 4, ((dimH >> 1) + 3) / 4, 1);
                } else {
                    // Fused: one wave per 8x8 group (4 windows), projection in the epilogue.
                    // binding 6 = t_act[2], the low grid the epilogue writes (stage 1's input).
                    dispatch(cmd, "sr_swin64m0", {&wbuf, &t_act[0], &t_act[1], &wrep, &wvec, &wrepT, &t_act[2]},
                             push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                }
            } else if (st == 1) { dimW = L2w; dimH = L2h; s0 = pe; s1 = 2;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                if (stage_legacy) {
                    dispatch(cmd, "sr_stage_c96m0", {&wbuf, &t_act[2], &t_act[3], &t_act[4], &t_act[2], &wrep}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                } else if (swin_sep) {
                    dispatch(cmd, "sr_swin96", {&wbuf, &t_act[2], &t_act[3], &wrep, &wvec, &wrepT}, push, 56,
                             (dimW + s0 + 3) / 4, (dimH + s1 + 3) / 4, 1);
                    dispatch(cmd, "sr_swin_proj96", {&wbuf, &t_act[3], &t_act[4], &wrep}, push, 56,
                             ((dimW >> 1) + 3) / 4, ((dimH >> 1) + 3) / 4, 1);
                } else {
                    // Fused: binding 6 = t_act[4], the low grid (tube0's input).
                    dispatch(cmd, "sr_swin96m0", {&wbuf, &t_act[2], &t_act[3], &wrep, &wvec, &wrepT, &t_act[4]},
                             push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                }
            } else if (st <= 7) {
                Buffer* in_b = (st == 2) ? &t_act[4] : (st & 1) ? &t_act[5] : &t_act[6];
                Buffer* out_b = (st & 1) ? &t_act[6] : &t_act[5];
                dimW = L3w; dimH = L3h;
                if ((st & 1) == 0) { s0 = pe; s1 = 2; } else { s0 = po; s1 = 0; }
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                // binding 3 = wrep, the row-major copy of the stage blob (the coopmm B operands)
                dispatch(cmd, "sr_tube_split", {&wbuf, in_b, out_b, &wrep}, push, 56, (dimW + s0 + 3) / 4, (dimH + s1 + 3) / 4, 1);
            } else if (st == 8) { dimW = L2w; dimH = L2h; s0 = po; s1 = 0;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                if (stage_legacy) {
                    // Oracle launch 11: wbuf+1341440, t6 (in), t3 (skip, StageArgs+16), t7 (out).
                    dispatch(cmd, "sr_stage_c96m2", {&wbuf, &t_act[6], &t_act[7], &t_act[7], &t_act[3], &wrep}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                } else if (swin_sep) {
                    // The up's output goes to t2, NOT in place on its low t6: each WG stages its
                    // 16 parents before its own stores, but across WGs there is no ordering - a
                    // later-WG gather reads low bytes an earlier-WG consume store has already
                    // overwritten (measured: t8 garbage, r 0.13; the isolated kernel scores
                    // 44.4 dB vs the model, so the shader itself is right). t2 (L2x96 = exactly
                    // this tensor's size) is dead here: written by stage 0's projection, last
                    // read by the tube chain (launch <= 10).
                    dispatch(cmd, "sr_swin_up96", {&wbuf, &t_act[6], &t_act[3], &t_act[2], &wrep}, push, 56,
                             (dimW + 7) / 8, (dimH + 7) / 8, 1);
                    dispatch(cmd, "sr_swin96", {&wbuf, &t_act[2], &t_act[7], &wrep, &wvec, &wrepT}, push, 56,
                             (dimW + s0 + 3) / 4, (dimH + s1 + 3) / 4, 1);
                } else {
                    // Fused: binding 1 = the low tensor (the prologue gathers from it),
                    // binding 6 = the encoder skip; no t2 scratch round trip any more.
                    dispatch(cmd, "sr_swin96m2", {&wbuf, &t_act[6], &t_act[7], &wrep, &wvec, &wrepT, &t_act[3]},
                             push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                }
            } else { dimW = L1w; dimH = L1h; s0 = po; s1 = 0;
                std::memcpy(push + 40, &dimW, 4); std::memcpy(push + 44, &dimH, 4);
                std::memcpy(push + 48, &s0, 4); std::memcpy(push + 52, &s1, 4);
                if (stage_legacy) {
                    // Oracle launch 12: wbuf+1505536, t7 (in), t1 (skip, StageArgs+16), t8 (out).
                    dispatch(cmd, "sr_stage_c64m2", {&wbuf, &t_act[7], &t_act[8], &t_act[8], &t_act[1], &wrep}, push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                } else if (swin_sep) {
                    // Same scratch rule as stage 8: t0 (>= L1x64, dead after stage 0 read it) holds
                    // the up's output; the block reads it into t8.
                    dispatch(cmd, "sr_swin_up64", {&wbuf, &t_act[7], &t_act[1], &t_act[0], &wrep}, push, 56,
                             (dimW + 7) / 8, (dimH + 7) / 8, 1);
                    dispatch(cmd, "sr_swin64", {&wbuf, &t_act[0], &t_act[8], &wrep, &wvec, &wrepT}, push, 56,
                             (dimW + s0 + 3) / 4, (dimH + s1 + 3) / 4, 1);
                } else {
                    // Fused: binding 1 = the low tensor, binding 6 = the encoder skip; t0 stays
                    // free (it was only the split chain's up-output scratch).
                    dispatch(cmd, "sr_swin64m2", {&wbuf, &t_act[7], &t_act[8], &wrep, &wvec, &wrepT, &t_act[1]},
                             push, 56, (dimW + s0 + 7) / 8, (dimH + s1 + 7) / 8, 1);
                }
            }
            static const char* st_pt[] = {"stage0", "stage1", "tube0", "tube1", "tube2",
                                          "tube3", "tube4", "tube5", "stage8", "stage9"};
            inject_after(cmd, st_pt[st], sr_l[2 + st]);
        }

        // 5. Dec0 (preset M) or k_sr_l0<2,2> (preset L, oracle launch 14: t8 + l0_skip ->
        //    gates/factors/aniso/hidden0 - preset L has no sr_dec0, docs/notes/k_sr_dec0.md)
        {
            if (preset == "L") {
                // L0Args 0x48.. : (W0,H0), window shift (f0 (0,0), f1 (2,0)), (Wr,Hr),
                // +0x60 s0 / +0x64 f0 = params.json dec0_s0 / dec0_f0 (the aniso scales;
                // dec0_f1 is not passed to this kernel), +0x68 the port's mode word.
                // Perf mu_l0: three WMMA dispatches replace the bit-exact k_sr_l0<2,2>.
                const int l2sx = (p != 0) ? 2 : 0;
                {
                    // Step (b1): the up projection + skip are folded into sr_l0blk's prologue
                    // (sr_swin64m2's pass A/pass B) - sr_l0up's dispatch and the l0_up U tensor
                    // round trip are gone. binding 1 = t8 (the low tensor), binding 6 = l0_skip;
                    // one wave per 8x8 group like the fused sr_swin*m2 launches.
                    const uint32_t ulo = 0;
                    const uint32_t bvo = 16384;
                    const int l2sy = 0;
                    uint8_t bpush[64] = {0};
                    std::memcpy(bpush + 0, &ulo, 4); std::memcpy(bpush + 8, &bvo, 4);
                    // Step (b2): render crop + aniso scales for the fused heads (offsets 12..27).
                    std::memcpy(bpush + 12, &W, 4); std::memcpy(bpush + 16, &H, 4);
                    std::memcpy(bpush + 20, &dec0_s0, 4); std::memcpy(bpush + 24, &dec0_f0, 4);
                    std::memcpy(bpush + 40, &W0, 4); std::memcpy(bpush + 44, &H0, 4);
                    std::memcpy(bpush + 48, &l2sx, 4); std::memcpy(bpush + 52, &l2sy, 4);
                    std::memcpy(bpush + 56, &W0, 4); std::memcpy(bpush + 60, &H0, 4);
                    dispatch(cmd, "sr_l0blk", {&w_dec0, &t_act[8], &ue["l0_blk"],
                                               &ue["wL0repD"], &ue["wL0tab"], &ue["wL0T"],
                                               &ue["l0_skip"], &gates, &factors, &aniso, &hidden[0]},
                             bpush, 64, (W0 + l2sx + 7) / 8, (H0 + l2sy + 7) / 8, 1);
                    // 6. heads are folded into the block epilogue (step b2); the dispatch above
                    // writes gates/factors/aniso/hidden0 itself. sr_l0heads stays registered but
                    // undispatched (kept for O5 until srl rebases).
                }
                inject_after(cmd, "dec0", sr_l[12]);
            } else {
            uint8_t push[160] = {0};
            std::memcpy(push + 64, &W, 4); std::memcpy(push + 68, &H, 4);
            std::memcpy(push + 72, &W, 4); std::memcpy(push + 76, &H, 4);
            std::memcpy(push + 80, &W0, 4); std::memcpy(push + 84, &H0, 4);
            std::memcpy(push + 88, &W15, 4); std::memcpy(push + 92, &H15, 4);
            std::memcpy(push + 96, &dec0_s0, 4);
            std::memcpy(push + 100, &dec0_f0, 4);
            // Dec0Args +0x68..+0x9f: the hidden-state warp's frame geometry.  Until 2026-10-26 this
            // block was left zero, so the shader warped with r15w = r15h = 0 (px = mv alone, hence
            // "valid" only where mv was 0) and with 1/W, 1/W15, F28* all zero - the resample read the
            // wrong tap everywhere.  Every field below is reproduced exactly by the oracle's own args
            // on both dumped frames (tools/sr/sr_dec0_model.py's parse_args reads them):
            //   F20x/F20y = this frame's jitter, F28x/F28y = the previous frame's (params_<N>.json
            //   "jitter" / "jitter_prev"); r15 = W15/W, H15/H maps render pixels to the level-15
            //   grid the motion vectors are expressed in; the rest are the reciprocal pairs and the
            //   previous jitter scaled back to render units.  +0x88 (warp enable: 0 = warp) and
            //   +0x9c stay 0, as the oracle's do.
            const float r15w = (float)W15 / (float)W, r15h = (float)H15 / (float)H;
            const float iW = 1.0f / (float)W, iH = 1.0f / (float)H;
            const float i15w = 1.0f / (float)W15, i15h = 1.0f / (float)H15;
            std::memcpy(push + 104, &jx, 4);                 // +0x68 F20x
            std::memcpy(push + 108, &jy, 4);                 // +0x6c F20y
            std::memcpy(push + 112, &jpx, 4);                // +0x70 F28x
            std::memcpy(push + 116, &jpy, 4);                // +0x74 F28y
            std::memcpy(push + 120, &r15w, 4);               // +0x78 r15w
            std::memcpy(push + 124, &r15h, 4);               // +0x7c r15h
            std::memcpy(push + 128, &iW, 4);                 // +0x80 1/W
            std::memcpy(push + 132, &iH, 4);                 // +0x84 1/H
            std::memcpy(push + 140, &i15w, 4);               // +0x8c 1/W15
            std::memcpy(push + 144, &i15h, 4);               // +0x90 1/H15
            const float F28xW = jpx * iW, F28yH = jpy * iH;  // +0x94, +0x98 (frame 1 has F28y/H != 0)
            std::memcpy(push + 148, &F28xW, 4);
            std::memcpy(push + 152, &F28yH, 4);
            // binding 7 = w_dec0rep, sr_repack_dec0's [q][n][k] f16 weights + f16 biases (the
            // original dec0.bin stays in w_dec0 for preset L's sr_l0)
            dispatch(cmd, "sr_dec0", {&t_act[8], &prev_mvec[p ^ 1], &hidden[p], &gates, &factors, &aniso, &hidden[p ^ 1], &w_dec0rep,
                                 &t_act[8], &w_dec0f8}, push, 160, W0 / 8, H0 / 8, 1);
            inject_after(cmd, "dec0", sr_l[12]);
            }
        }

        // 6. Post: NeuralHub's k_sr_post<false,12,16,false,false> (SR M, oracle launch 14) or
        //    <true,12,16,false,false> (SR L, oracle sr_L launch 15): one shader, two template
        //    instances selected by the push's isL at +0xd0 (see shaders/sr/sr_post.comp; the models are
        //    tools/sr/sr_post_model.py / sr_postL_model.py). PERF (2026-10-28): the group is 32 output
        //    columns x 16 row PAIRS, not 16x16 - each thread owns a 2x2 output block, so its four
        //    stores (and its history taps) sit in contiguous memory instead of two rows W15 elements
        //    apart, and the dhist 2x2 block falls inside one thread (no LDS reduction, no barrier).
        //    Measured 12.87 -> 8.5-9.8 ms at 1080p->4K together with the source-tile LDS staging.
        //    The 256-byte push is PostArgs at the
        //    oracle's own offsets (tools/sr/srref/srref.cpp, "post (FUN_180117ee0)"), so an oracle args
        //    hex passes verbatim except the port-only +0xd0 flag (the oracle dumps carry 0 there; both
        //    oracle instances read +0xd0 only as a ==0 test that masks four guards the port does not
        //    implement - with the dumps' 0 it passes everything; the L fixtures patch +0xd0 to 1, mk_fix
        //    pattern). The shader
        //    reads 0x10, 0x88..0xa4, 0xa8, 0xac, 0xb0, 0xb4, 0xb8, 0xcc and 0xd0; every other
        //    int/float offset is filled as the oracle does where the host has the value. The pointers
        //    (0x00..0x87) stay zero: the shader does not read them (the oracle ISA proves neither
        //    instance reads +0xe0 g_mvhi / +0xf0 mv_scale either - docs/notes/k_sr_post.md, l-post
        //    decode). 0xd4 (ldr) and 0xf0 (mv_scale) are not in FrameParams and stay zero. 0xf8 is the
        //    oracle's mv_offset_kernel value, 0 on both dumped frames (srref: "0 for unjittered MVs").
        //    The previous build's stand-in offsets (136..252) are gone. DLSS5_AMD_POST_PUSH_DUMP=<file>
        //    appends each frame's push (256 bytes) to <file>, for the byte-for-byte comparison with the
        //    oracle's args.
        {
            uint8_t push[256] = {0};
            const uint64_t p64 = uint64_t(p);
            std::memcpy(push + 0x10, &p64, 8);
            std::memcpy(push + 0x88, &W, 4); std::memcpy(push + 0x8c, &H, 4);
            std::memcpy(push + 0x90, &W, 4); std::memcpy(push + 0x94, &H, 4);
            std::memcpy(push + 0x98, &W15, 4); std::memcpy(push + 0x9c, &H15, 4);
            std::memcpy(push + 0xa0, &hW15, 4); std::memcpy(push + 0xa4, &hH15, 4);
            std::memcpy(push + 0xa8, &dec0_f1, 4);
            const float fW = float(W), fH = float(H), f15w = float(W15), f15h = float(H15);
            const float sx = fW / f15w, sy = fH / f15h;
            std::memcpy(push + 0xac, &sx, 4); std::memcpy(push + 0xb0, &sy, 4);
            std::memcpy(push + 0xb4, &jx, 4); std::memcpy(push + 0xb8, &jy, 4);
            const float iW = 1.0f / fW, iH = 1.0f / fH, i15w = 1.0f / f15w, i15h = 1.0f / f15h;
            std::memcpy(push + 0xbc, &iW, 4); std::memcpy(push + 0xc0, &iH, 4);
            std::memcpy(push + 0xc4, &i15w, 4); std::memcpy(push + 0xc8, &i15h, 4);
            const uint32_t log2_10 = 0x40549a78u;
            std::memcpy(push + 0xcc, &log2_10, 4);
            const uint32_t isL = (preset == "L") ? 1u : 0u;      // sr_post.comp's port-only instance flag
            std::memcpy(push + 0xd0, &isL, 4);
            std::memcpy(push + 0xd4, &ldr, 4);
            const uint64_t mvhi64 = uint64_t(mvhi);
            std::memcpy(push + 0xd8, &mvhi64, 8);
            std::memcpy(push + 0xe8, &oW, 4); std::memcpy(push + 0xec, &oH, 4);
            if (const char* dump = std::getenv("DLSS5_AMD_POST_PUSH_DUMP")) {   // A3 check: append each frame's push
                if (FILE* f = std::fopen(dump, "ab")) { std::fwrite(push, 1, 256, f); std::fclose(f); }
            }
            // bindings 5 and 9 are the history IMAGES (5 = sampled read, 9 = storage write; they live
            // in VK_IMAGE_LAYOUT_GENERAL for their whole life), so those slots carry nullptr in the
            // buffer vector. Binding 12 is the unused slot that keeps full15's packed view at 13.
            std::vector<DispatchImage> himg = {
                {5, sr_hist_view[p], sr_hist_sampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                 VK_IMAGE_LAYOUT_GENERAL},
                {9, sr_hist_view[p ^ 1], VK_NULL_HANDLE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                 VK_IMAGE_LAYOUT_GENERAL},
            };
            dispatch(cmd, "sr_post", {&g_col, &expo, &gates, &factors, &aniso, nullptr, &prev_mvec[p ^ 1], &mv_ero, &mv_nondil,
                                 nullptr, &full15, &dhist, nullptr, &full15}, push, 256, (W15 + 31) / 32, (hH15 + 15) / 16, 1, himg);
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
            // RR_BPP bytes per texel (tools/verify/texfmt.py); pitch = W * bpp(code).
            static const int kRrBpp[24] = {8,4,4,4,2,4,4,4,1,2,8,16,4,4,8,8,4,4,4,4,2,2,2,1};
            auto pitch = [&](int code) { return uint32_t(W) * uint32_t(kRrBpp[code]); };
            uint8_t push[68] = {0};
            uint32_t pval[7] = {pitch(fmt_col), pitch(fmt_mv), pitch(fmt_dep), pitch(fmt_nrm),
                                pitch(fmt_alb), pitch(fmt_spa), pitch(fmt_rgh)};
            std::memcpy(push + 0, pval, sizeof pval);
            int32_t fval[7] = {fmt_col, fmt_mv, fmt_dep, fmt_nrm, fmt_alb, fmt_spa, fmt_rgh};
            std::memcpy(push + 28, fval, sizeof fval);
            int32_t nu = fp.normals_unorm;
            std::memcpy(push + 56, &nu, 4);
            std::memcpy(push + 60, &W, 4); std::memcpy(push + 64, &H, 4);
            dispatch(cmd, "rr_gather", {&in_col, &in_mv_, &in_dep, &in_norm, &in_alb, &in_spa, &in_rough_, &g_col, &g_dep, &g_mv, &out_norm, &out_alb, &out_spa, &out_rough_}, push, 68, (W + 15) / 16, (H + 15) / 16, 1);
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
            // rr_ae_final's args are rrref's AEArgs (tools/rr/rrref/rrref.cpp): +0x08/+0x0c W,H;
            // +0x18 scale = 2.3283064365386963e-06f / (W*H) (reproduces the oracle's bytes on
            // both dumped frames); +0x1c reset; +0x20 = E0c26 + 4*parity = NEXT; +0x28 = the other
            // slot = PREV; +0x30 = E0c22, the exposure the rest of the chain consumes.  The dump
            // pins the roles: ae_final writes the +0x20 slot on both frames and never +0x28.
            uint8_t push[16] = {0};
            std::memcpy(push + 8, &W, 4);
            std::memcpy(push + 12, &H, 4);
            if (reset) {   // rrref reseeds both slots on every reset, before this launch
                const uint16_t ae_one = 0x3c00;
                std::memcpy(ae_next.mapped, &ae_one, 2);
                std::memcpy(ae_exp.mapped, &ae_one, 2);
            }
            trace_label = "launch " + std::to_string(1 + roff) + " k_ae_block";
            dispatch(cmd, "rr_ae_block", {&g_col, &dhist}, push, 16, (W + 15) / 16, (H + 15) / 16, 1);
            inject_after(cmd, "ae_block", 1 + roff);

            uint8_t push_final[32] = {0};
            std::memcpy(push_final + 8, &W, 4);
            std::memcpy(push_final + 12, &H, 4);
            const float ae_scale = 2.3283064365386963e-06f / float(W * H);
            std::memcpy(push_final + 24, &ae_scale, 4);
            const uint32_t ae_reset = uint32_t(reset);
            std::memcpy(push_final + 28, &ae_reset, 4);
            Buffer* ae_prev = (p == 0) ? &ae_exp : &ae_next;    // +0x28
            Buffer* ae_next_slot = (p == 0) ? &ae_next : &ae_exp; // +0x20
            trace_label = "launch " + std::to_string(2 + roff) + " k_ae_final";
            dispatch(cmd, "rr_ae_final", {&dhist, ae_prev, ae_next_slot, &expo}, push_final, 32, 1, 1, 1);
            inject_after(cmd, "ae_final", 2 + roff);
        }
        // 3. Enc0 feat (launch 3 + roff)
        {
            // Diagnostic (DLSS5_AMD_FEAT_DUMP=<file>, my agent's own tool): t_act[0] still holds
            // the PREVIOUS frame's E0be8 at record time (the runner vkQueueWaitIdle()s per
            // frame, host-coherent) -> append it so the chain's own feature tensor can be
            // scored against the oracle multidump.  Chunk 0 = the zero buffer before frame 0,
            // chunks 1..8 = frames 0..7.  No effect unless the env var is set.
            if (const char* fd = std::getenv("DLSS5_AMD_FEAT_DUMP")) {
                if (FILE* f = std::fopen(fd, "ab")) {
                    std::fwrite(t_act[0].mapped, 1, size_t(W0) * H0 * 64, f);
                    std::fclose(f);
                }
            }
            uint8_t push[68] = {0};
            std::memcpy(push + 0, &W0, 4); std::memcpy(push + 4, &H0, 4);
            std::memcpy(push + 8, &W, 4); std::memcpy(push + 12, &H, 4);
            // E0Args +0x10 fc = E[0x19] & 7 (the dq() dither row; this->fc is the same 8-frame
            // ring), +0x14 rst (the history gate), +0x198's table offset inside binding 13
            // (blob0's meta[20] = 37984, rr_tab slot 20), +0x200 SC0 (params.json enc0_s2598
            // f16[0], stashed at rr_tab slot 27).
            const uint32_t fc7 = uint32_t(fc) & 7u, rst32 = uint32_t(reset);
            const uint32_t* tab = static_cast<const uint32_t*>(rr_tab.mapped);
            const uint32_t dith_off = uint32_t(rr_blob_off[0]) + (tab ? tab[20] : 0u);
            const float sc0 = f16_bits_to_f32(tab ? tab[27] : 0u);
            std::memcpy(push + 16, &fc7, 4);
            std::memcpy(push + 20, &rst32, 4);
            std::memcpy(push + 24, &dith_off, 4);
            std::memcpy(push + 28, &sc0, 4);
            // History branch words (model SPEC 6): pj = E0Args +0x1a0/+0x1a4 = this frame's
            // jitter (the runner's jx/jy, VERIFIED == the oracle args on both dumped frames);
            // bev = the previous frame's mv-record [0,0].xy (the oracle's +0x160 E0bf2/E0bf6
            // parity slot; the port's single bf6 buffer still holds last frame's enc0_feat
            // write here - the runner vkQueueWaitIdle()s per frame and the memory is host
            // coherent, so reading it while recording is race-free; first frame = zeros and
            // rst skips the branch); W15/H15 = the C-grid (oracle E: 768x432).
            // c22-family (SPEC 6b): SC1 = enc0_s2598 high half (rr_tab slot 28); pj_prev =
            // the PREVIOUS frame's jitter (E0Args +0x1a8/0x1ac = the E0c02 texel offset).  The
            // oracle RR params carry no jitter_prev, so the host remembers the last frame's
            // jx/jy in a file static (updated below; frame0 = 0 and rst skips the branch).
            static float s_prev_jx = 0.0f, s_prev_jy = 0.0f;
            const float sc1 = f16_bits_to_f32(tab ? tab[28] : 0u);
            const float pjx_ = jx, pjy_ = jy;
            const float pjpx_ = s_prev_jx, pjpy_ = s_prev_jy;
            const uint16_t* b6 = static_cast<const uint16_t*>(bf6.mapped);
            const float bev_x = b6 ? f16_bits_to_f32(b6[0]) : 0.0f;
            const float bev_y = b6 ? f16_bits_to_f32(b6[1]) : 0.0f;
            std::memcpy(push + 32, &pjx_, 4);
            std::memcpy(push + 36, &pjy_, 4);
            std::memcpy(push + 40, &bev_x, 4);
            std::memcpy(push + 44, &bev_y, 4);
            std::memcpy(push + 48, &W15, 4);
            std::memcpy(push + 52, &H15, 4);
            std::memcpy(push + 56, &sc1, 4);
            std::memcpy(push + 60, &pjpx_, 4);
            std::memcpy(push + 64, &pjpy_, 4);
            s_prev_jx = jx; s_prev_jy = jy;          // remember for the next frame's pj_prev
            trace_label = "launch " + std::to_string(3 + roff) + " k_enc0_feat";
            dispatch(cmd, "rr_enc0_feat", {&g_col, &g_mv, &g_dep, &out_norm, &out_alb, &out_spa, &out_rough_, &expo,
                                     &t_act[0], &bec, &bf6, &bfa, &bfe, &wbuf, &c0e, &ue.at("E0c12"),
                                     &ue.at("E0c02")}, push, 68,
                     (W0 + 15) / 16, (H0 + 15) / 16, 1);
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

            uint8_t push[80] = {0};
            const int ints[10] = {uw, uh, sx, sy, U.mode, U.C, U.g, U.Cdown, U.Cup, U.n};
            std::memcpy(push + 0, ints, sizeof ints);          // offsets 0..39
            const int in_ch = U.in_ch, blob_off = int(rr_blob_off[u]), unit_idx = u;
            std::memcpy(push + 56, &in_ch, 4);                  // offset 56: channels per token of binding 1
            std::memcpy(push + 60, &blob_off, 4);               // offset 60: this unit's weights in binding 0
            std::memcpy(push + 64, &unit_idx, 4);               // offset 64: this unit's row of binding 13 (rr_tab)
            // offsets 68 / 72: the render grid (Args +0xe0 / +0xe4, unit 15's render outputs); 76: head flags
            const int rr_wr = W, rr_hr = H;
            const int rr_flags = (U.role[R_O98][0] ? 1 : 0) | (U.role[R_OA0][0] ? 2 : 0);
            std::memcpy(push + 68, &rr_wr, 4); std::memcpy(push + 72, &rr_hr, 4);
            std::memcpy(push + 76, &rr_flags, 4);

            Buffer* in = rr_buf(U.role[R_IN]);
            Buffer* o78 = rr_buf(U.role[R_O78]);
            Buffer* o98 = rr_buf(U.role[R_O98]);
            Buffer* primary = o78 ? o78 : &rr_dummy;            // binding 2: fp8 out (unit 15 writes none)
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
                &rr_tab, &wrep,
            };
            char lbl[96];
            std::snprintf(lbl, sizeof lbl, "launch %d k_level<%d,%d,%d,%d,%d,%d>", u + 4 + roff,
                          U.C, U.g, U.mode, U.Cdown, U.Cup, U.n);
            trace_label = lbl;
            // mu_rrl: units on the from-scratch rr_swin path (binding 0 = tiled weights
            // at rrsw_moff[u], binding 13 = the f32 table buffer). u0/u5 integrated;
            // u15 parked on the old path until mu_h1's precision diff lands (HANDOFF.md).
            // The rest follow as helpers deliver (HANDOFF.md).
            const char* swin_pipe = nullptr;  /* hybrid P1: u0-u5 only (guard-clean) */
            if (u == 0) swin_pipe = "rr_swin_u0";
            else if (u == 5) swin_pipe = "rr_swin_u5";
            else if (u == 1) swin_pipe = "rr_swin_u1";
            else if (u == 2) swin_pipe = "rr_swin_u2";
            else if (u == 3) swin_pipe = "rr_swin_u3";
            else if (u == 4) swin_pipe = "rr_swin_u4";
            else if (u == 14) swin_pipe = "rr_swin_u14";
            else if (u == 12) swin_pipe = "rr_swin_u12";
            else if (u == 7) swin_pipe = "rr_swin_u7";
            else if (u == 9) swin_pipe = "rr_swin_u9";
            else if (u == 11) swin_pipe = "rr_swin_u11";
            else if (u == 13) swin_pipe = "rr_swin_u13";
            else if (u == 6) swin_pipe = "rr_swin_u6";
            else if (u == 8) swin_pipe = "rr_swin_u8";
            else if (u == 10) swin_pipe = "rr_swin_u10";
            else if (u == 15) swin_pipe = "rr_swin_u15";
            if (swin_pipe) {
                std::vector<Buffer*> bufs5 = {
                    &rr_swin_w, in, primary,
                    or_in(rr_buf(U.role[R_SKIP])),
                    or_dummy(rr_buf(U.role[R_O80])),
                    or_in(rr_buf(U.role[R_I88])), or_in(rr_buf(U.role[R_I90])),
                    or_dummy(o98), or_dummy(rr_buf(U.role[R_OA0])),
                    or_dummy(rr_buf(U.role[R_OC0])), or_dummy(rr_buf(U.role[R_OC8])),
                    or_dummy(rr_buf(U.role[R_OD0])), or_dummy(rr_buf(U.role[R_OD8])),
                    &rr_swin_w,
                };
                uint8_t push5[80];
                std::memcpy(push5, push, 80);
                const int swin_off = int(rrsw_moff[u]);
                std::memcpy(push5 + 60, &swin_off, 4);
                dispatch(cmd, swin_pipe, bufs5, push5, 80, gx, gy, 1);
            } else {
                dispatch(cmd, "rr_level", bufs, push, 80, gx, gy, 1);
            }
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
        // 6. Post (launch 21 + roff): NeuralHub's rrio::k_post<24>. The 184-byte push is PostArgs at
        //    the oracle's own field offsets (tools/rr/rrref/rrref.cpp:533-547), so an oracle
        //    launch's args hex passes verbatim; the shader reads 0x00..0x1c and 0x34. The bindings
        //    are the PostArgs pointer slots 0x40..0xa0 in order: taps, exposure, factors, the mv
        //    record (0x58 = parity-selected E0bf6|E0bf2, both the port's bf6), E0bfa, E0bfe, the
        //    history READ (0x70 = E0c06|E0c0a = ihist[p^1]), the E0d02 Catmull-Rom table, the
        //    history WRITE (0x88 = ihist[p]), E0cf6, E0cfa, E0c0e. Grid: the oracle's own
        //    [(W15+15)/16, ((H15+1)/2+15)/16] pairs of output rows at 16x16.
        {
            uint8_t push[184] = {0};
            std::memcpy(push + 0x00, &W, 4);
            std::memcpy(push + 0x04, &H, 4);
            std::memcpy(push + 0x08, &W15, 4);
            std::memcpy(push + 0x0c, &H15, 4);
            std::memcpy(push + 0x10, &reset, 4);
            const float sx = float(W) / float(W15);
            const float sy = float(H) / float(H15);
            std::memcpy(push + 0x14, &sx, 4);
            std::memcpy(push + 0x18, &sy, 4);
            std::memcpy(push + 0x1c, &jx, 4);
            std::memcpy(push + 0x20, &jy, 4);
            const float iW = 1.0f / float(W), iH = 1.0f / float(H);
            const float i15w = 1.0f / float(W15), i15h = 1.0f / float(H15);
            std::memcpy(push + 0x24, &iW, 4);            // the shader does not read 0x24..0x33,
            std::memcpy(push + 0x28, &iH, 4);            // they are filled as the oracle does
            std::memcpy(push + 0x2c, &i15w, 4);
            std::memcpy(push + 0x30, &i15h, 4);
            // PostArgs +0x34: the packer's third dec0 scalar, exp(f16 table 0x1840) of blob15
            // (tools/rr/rr_pack.py:126-130; units[15].dec0_scalars[2] of params.json =
            // 1.126047134399414, the value both oracle dumps carry at 0x34). k_post's conf needs
            // it; rr_tab only carries the first two scalars, so it is stated here.
            const float dec0_post = 1.126047134399414f;
            std::memcpy(push + 0x34, &dec0_post, 4);
            const int pitch = W15;                       // +0x38 = E[0xbf1], the texture pitch
            std::memcpy(push + 0x38, &pitch, 4);
            trace_label = "launch " + std::to_string(21 + roff) + " k_post";
            dispatch(cmd, "rr_post",
                     {&cf2, &expo, &ue.at("E0c1a"), &bf6, &bfa, &bfe, &ihist[p ^ 1],
                      &ue.at("E0d02"), &ihist[p], &cf6, &cfa, &c0e},
                     push, 184, (W15 + 15) / 16, (((H15 + 1) / 2) + 15) / 16, 1);
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
