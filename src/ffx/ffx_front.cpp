// ffx_front.cpp - DLSS5-AMD (FFX): the AMD FidelityFX API surface in front of the repo's
// Vulkan chain.  SKELETON: the five ffx* exports exist, validate their descriptors enough to
// return the right status code, and log; the chain is NOT wired up yet (see
// docs/notes/ffx-seam-plan.md section 9 for the implementation order).
//
// Provenance / licence:
//   - The FidelityFX API declarations used here (structure layouts, descriptor type ids, return
//     codes, flag values) come from AMD's FidelityFX SDK API headers - FidelityFX SDK 1.1.4 /
//     fsr3-v3.0.x (ffx_api.h, ffx_api_types.h, ffx_upscale.h, ffx_api_dx12.h), MIT licence,
//     https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK.  The subset needed is
//     reproduced below with the licence notice kept; when src/ffx/ffx-api/ lands (plan step 2)
//     these local definitions are replaced by those headers.
//   - Everything else is this repo's own work (same licence as the repository, LICENSE).
//   - No WrongGPU material is used or imitated; the seam behaviour is designed against the
//     public AMD headers and OptiScaler v0.9.4's documented probing (docs/notes/wronggpu-re.md
//     is a behavioural study of a third-party binary, not a source to copy from).
//
// Contract summary implemented here (v0.9.4 consumer: OptiScaler FFXFeatureDx12):
//   ffxCreateContext  - walk the pNext chain, require UPSCALE + BACKEND_DX12, validate sizes.
//   ffxDestroyContext - tolerate NULL/foreign contexts.
//   ffxConfigure      - GLOBALDEBUG1 accepted, UPSCALE_KEYVALUE tuning keys accepted as no-ops,
//                       everything else UNKNOWN_DESCTYPE.
//   ffxQuery          - GET_VERSIONS two-call protocol (one version, parseable name), provider
//                       version, quality-mode ratio/render-res tables, jitter phase/offset,
//                       resource requirements; GPU memory usage reported as zeros (v1).
//   ffxDispatch       - UPSCALE validated and refused with ERROR (chain wiring pending);
//                       GENERATE_REACTIVE_MASK answered OK (nothing recorded), like the
//                       upstream FSR runtime does when it has no reactive pass.
//
// Provider label: "DLSS5-AMD (FFX)" (exported for tooling; nothing in OptiScaler v0.9.4
// resolves it - same pattern other front ends use for their own diagnostics).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <new>

// Step 3 (2026-11-01): the FFX surface now drives the VERY SAME bridge the NGX surface runs -
// one dxgi_to_vk map, one interop dance, one record path.  The shared declarations live in
// src/pe/ngx_bridge_shared.h; nvngx.cpp (static -> extern'd) provides the definitions.  This
// TU is compiled TOGETHER with src/vk/pipeline.cpp + src/pe/nvngx.cpp into dlss5-amd-ffx.dll
// (src/ffx/build.sh), so the NGX surface's machinery is linked, not duplicated.
#include "../pe/ngx_bridge_shared.h"
#include <d3d12.h>

// ---------------------------------------------------------------------------
// FidelityFX SDK API subset (MIT, AMD FidelityFX SDK; see the provenance note above).
// ---------------------------------------------------------------------------
typedef void* ffxContext;
typedef uint32_t ffxReturnCode_t;
typedef uint64_t ffxStructType_t;

enum FfxApiReturnCodes {
    FFX_API_RETURN_OK                             = 0,
    FFX_API_RETURN_ERROR                          = 1,
    FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE         = 2,
    FFX_API_RETURN_ERROR_RUNTIME_ERROR            = 3,
    FFX_API_RETURN_NO_PROVIDER                    = 4,
    FFX_API_RETURN_ERROR_MEMORY                   = 5,
    FFX_API_RETURN_ERROR_PARAMETER                = 6,
    FFX_API_RETURN_PROVIDER_NO_SUPPORT_NEW_DESCTYPE = 7,
};

#define FFX_API_EFFECT_MASK 0xffff0000u
#define FFX_API_EFFECT_ID_GENERAL 0x00000000u

struct ffxApiHeader {
    ffxStructType_t      type;
    struct ffxApiHeader* pNext;
};
typedef ffxApiHeader ffxCreateContextDescHeader;
typedef ffxApiHeader ffxConfigureDescHeader;
typedef ffxApiHeader ffxQueryDescHeader;
typedef ffxApiHeader ffxDispatchDescHeader;

typedef void (*ffxApiMessage)(uint32_t type, const wchar_t* message);

#define FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1 0x0000001u
struct ffxConfigureDescGlobalDebug1 {
    ffxConfigureDescHeader header;
    ffxApiMessage          fpMessage;
    uint32_t               debugLevel;
};

#define FFX_API_QUERY_DESC_TYPE_GET_VERSIONS 4u
struct ffxQueryDescGetVersions {
    ffxQueryDescHeader header;
    uint64_t           createDescType;
    void*              device;
    uint64_t*          outputCount;
    uint64_t*          versionIds;
    const char**       versionNames;
};

#define FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION 6u
struct ffxQueryGetProviderVersion {
    ffxQueryDescHeader header;
    uint64_t           versionId;
    const char*        versionName;
};

#define FFX_API_DESC_TYPE_OVERRIDE_VERSION 5u
struct ffxOverrideVersion {
    ffxApiHeader header;
    uint64_t     versionId;
};

#define FFX_API_EFFECT_ID_UPSCALE 0x00010000u
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE 0x00010000u
struct FfxApiDimensions2D { uint32_t width, height; };
struct FfxApiFloatCoords2D { float x, y; };

struct ffxCreateContextDescUpscale {
    ffxCreateContextDescHeader header;
    uint32_t                   flags;
    struct FfxApiDimensions2D  maxRenderSize;
    struct FfxApiDimensions2D  maxUpscaleSize;
    ffxApiMessage              fpMessage;
};

// enum FfxApiCreateContextUpscaleFlags (ffx_upscale.h)
enum {
    FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE                 = (1 << 0),
    FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS  = (1 << 1),
    FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION = (1 << 2),
    FFX_UPSCALE_ENABLE_DEPTH_INVERTED                     = (1 << 3),
    FFX_UPSCALE_ENABLE_DEPTH_INFINITE                     = (1 << 4),
    FFX_UPSCALE_ENABLE_AUTO_EXPOSURE                      = (1 << 5),
    FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION                 = (1 << 6),
    FFX_UPSCALE_ENABLE_DEBUG_CHECKING                     = (1 << 7),
    FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE              = (1 << 8),
    FFX_UPSCALE_ENABLE_DEBUG_VISUALIZATION                = (1 << 9),
};

#define FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12 0x0000002u
struct ffxCreateBackendDX12Desc {
    ffxCreateContextDescHeader header;
    void*                      device;  // ID3D12Device* (kept void* so the skeleton needs no d3d12.h)
};

#define FFX_API_DISPATCH_DESC_TYPE_UPSCALE 0x00010001u
struct FfxApiResource {
    void* resource;
    struct FfxApiResourceDescription* description;  // simplified for the skeleton; see below
    uint32_t state;
};
// NOTE: the real FfxApiResource embeds FfxApiResourceDescription by value (ffx_api_types.h);
// the skeleton does not touch resources yet, so this translation unit keeps a pointer only.
// The real struct lands with step 3 (ffx_d3d12.cpp) and this file switches to the vendored
// headers wholesale.

struct ffxDispatchDescUpscale {
    ffxDispatchDescHeader      header;
    void*                      commandList;
    struct FfxApiResource      color;
    struct FfxApiResource      depth;
    struct FfxApiResource      motionVectors;
    struct FfxApiResource      exposure;
    struct FfxApiResource      reactive;
    struct FfxApiResource      transparencyAndComposition;
    struct FfxApiResource      output;
    struct FfxApiFloatCoords2D jitterOffset;
    struct FfxApiFloatCoords2D motionVectorScale;
    struct FfxApiDimensions2D  renderSize;
    struct FfxApiDimensions2D  upscaleSize;
    bool                       enableSharpening;
    float                      sharpness;
    float                      frameTimeDelta;
    float                      preExposure;
    bool                       reset;
    float                      cameraNear;
    float                      cameraFar;
    float                      cameraFovAngleVertical;
    float                      viewSpaceToMetersFactor;
    uint32_t                   flags;
};

#define FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK 0x00010006u
struct ffxDispatchDescUpscaleGenerateReactiveMask {
    ffxDispatchDescHeader     header;
    void*                     commandList;
    struct FfxApiResource     colorOpaqueOnly;
    struct FfxApiResource     colorPreUpscale;
    struct FfxApiResource     outReactive;
    struct FfxApiDimensions2D renderSize;
    float                     scale;
    float                     cutoffThreshold;
    float                     binaryValue;
    uint32_t                  flags;
};

#define FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE 0x00010007u
struct ffxConfigureDescUpscaleKeyValue {
    ffxConfigureDescHeader  header;
    uint64_t                key;
    uint64_t                u64;
    void*                   ptr;
};

// enum FfxApiConfigureUpscaleKey (ffx_upscale.h)
enum {
    FFX_API_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR           = 0,
    FFX_API_CONFIGURE_UPSCALE_KEY_FREACTIVENESSSCALE        = 1,
    FFX_API_CONFIGURE_UPSCALE_KEY_FSHADINGCHANGESCALE       = 2,
    FFX_API_CONFIGURE_UPSCALE_KEY_FACCUMULATIONADDEDPERFRAME = 3,
    FFX_API_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION = 4,
};

#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE 0x00010002u
struct ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode {
    ffxQueryDescHeader header;
    uint32_t           qualityMode;
    float*             pOutUpscaleRatio;
};

#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE 0x00010003u
struct ffxQueryDescUpscaleGetRenderResolutionFromQualityMode {
    ffxQueryDescHeader header;
    uint32_t           displayWidth;
    uint32_t           displayHeight;
    uint32_t           qualityMode;
    uint32_t*          pOutRenderWidth;
    uint32_t*          pOutRenderHeight;
};

#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT 0x00010004u
struct ffxQueryDescUpscaleGetJitterPhaseCount {
    ffxQueryDescHeader header;
    uint32_t           renderWidth;
    uint32_t           displayWidth;
    int32_t*           pOutPhaseCount;
};

#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET 0x00010005u
struct ffxQueryDescUpscaleGetJitterOffset {
    ffxQueryDescHeader header;
    int32_t            index;
    int32_t            phaseCount;
    float*             pOutX;
    float*             pOutY;
};

#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS 0x0001000au
struct ffxQueryDescUpscaleGetResourceRequirements {
    ffxQueryDescHeader header;
    uint64_t           required_resources;
    uint64_t           optional_resources;
};

// enum FfxApiQueryResourceIdentifiers (ffx_upscale.h)
enum {
    FFX_API_QUERY_RESOURCE_INPUT_COLOR                   = (1 << 0),
    FFX_API_QUERY_RESOURCE_INPUT_DEPTH                   = (1 << 1),
    FFX_API_QUERY_RESOURCE_INPUT_MV                      = (1 << 2),
    FFX_API_QUERY_RESOURCE_INPUT_EXPOSURE                = (1 << 3),
    FFX_API_QUERY_RESOURCE_INPUT_REACTIVEMASK            = (1 << 4),
    FFX_API_QUERY_RESOURCE_INPUT_TRANSPARENCYCOMPOSITION = (1 << 5),
};

#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE 0x00010008u
struct FfxApiEffectMemoryUsage { uint64_t totalUsageInBytes, aliasableUsageInBytes; };
struct ffxQueryDescUpscaleGetGPUMemoryUsage {
    ffxQueryDescHeader header;
    struct FfxApiEffectMemoryUsage* gpuMemoryUsageUpscaler;
};

// ---------------------------------------------------------------------------
// Logging: the SHARED bridge log (ngx_log writes stderr + the CWD's dlss5-amd-ngx.log, the
// one file the harness gates already grep).  Every line this surface writes is prefixed
// "[ffx] " so one surface's lines are grep-able exactly (ngx_bridge_shared.h contract).
// The skeleton's no-op local ffx_log is gone; ngx_log's varargs replace it.
// ---------------------------------------------------------------------------
static void ffx_log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void ffx_log(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char buf[512];
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    ngx_log("[ffx] %s", buf);
}

// ---------------------------------------------------------------------------
// Version identity.  The name's leading triple must parse (OptiScaler calls
// feature_version::parse_version on versionNames[0], FfxApi_Proxy.h:956/1017).  3.1.x keeps
// the v0.9.4 host's 4.x-only quirk paths off us (docs/notes/ffx-seam-plan.md section 3).
// ---------------------------------------------------------------------------
static const char kProviderName[] = "3.1.200 DLSS5-AMD (FFX)";
static const char kProviderLabel[] = "DLSS5-AMD (FFX)";
static const uint64_t kProviderVersionId = 200;

// FSR quality-mode ratios (ffx_upscale.h enum order).
static const float kQualityRatio[5] = { 1.0f, 1.5f, 1.7f, 2.0f, 3.0f };

// Published FSR jitter-phase arithmetic (Halton(2,3) phases per axis; phase count from the
// render/display ratio).  Kept local to the skeleton so ffxQuery is already honest.
static int32_t jitter_phase_count(uint32_t renderWidth, uint32_t displayWidth) {
    if (renderWidth == 0 || displayWidth == 0) return -1;
    const double ratio = (double)displayWidth / (double)renderWidth;
    if (ratio <= 1.0) return 1;
    return (int32_t)(2.0 * (int)(ratio + 0.5));  // 1.5x -> 8, 1.7x -> 8, 2.0x -> 10, 3.0x -> 14
}

static void halton_jitter(int32_t index, int32_t phaseCount, float* outX, float* outY) {
    // Halton sequence bases 2 and 3, centred (the standard FSR jitter sequence).
    auto halton = [](int32_t i, int32_t b) -> float {
        float f = 1.0f, r = 0.0f;
        while (i > 0) {
            f /= (float)b;
            r += f * (float)(i % b);
            i /= b;
        }
        return r;
    };
    if (index < 0) index += phaseCount;
    const float x = halton(index + 1, 2) - 0.5f;
    const float y = halton(index + 1, 3) - 0.5f;
    if (outX) *outX = x;
    if (outY) *outY = y;
}

// ---------------------------------------------------------------------------
// Context object: descriptor snapshot + the BRIDGE feature state.  The chain is built lazily
// on the first dispatch (build-then-frame: FFX hands us the device in create, but the real
// acquire/format evidence only exists at the first frame's resources; nvngx.cpp builds the
// chain inside CreateFeature and reacts to resize by destroy+recreate - FFX has no resize
// desc, the plan's lifecycle section pins the same answer here).
// ---------------------------------------------------------------------------
struct FfxUpscaleContext {
    uint32_t                     flags = 0;
    FfxApiDimensions2D           maxRenderSize = {};
    FfxApiDimensions2D           maxUpscaleSize = {};
    ffxApiMessage                fpMessage = nullptr;
    void*                        d3d12Device = nullptr;
    unsigned long long           creates = 0;
    // The bridge side (ngx_bridge_shared.h FeatureState_): chain + interop + diag counters.
    FeatureState_                fs = {};
    // The frame parameters the NEXT dispatch feeds the chain (jitter pair: the caller hands
    // the current jitter only, the chain wants prev+cur - same trick nvngx.cpp's Evaluate does).
    bool have_prev_j = false;
    float prev_jx = 0.0f, prev_jy = 0.0f;
    // MV scale observed (S6 observability parity: one line per feature at the first non-identity).
    float mvx = 1.0f, mvy = 1.0f;
    // One-time chain-build failure (so a permanently broken config doesn't retry every frame).
    bool build_failed = false;
    char build_err[192] = "";
};

static FfxUpscaleContext* as_ctx(ffxContext* c) { return c ? (FfxUpscaleContext*)*c : nullptr; }

// ---------------------------------------------------------------------------
// The five exports.
// ---------------------------------------------------------------------------
extern "C" {

__declspec(dllexport) ffxReturnCode_t __stdcall
ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const void* memCb) {
    if (!context || !desc) {
        ffx_log("ffxCreateContext: null pointer");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    // Walk the chain; require UPSCALE + BACKEND_DX12, take the first of each.
    bool haveUpscale = false, haveBackend = false;
    ffxApiHeader* h = (ffxApiHeader*)desc;
    ffxCreateContextDescUpscale* up = nullptr;
    ffxCreateBackendDX12Desc* be = nullptr;
    int seen = 0;
    while (h && seen < 16) {
        if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE) {
            if (!haveUpscale) { up = (ffxCreateContextDescUpscale*)h; haveUpscale = true; }
        } else if (h->type == FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12) {
            if (!haveBackend) { be = (ffxCreateBackendDX12Desc*)h; haveBackend = true; }
        }
        h = h->pNext;
        ++seen;
    }
    if (!haveUpscale || !haveBackend || !up || !be || !be->device) {
        ffx_log("ffxCreateContext: missing UPSCALE or BACKEND_DX12 in the chain");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (up->maxRenderSize.width == 0 || up->maxRenderSize.height == 0 ||
        up->maxUpscaleSize.width == 0 || up->maxUpscaleSize.height == 0) {
        ffx_log("ffxCreateContext: maxRenderSize or maxUpscaleSize is 0");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (up->flags & FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE) {
        ffx_log("ffxCreateContext: NON_LINEAR_COLORSPACE refused (chain wants linear colour)");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    (void)memCb;  // system allocator is fine; the real signature keeps the callbacks unused
    FfxUpscaleContext* c = new (std::nothrow) FfxUpscaleContext();
    if (!c) return FFX_API_RETURN_ERROR_MEMORY;
    c->flags = up->flags;
    c->maxRenderSize = up->maxRenderSize;
    c->maxUpscaleSize = up->maxUpscaleSize;
    c->fpMessage = up->fpMessage;
    c->d3d12Device = be->device;
    c->creates = 0;
    // The bridge feature state: sizes stay 0 until the first dispatch names the REAL frame
    // geometry (FFX contexts are max-size, frames are actual-size).  feature int: NVSDK enum 1
    // = SuperSampling, so the shared record path takes its SR branch (rr stays false).
    c->fs = FeatureState_{};
    c->fs.feature = NVSDK_NGX_Feature_SuperSampling;
    c->fs.d3d_device = (ID3D12Device*)be->device;
    if (c->fs.d3d_device) c->fs.d3d_device->AddRef();
    // The vkd3d interop dance NOW (nvngx.cpp: ngxd_interop_device_for) - without it the first
    // GetVulkanResourceInfo has nothing to ask; it also fills g_vk_* + the entry points.
    ID3D12DXVKInteropDevice* interop = nullptr;
    if (c->fs.d3d_device && ngxd_interop_device_for(c->fs.d3d_device, &interop) && interop) {
        c->fs.interop = interop;   // consumed by the fetch/record path; ref owned by fs
        ffx_log("create: interop device ok (vkd3d handing the game's VkDevice)");
    } else {
        ffx_log("create: NO interop device (device=%p not vkd3d-proton?) - dispatch will refuse",
                (void*)c->fs.d3d_device);
    }
    *context = (ffxContext)c;
    ffx_log("ffxCreateContext: UPSCALE maxRender=%ux%u maxUpscale=%ux%u flags=0x%x",
            up->maxRenderSize.width, up->maxRenderSize.height,
            up->maxUpscaleSize.width, up->maxUpscaleSize.height, up->flags);
    return FFX_API_RETURN_OK;
}

__declspec(dllexport) ffxReturnCode_t __stdcall
ffxDestroyContext(ffxContext* context, const void* memCb) {
    (void)memCb;
    FfxUpscaleContext* c = as_ctx(context);
    if (!c) return FFX_API_RETURN_OK;  // NULL / foreign double-destroy tolerance (plan section 3)
    if (c->fs.chain) { delete c->fs.chain; c->fs.chain = nullptr; }
    if (c->fs.interop) { c->fs.interop->Release(); c->fs.interop = nullptr; }
    if (c->fs.d3d_device) { c->fs.d3d_device->Release(); c->fs.d3d_device = nullptr; }
    delete c;
    if (context) *context = nullptr;
    ffx_log("ffxDestroyContext: destroyed");
    return FFX_API_RETURN_OK;
}

__declspec(dllexport) ffxReturnCode_t __stdcall
ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    (void)context;
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1) {
        const ffxConfigureDescGlobalDebug1* d = (const ffxConfigureDescGlobalDebug1*)desc;
        ffx_log("ffxConfigure: GLOBALDEBUG1 fpMessage=%p debugLevel=0x%x", (void*)d->fpMessage,
                d->debugLevel);
        return FFX_API_RETURN_OK;
    }
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE) {
        const ffxConfigureDescUpscaleKeyValue* d = (const ffxConfigureDescUpscaleKeyValue*)desc;
        switch (d->key) {  // the five FSR3.1 tuning keys: accepted as no-ops (plan section 3)
        case FFX_API_CONFIGURE_UPSCALE_KEY_FVELOCITYFACTOR:
        case FFX_API_CONFIGURE_UPSCALE_KEY_FREACTIVENESSSCALE:
        case FFX_API_CONFIGURE_UPSCALE_KEY_FSHADINGCHANGESCALE:
        case FFX_API_CONFIGURE_UPSCALE_KEY_FACCUMULATIONADDEDPERFRAME:
        case FFX_API_CONFIGURE_UPSCALE_KEY_FMINDISOCCLUSIONACCUMULATION:
            ffx_log("ffxConfigure: key=%llu accepted (no-op)", (unsigned long long)d->key);
            return FFX_API_RETURN_OK;
        default:
            ffx_log("ffxConfigure: key=%llu unknown", (unsigned long long)d->key);
            return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
        }
    }
    ffx_log("ffxConfigure: desc type %llu unknown", (unsigned long long)desc->type);
    return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
}

__declspec(dllexport) ffxReturnCode_t __stdcall
ffxQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    (void)context;
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;
    switch (desc->type) {
    case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
        ffxQueryDescGetVersions* d = (ffxQueryDescGetVersions*)desc;
        if (!d->outputCount) return FFX_API_RETURN_ERROR_PARAMETER;
        if (*d->outputCount == 0) { *d->outputCount = 1; return FFX_API_RETURN_OK; }
        if (d->versionIds) *d->versionIds = kProviderVersionId;
        if (d->versionNames) *d->versionNames = kProviderName;
        *d->outputCount = 1;
        ffx_log("ffxQuery: GET_VERSIONS -> 1 (%s)", kProviderName);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
        ffxQueryGetProviderVersion* d = (ffxQueryGetProviderVersion*)desc;
        d->versionId = kProviderVersionId;
        d->versionName = kProviderName;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
        ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode* d =
            (ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*)desc;
        if (d->qualityMode > 4 || !d->pOutUpscaleRatio) return FFX_API_RETURN_ERROR_PARAMETER;
        *d->pOutUpscaleRatio = kQualityRatio[d->qualityMode];
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
        ffxQueryDescUpscaleGetRenderResolutionFromQualityMode* d =
            (ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*)desc;
        if (d->qualityMode > 4 || !d->pOutRenderWidth || !d->pOutRenderHeight)
            return FFX_API_RETURN_ERROR_PARAMETER;
        *d->pOutRenderWidth = d->displayWidth / (int)kQualityRatio[d->qualityMode];
        *d->pOutRenderHeight = d->displayHeight / (int)kQualityRatio[d->qualityMode];
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
        ffxQueryDescUpscaleGetJitterPhaseCount* d = (ffxQueryDescUpscaleGetJitterPhaseCount*)desc;
        if (!d->pOutPhaseCount) return FFX_API_RETURN_ERROR_PARAMETER;
        *d->pOutPhaseCount = jitter_phase_count(d->renderWidth, d->displayWidth);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
        ffxQueryDescUpscaleGetJitterOffset* d = (ffxQueryDescUpscaleGetJitterOffset*)desc;
        halton_jitter(d->index, d->phaseCount, d->pOutX, d->pOutY);
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS: {
        ffxQueryDescUpscaleGetResourceRequirements* d =
            (ffxQueryDescUpscaleGetResourceRequirements*)desc;
        d->required_resources = FFX_API_QUERY_RESOURCE_INPUT_COLOR |
                                FFX_API_QUERY_RESOURCE_INPUT_DEPTH |
                                FFX_API_QUERY_RESOURCE_INPUT_MV;
        d->optional_resources = FFX_API_QUERY_RESOURCE_INPUT_EXPOSURE |
                                FFX_API_QUERY_RESOURCE_INPUT_REACTIVEMASK |
                                FFX_API_QUERY_RESOURCE_INPUT_TRANSPARENCYCOMPOSITION;
        return FFX_API_RETURN_OK;
    }
    case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: {
        ffxQueryDescUpscaleGetGPUMemoryUsage* d = (ffxQueryDescUpscaleGetGPUMemoryUsage*)desc;
        if (d->gpuMemoryUsageUpscaler) {
            d->gpuMemoryUsageUpscaler->totalUsageInBytes = 0;
            d->gpuMemoryUsageUpscaler->aliasableUsageInBytes = 0;
        }
        return FFX_API_RETURN_OK;
    }
    default:
        ffx_log("ffxQuery: desc type %llu unknown", (unsigned long long)desc->type);
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    }
}

__declspec(dllexport) ffxReturnCode_t __stdcall
ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;
    if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK) {
        // No reactive pass in the chain: OK with nothing recorded (the FFX-conformant answer).
        return FFX_API_RETURN_OK;
    }
    if (desc->type != FFX_API_DISPATCH_DESC_TYPE_UPSCALE)
        return FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE;
    FfxUpscaleContext* c = as_ctx(context);
    const ffxDispatchDescUpscale* d = (const ffxDispatchDescUpscale*)desc;
    if (!c) {
        ffx_log("ffxDispatch: null/foreign context");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (!d->color.resource || !d->output.resource || !d->commandList ||
        d->renderSize.width == 0 || d->renderSize.height == 0) {
        ffx_log("ffxDispatch: missing color/output/commandList or zero renderSize");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    if (c->creates == 0) {
        ffx_log("ffxDispatch: first frame R=%ux%u O=%ux%u jitter=(%.6g,%.6g) "
                "mvScale=(%.6g,%.6g) reset=%u",
                d->renderSize.width, d->renderSize.height,
                d->upscaleSize.width, d->upscaleSize.height,
                d->jitterOffset.x, d->jitterOffset.y,
                d->motionVectorScale.x, d->motionVectorScale.y, d->reset ? 1u : 0u);
    }
    if (c->build_failed) {
        // One refuse per context after a failed build (the reason is already in the log).
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    // --- frame geometry -----------------------------------------------------------
    // First frame (or after the context saw a NEW size): set the shared FeatureState_'s sizes
    // and build the chain once.  FFX has no resize desc (plan §3) - a changed render or upscale
    // size destroys + rebuilds the chain here, the same answer nvngx.cpp's resize takes.
    const bool same =
        c->fs.rw == d->renderSize.width && c->fs.rh == d->renderSize.height &&
        c->fs.ow == d->upscaleSize.width && c->fs.oh == d->upscaleSize.height &&
        c->fs.chain != nullptr;
    if (!same) {
        if (c->fs.chain) { delete c->fs.chain; c->fs.chain = nullptr; ffx_log("dispatch: size change -> rebuild"); }
        c->fs.rw = d->renderSize.width;
        c->fs.rh = d->renderSize.height;
        c->fs.ow = d->upscaleSize.width ? d->upscaleSize.width : c->maxUpscaleSize.width;
        c->fs.oh = d->upscaleSize.height ? d->upscaleSize.height : c->maxUpscaleSize.height;
        // Preset from the dispatch quality flags is not carried per-frame; v1 keeps M (the
        // 1920x1080 -> 3840x2160 default the harness + the game's pq=0 S2 rows tested).  The
        // quality-mode query path (kQualityRatio) is advisory only at create time.
        c->fs.preset = 'M';
        c->fs.frame_idx = 0;
        c->fs.eval_id = 0;
        c->fs.s6_logged = 0;
        // build the chain RIGHT HERE (nvngx.cpp build_chain's FFX twin, against the shared
        // FeatureState_'s fields; weights at DLSS5_AMD_WEIGHTS, shaders beside the DLL).
        // module_path_tail is private to nvngx.cpp; the DLL's dir trick works identically by
        // asking the weights root + the SAME dlss5-amd\\shaders tail the NGX surface uses.
        if (!c->fs.interop) {
            c->build_failed = true;
            ffx_log("dispatch: refuse - no interop device (create logged the reason)");
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        char shaders[MAX_PATH] = {0};
        {
            HMODULE self = nullptr;
            GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCWSTR)&ffxCreateContext, &self);
            wchar_t wpath[MAX_PATH] = {0};
            if (self && GetModuleFileNameW(self, wpath, MAX_PATH)) {
                // ...\\bin64\\dlss5-amd-ffx.dll -> ...\\bin64\\dlss5-amd\\shaders
                std::wstring ws(wpath);
                const size_t slash = ws.find_last_of(L"\\/");
                if (slash != std::wstring::npos) {
                    const std::wstring dir = ws.substr(0, slash) + L"\\dlss5-amd\\shaders";
                    WideCharToMultiByte(CP_UTF8, 0, dir.c_str(), -1, shaders, sizeof shaders, nullptr, nullptr);
                }
            }
            if (!shaders[0]) ffx_log("dispatch: shaders dir unresolved (GetModuleFileName failed)");
        }
        const std::string root = ngxd_weights_root();
        if (root.empty()) {
            c->build_failed = true;
            ffx_log("dispatch: DLSS5_AMD_WEIGHTS is not set (installer has to say where the packed "
                    "weights are) - refusing frames");
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        const std::string w = root + "/M";
        ffx_log("dispatch: creating the chain %ux%u -> %ux%u from %s", c->fs.rw, c->fs.rh, c->fs.ow, c->fs.oh,
                w.c_str());
        const unsigned long long t0 = ngx_ms();
        std::string err;
        c->fs.chain = ngxd::FrameChain::create(ngxd_vk_device(), ngxd_vk_phys_dev(), "sr", "M",
                                               int(c->fs.rw), int(c->fs.rh), int(c->fs.ow), int(c->fs.oh),
                                               w, shaders, &err);
        if (!c->fs.chain) {
            c->build_failed = true;
            ffx_log("dispatch: no frame chain (%s)", err.c_str());
            return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
        }
        ffx_log("dispatch: preset M %ux%u -> %ux%u chain ready in %llu ms, %zu pipelines",
                c->fs.rw, c->fs.rh, c->fs.ow, c->fs.oh, ngx_ms() - t0, c->fs.chain->pipeline_count());
    }
    // --- the command list -> the chain ---------------------------------------------
    ID3D12GraphicsCommandList* cl = (ID3D12GraphicsCommandList*)d->commandList;
    ID3D12GraphicsCommandListExt* ext = nullptr;
    if (FAILED(cl->QueryInterface(ngxd_IID_ID3D12GraphicsCommandListExt, reinterpret_cast<void**>(&ext))) || !ext) {
        ffx_log("dispatch: the command list has no ID3D12GraphicsCommandListExt (not vkd3d?)");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    VkCommandBuffer vkcmd = VK_NULL_HANDLE;
    ext->GetVulkanHandle(&vkcmd);
    ext->Release();
    if (!vkcmd) {
        ffx_log("dispatch: GetVulkanHandle gave a null VkCommandBuffer");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    // --- resource fetch --------------------------------------------------------------
    // Bridge slots, exactly the NGX D3D12 path's table.  FFX hands the D3D12 resource pointer
    // + its state in the desc; the shared dxgi_to_vk map + interop translate.
    // NOTE: FfxApiResource here is the SKELETON's simplified shape (description as POINTER);
    // the vendored ffx_api_types.h has it BY VALUE.  v0.9.4's FfxApi_Proxy fills ours, so the
    // skeleton shape stays until step 4's switch to the vendored headers.
    (void)d;  // resources are consumed below via the desc's pointers
    // Color (RG16B16A16 in-game; the harness cell stages RGBA16F - both code 0)
    auto fill_bridge = [&](const FfxApiResource& r, const char* name, VkBuffer buf,
                           D3D12_RESOURCE_STATES st, BridgeImage_& o) -> bool {
        o = BridgeImage_{};
        o.buf = buf;
        o.name = name;
        if (!r.resource || !c->fs.interop) return false;
        ID3D12Resource* res = (ID3D12Resource*)r.resource;
        const auto desc = res->GetDesc();
        UINT64 handle = 0, off2 = 0;
        const bool have = SUCCEEDED(c->fs.interop->GetVulkanResourceInfo(res, &handle, &off2)) && handle != 0;
        o.image = have ? (VkImage)handle : VK_NULL_HANDLE;
        o.format = ngxd_dxgi_to_vk(desc.Format);
        o.w = unsigned(desc.Width);
        o.h = unsigned(desc.Height);
        o.dxgi = unsigned(desc.Format);
        // depth-stencil images carry their texels on the DEPTH aspect (see the header audit)
        if (o.format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
            o.format == VK_FORMAT_D24_UNORM_S8_UINT ||
            o.format == VK_FORMAT_D32_SFLOAT) {
            o.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        } else {
            o.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        c->fs.interop->GetVulkanImageLayout(res, st, &o.layout);
        if (!have || o.format == VK_FORMAT_UNDEFINED || o.layout == VK_IMAGE_LAYOUT_UNDEFINED) {
            ffx_log("dispatch: %s fetch failed (have=%d fmt=%d layout=%d dxgi=%u)", name, int(have),
                    int(o.format), int(o.layout), o.dxgi);
            return false;
        }
        return true;
    };
    BridgeImage_ bi_color, bi_mv, bi_dep, bi_out;
    const bool ok_c = fill_bridge(d->color,    "Color",        c->fs.chain->in_color_buf(),   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, bi_color);
    const bool ok_m = fill_bridge(d->motionVectors, "MotionVectors", c->fs.chain->in_mv_buf(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, bi_mv);
    const bool ok_d = d->depth.resource ? fill_bridge(d->depth, "Depth", c->fs.chain->in_depth_buf(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, bi_dep) : true;
    const bool ok_o = fill_bridge(d->output,   "Output",       c->fs.chain->out_color_buf(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, bi_out);
    if (!ok_c || !ok_m || !ok_d || !ok_o) {
        ffx_log("dispatch: fetch refused (c=%d m=%d d=%d o=%d)", int(ok_c), int(ok_m), int(ok_d), int(ok_o));
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    // --- frame parameters -------------------------------------------------------------
    ngxd::FrameParams fp{};
    fp.jx = d->jitterOffset.x;
    fp.jy = d->jitterOffset.y;
    if (c->have_prev_j) { fp.jpx = c->prev_jx; fp.jpy = c->prev_jy; }
    fp.reset = d->reset ? 1 : 0;
    fp.pre_exposure = d->preExposure > 0.0f ? d->preExposure : 1.0f;
    fp.exposure = 1.0f;   // the chain's exposure comes pre-applied on the color input in FFX mode
    // MV scale: FFX carries the SAME pair NGX's MV.Scale.X/.Y carries (measured in-game pair
    // (1280,-720) = UV-normalized flipped-Y).  The S6 gather decode consumes it unchanged.
    fp.mv_scale_x = d->motionVectorScale.x;
    fp.mv_scale_y = d->motionVectorScale.y;
    if ((fp.mv_scale_x != 1.0f || fp.mv_scale_y != 1.0f) && !c->fs.s6_logged) {
        c->fs.s6_logged = 1;
        ffx_log("[S6] mv scale=(%g,%g)", fp.mv_scale_x, fp.mv_scale_y);
    }
    c->prev_jx = fp.jx; c->prev_jy = fp.jy; c->have_prev_j = true;
    // --- record --------------------------------------------------------------------------
    BridgeImage_ ins[3] = {bi_color, bi_mv, bi_dep};
    const NVSDK_NGX_Result rr = ngxd_record_bridge(vkcmd, &c->fs, fp, ins, d->depth.resource ? 3 : 2, &bi_out);
    ++c->fs.frame_idx;
    if (!ngx_ok(rr)) {
        ffx_log("dispatch: record refused (0x%x)", unsigned(rr));
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }
    // Per-frame + 1 Hz observability, the d5 style the gates can grep:
    static unsigned long long s_last_ms = 0;
    const unsigned long long now = ngx_ms();
    if (c->creates == 0 || now - s_last_ms >= 1000u) {
        s_last_ms = now;
        ffx_log("frame %u: %ux%u -> %ux%u eval #%llu OK", c->fs.frame_idx, c->fs.rw, c->fs.rh, c->fs.ow,
                c->fs.oh, c->fs.eval_id);
    }
    ++c->creates;
    return FFX_API_RETURN_OK;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Extra export (our own, like WrongGPU's wgGetActiveProviderLabel - tooling only, no host in
// OptiScaler v0.9.4 resolves it).
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport) const char* dlss5AmdFfxProviderLabel(void) {
    return kProviderLabel;
}
