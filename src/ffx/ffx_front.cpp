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
// Local logging (ffx_front writes its own log like the NGX bridge writes dlss5-amd-ngx.log).
// v1 placeholder: stderr-free, AppendDebugView-free - a plain file next to the DLL is decided
// with the ffx_d3d12.cpp step; for the skeleton a minimal CRT file logger keeps mingw happy.
// ---------------------------------------------------------------------------
static void ffx_log(const char* fmt, ...) {
    (void)fmt;  // wired in step 3; the skeleton only needs the symbol to exist
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
// Context object (skeleton: descriptor snapshot only; chain handle arrives in step 3).
// ---------------------------------------------------------------------------
struct FfxUpscaleContext {
    uint32_t                     flags = 0;
    FfxApiDimensions2D           maxRenderSize = {};
    FfxApiDimensions2D           maxUpscaleSize = {};
    ffxApiMessage                fpMessage = nullptr;
    void*                        d3d12Device = nullptr;
    unsigned long long           creates = 0;
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
    ++c->creates;
    // SKELETON: the frame chain is not wired in yet.  Report a runtime error so a host that
    // dispatches anyway fails cleanly (Opti recreates the feature on RUNTIME_ERROR) instead of
    // reading an unwritten output.
    return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// Extra export (our own, like WrongGPU's wgGetActiveProviderLabel - tooling only, no host in
// OptiScaler v0.9.4 resolves it).
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport) const char* dlss5AmdFfxProviderLabel(void) {
    return kProviderLabel;
}
