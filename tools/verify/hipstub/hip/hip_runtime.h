// Declarations only, for `g++ -fsyntax-only` of the ROCm oracles (srref, rrref) on a machine without
// ROCm (tools/verify/offline_check.sh). Nothing here is defined, so nothing can link or run; the
// signatures follow HIP's public API for the calls the oracles make.
#pragma once
#include <cstddef>
#include <cstdint>

typedef enum hipError_t { hipSuccess = 0 } hipError_t;
typedef struct ihipModule_t* hipModule_t;
typedef struct ihipModuleSymbol_t* hipFunction_t;
typedef struct ihipStream_t* hipStream_t;
typedef struct ihipEvent_t* hipEvent_t;
typedef void* hipDeviceptr_t;
typedef unsigned long long hipTextureObject_t;
enum hipMemcpyKind { hipMemcpyHostToHost, hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice };
enum hipChannelFormatKind { hipChannelFormatKindSigned, hipChannelFormatKindUnsigned, hipChannelFormatKindFloat };
enum hipResourceType { hipResourceTypeArray, hipResourceTypeMipmappedArray, hipResourceTypeLinear, hipResourceTypePitch2D };
enum hipTextureAddressMode { hipAddressModeWrap, hipAddressModeClamp, hipAddressModeMirror, hipAddressModeBorder };
enum hipTextureFilterMode { hipFilterModePoint, hipFilterModeLinear };
enum hipTextureReadMode { hipReadModeElementType, hipReadModeNormalizedFloat };
struct hipChannelFormatDesc { int x, y, z, w; hipChannelFormatKind f; };
struct hipResourceDesc {
    hipResourceType resType;
    union {
        struct { void* devPtr; hipChannelFormatDesc desc; size_t width, height, pitchInBytes; } pitch2D;
    } res;
};
struct hipTextureDesc {
    hipTextureAddressMode addressMode[3];
    hipTextureFilterMode filterMode;
    hipTextureReadMode readMode;
    int sRGB;
    float borderColor[4];
    int normalizedCoords;
};
struct hipResourceViewDesc;
struct hipDeviceProp_t { size_t texturePitchAlignment; };
struct dim3 {
    unsigned x, y, z;
    dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
#define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define HIP_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define HIP_LAUNCH_PARAM_END ((void*)0x03)

const char* hipGetErrorString(hipError_t);
hipError_t hipDeviceSynchronize();
hipError_t hipMalloc(void**, size_t);
hipError_t hipMallocPitch(void**, size_t*, size_t, size_t);
hipError_t hipFree(void*);
hipError_t hipMemset(void*, int, size_t);
hipError_t hipMemset2D(void*, size_t, int, size_t, size_t);
hipError_t hipMemcpy(void*, const void*, size_t, hipMemcpyKind);
hipError_t hipMemcpy2D(void*, size_t, const void*, size_t, size_t, size_t, hipMemcpyKind);
hipError_t hipModuleLoad(hipModule_t*, const char*);
hipError_t hipModuleGetFunction(hipFunction_t*, hipModule_t, const char*);
hipError_t hipModuleGetGlobal(hipDeviceptr_t*, size_t*, hipModule_t, const char*);
hipError_t hipModuleLaunchKernel(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                 hipStream_t, void**, void**);
hipChannelFormatDesc hipCreateChannelDesc(int, int, int, int, hipChannelFormatKind);
hipError_t hipCreateTextureObject(hipTextureObject_t*, const hipResourceDesc*, const hipTextureDesc*, const hipResourceViewDesc*);
hipError_t hipEventCreate(hipEvent_t*);
hipError_t hipEventRecord(hipEvent_t, hipStream_t);
hipError_t hipEventElapsedTime(float*, hipEvent_t, hipEvent_t);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
