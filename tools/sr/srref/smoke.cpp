// Load NeuralHub's gfx1201 SR code object with the ROCm runtime and run k_sr_unorm8_table.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); std::exit(1); } } while (0)
int main(int argc, char** argv) {
    hipModule_t m; CK(hipModuleLoad(&m, argv[1]));
    hipFunction_t f; CK(hipModuleGetFunction(&f, m, "_ZN2sr2io17k_sr_unorm8_tableEPt"));
    unsigned short* d; CK(hipMalloc(&d, 512));
    struct { unsigned short* p; } a{d};
    size_t sz = sizeof a;
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(f, 1, 1, 1, 256, 1, 1, 0, nullptr, nullptr, cfg));
    CK(hipDeviceSynchronize());
    std::vector<unsigned short> h(256); CK(hipMemcpy(h.data(), d, 512, hipMemcpyDeviceToHost));
    for (int i : {0, 1, 2, 127, 128, 255}) std::printf("unorm8[%d] = 0x%04x\n", i, h[i]);
    return 0;
}
