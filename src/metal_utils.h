// Shared Metal device/queue/library + MPS zero-copy buffer binding.
// Consolidates singletons used across rasterize, interpolate, texture, antialias.
//
// On Apple Silicon, MPS and CPU tensors share unified memory.
// data_ptr() returns a valid address for both — we bind it directly via
// newBufferWithBytesNoCopy (zero copies).
#pragma once

#import <Metal/Metal.h>
#import <torch/extension.h>
#include <string>
#include <unordered_map>

// Public MPS API for synchronization
#if __has_include(<torch/mps.h>)
#include <torch/mps.h>
#define MTLDIFFRAST_HAS_MPS 1
#else
#define MTLDIFFRAST_HAS_MPS 0
#endif

#include <dlfcn.h>

namespace mtldiffrast {

//------------------------------------------------------------------------
// Shared Metal device / command queue / library singletons.

inline id<MTLDevice> mtl_get_device() {
    static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev;
}

inline id<MTLCommandQueue> mtl_get_queue() {
    static id<MTLCommandQueue> q = [mtl_get_device() newCommandQueue];
    return q;
}

// Anchor symbol for dladdr-based metallib discovery.
static void _mtl_utils_anchor() {}

inline id<MTLLibrary> mtl_get_library() {
    static id<MTLLibrary> lib = nil;
    if (!lib) {
        NSError* error = nil;
        Dl_info dl_info;
        dladdr((void*)&_mtl_utils_anchor, &dl_info);
        NSString* soPath = [NSString stringWithUTF8String:dl_info.dli_fname];
        NSString* dir = [soPath stringByDeletingLastPathComponent];

        // Try combined metallib next to .so
        NSString* libPath = [dir stringByAppendingPathComponent:@"mtldiffrast.metallib"];
        if ([[NSFileManager defaultManager] fileExistsAtPath:libPath]) {
            lib = [mtl_get_device() newLibraryWithURL:[NSURL fileURLWithPath:libPath] error:&error];
        }
        if (!lib) lib = [mtl_get_device() newDefaultLibrary];
        TORCH_CHECK(lib != nil, "[mtldiffrast] Failed to load Metal library");
    }
    return lib;
}

inline id<MTLComputePipelineState> mtl_get_pipeline(const char* name) {
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;
    std::string key(name);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    NSError* error = nil;
    id<MTLFunction> func = [mtl_get_library() newFunctionWithName:
        [NSString stringWithUTF8String:name]];
    TORCH_CHECK(func != nil, "[mtldiffrast] Metal function not found: ", name);
    id<MTLComputePipelineState> pso = [mtl_get_device() newComputePipelineStateWithFunction:func error:&error];
    TORCH_CHECK(pso != nil, "[mtldiffrast] Failed to create pipeline for: ", name);
    cache[key] = pso;
    return pso;
}

//------------------------------------------------------------------------
// Tensor device helpers.

inline bool tensor_is_mps(const torch::Tensor& t) {
    return t.device().type() == torch::kMPS;
}

//------------------------------------------------------------------------
// Zero-copy Metal buffer binding.
// Apple Silicon unified memory: both MPS and CPU tensor data_ptr() are
// valid MTLBuffer-compatible addresses. No copies needed.

struct MtlBufferRef {
    id<MTLBuffer> buffer;
    NSUInteger offset;  // byte offset into buffer
};

inline MtlBufferRef tensor_to_mtl_buffer(const torch::Tensor& t) {
    TORCH_CHECK(t.is_contiguous(), "[mtldiffrast] tensor must be contiguous for Metal binding");

    // MPS tensors must be moved to CPU first — MPS data_ptr() points into
    // MPS's own MTLBuffer allocator and cannot be re-wrapped. On Apple Silicon
    // unified memory this is a metadata-only operation (same physical pages).
    auto tc = tensor_is_mps(t) ? t.cpu() : t;

    size_t nbytes = tc.nbytes();
    TORCH_CHECK(nbytes > 0, "[mtldiffrast] cannot create Metal buffer from empty tensor");

    // CPU tensors: wrap existing memory directly — zero copies on unified memory.
    id<MTLBuffer> buf = [mtl_get_device() newBufferWithBytesNoCopy:tc.data_ptr()
                                                             length:nbytes
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
    TORCH_CHECK(buf != nil, "[mtldiffrast] Failed to create Metal buffer from tensor");
    return {buf, 0};
}

// Create output tensors on CPU — Metal kernels write to CPU memory via
// newBufferWithBytesNoCopy. On Apple Silicon unified memory, CPU tensors
// are GPU-accessible.
inline torch::Tensor make_output_tensor(
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    const torch::Tensor& reference_tensor
) {
    return torch::zeros(sizes, torch::TensorOptions().dtype(dtype));
}

inline torch::Tensor make_empty_tensor(
    const std::vector<int64_t>& sizes,
    torch::ScalarType dtype,
    const torch::Tensor& reference_tensor
) {
    return torch::empty(sizes, torch::TensorOptions().dtype(dtype));
}

// Check if any tensor in a parameter pack is on MPS.
template<typename... Ts>
inline bool any_tensor_on_mps(const Ts&... tensors) {
    bool result = false;
    (void)std::initializer_list<int>{(result = result || tensor_is_mps(tensors), 0)...};
    return result;
}

// Synchronize MPS command stream before Metal dispatch.
// Must be called when reading from MPS tensors in custom Metal kernels.
inline void mps_sync() {
#if MTLDIFFRAST_HAS_MPS
    torch::mps::commit();
#endif
}

} // namespace mtldiffrast
