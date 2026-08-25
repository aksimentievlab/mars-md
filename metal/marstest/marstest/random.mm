//
//  random.cpp
//  arbdtest
//
//  Created by PinYi on 8/3/25.
//

// Example: Using arbdtest.metal kernels from Objective-C++

#import <Metal/Metal.h>
#include <vector>
#include <iostream>

id<MTLDevice> device = MTLCreateSystemDefaultDevice();
id<MTLCommandQueue> commandQueue = [device newCommandQueue];

// Load metallib
NSString* metallibPath = [[NSBundle mainBundle] pathForResource:@"arbdtest" ofType:@"metallib"];
NSError* error = nil;
id<MTLLibrary> library = [device newLibraryWithFile:metallibPath error:&error];
if (!library) {
    std::cerr << "Failed to create Metal library: " << error.localizedDescription.UTF8String << std::endl;
    return 1;
}

id<MTLFunction> uniformKernel = [library newFunctionWithName:@"UniformFunctor_operator()"];
if (!uniformKernel) {
    std::cerr << "Could not find the UniformFunctor operator kernel." << std::endl;
    return 1;
}

id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:uniformKernel error:&error];
if (!pipeline) {
    std::cerr << "Failed to create pipeline: " << error.localizedDescription.UTF8String << std::endl;
    return 1;
}

// Example: output buffer for 16 float4s
typedef struct {
    float min_val, max_val;
    uint64_t base_seed;
    uint32_t base_ctr;
    uint32_t global_seed;
} UniformFunctorData;

const int count = 16;
std::vector<float> output(count * 4, 0.0f);

UniformFunctorData params = {0.0f, 1.0f, 123456789ULL, 0, 0xAAAAAAAA};

id<MTLBuffer> outputBuffer = [device newBufferWithBytes:output.data() length:output.size()*sizeof(float) options:MTLResourceStorageModeShared];
id<MTLBuffer> paramBuffer = [device newBufferWithBytes:&params length:sizeof(params) options:MTLResourceStorageModeShared];

id<MTLCommandBuffer> cmdbuf = [commandQueue commandBuffer];
id<MTLComputeCommandEncoder> encoder = [cmdbuf computeCommandEncoder];
[encoder setComputePipelineState:pipeline];
[encoder setBuffer:outputBuffer offset:0 atIndex:0];
[encoder setBuffer:paramBuffer offset:0 atIndex:1];

MTLSize grid = MTLSizeMake(count, 1, 1);
MTLSize threadgroup = MTLSizeMake(1, 1, 1);
[encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
[encoder endEncoding];
[cmdbuf commit];
[cmdbuf waitUntilCompleted];

// Copy back
float* results = (float*)outputBuffer.contents;
for (int i = 0; i < count * 4; ++i) {
    std::cout << results[i] << (i % 4 == 3 ? "\n" : ", ");
}
