#include "kernel.h"
#include <stdio.h>

namespace kernel {

void identity(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
    const int64_t *sizes = reinterpret_cast<const int64_t *>(opaque);
    size_t n_inputs = opaque_len / sizeof(int64_t);
    for (size_t i = 0; i < n_inputs; i++) {
        const void *input = reinterpret_cast<const void *>(buffers[i]);
        void *output = reinterpret_cast<void *>(buffers[i + n_inputs]);
        if (input != output) {
            printf("WARNING: The inputs and outputs of idenity marker are not aliases\n");
            cudaMemcpy(output, input, sizes[i], cudaMemcpyDeviceToDevice);
        }
    }
}

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

void dummy_compute_on_default_stream(int device_id) {
    if (device_id == -1) {
        int n_devices;
        cudaGetDeviceCount(&n_devices);
        for (int i = 0; i < n_devices; ++i) {
            cudaSetDevice(i);
            kernel<<<1,1>>>(0, 0);
        }
        return;
    }
    cudaSetDevice(device_id);
    kernel<<<1,1>>>(0, 0);
}

};  // end namespace kernel
