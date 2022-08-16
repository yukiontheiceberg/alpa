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

int *da[16], *db[16], *dc[16];
int n_devices;

// __global__ void add_dummy(int *a, int *b, int *c) {
//     *c = *a + *b;
// }

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

// void initialize_memory() {
//     cudaGetDeviceCount(&n_devices);
//     for (int i = 0; i < n_devices; ++i) {
//         cudaSetDevice(i);
//         int size = sizeof(int);
//         cudaMalloc((void **)&da[i], size);
//         cudaMalloc((void **)&db[i], size);
//         cudaMalloc((void **)&dc[i], size);
//     }
// }

void dummy_compute_on_default_stream() {
    for (int i = 0; i < n_devices; ++i) {
        cudaSetDevice(i);
        kernel<<<1,1>>>(0, 0);
        // add_dummy<<1,1,0,0>>((int *)da[i], (int *)db[i], (int *)dc[i]);
    }
}

};  // end namespace kernel
