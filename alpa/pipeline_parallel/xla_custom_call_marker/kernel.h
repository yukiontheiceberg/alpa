#include <cuda_runtime_api.h>

namespace kernel {

void identity(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len);

// __global__ void add_dummy(int *a, int *b, int *c);

// void initialize_memory();

void dummy_compute_on_default_stream();

}  // namespace kernel
