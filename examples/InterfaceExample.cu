#include "../DevicePtr.hpp"
#include "common.cuh"

typedef float buffer_t;

void runKernelsExample(std::vector<buffer_t> &v_a)
{
    // Create new DevicePtr
    dptr::DevicePtr<buffer_t> a;

    // Initialize new DevicePtr from an existing allocated device pointer
    buffer_t *d_a;
    cudaMalloc(&d_a, v_a.size() * sizeof(buffer_t));
    // The DevicePtr takes ownership of the raw CUDA pointer (d_a)
    // the pointer will be freed at scope exit
    a.reset(d_a, v_a.size());

    dim3 blockSize(256, 1, 1);
    dim3 gridSize(ceil(static_cast<float>(a.getCount())/blockSize.x),
                  1,
                  1);

    // Launch dummy kernel on the data
    // The CUDA kernel receives the raw pointer to the GPU resource with obtained with *a.get()*
    // The kernel internally operates with the data as if the resource was directly allocated with cudaMalloc
    dummyKernel<<<gridSize, blockSize>>>(a.get(), a.getCount());

    uint64_t count = a.getCount();

    // Release the pointer from the DevicePtr
    // User takes ownership of the pointer and needs to take care of freeing it
    // The DevicePtr state is default initialization
    buffer_t *d_raw_a = a.release();

    // Launch dummy kernel on the data
    // This time the kernel is called using the raw CUDA pointer
    dummyKernel<<<gridSize, blockSize>>>(d_raw_a, count);

    // Download the data to the CPU
    cudaMemcpy(v_a.data(), d_raw_a, count * sizeof(buffer_t), cudaMemcpyDeviceToHost);

    // Free the pointer
    cudaFree(d_raw_a);
}

int main(int argc, char **argv)
{
    uint32_t count = 10;
    std::vector<buffer_t> v_a(count);
    fillVector(v_a);

    runKernelsExample(v_a);

    printVector(v_a);

    cudaDeviceReset();
    return 1;

}
