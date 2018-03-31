#include <vector>

#include "../DevicePtr.hpp"
#include "common.cuh"

typedef float buffer_t;

void runKernelExample1(std::vector<buffer_t> &v_a)
{
    // Create a new DevicePtr
    // the object takes care of allocating the necessary resources in the GPU
    // and to copy the data in *v_a* into it. The object will take care of
    // deallocating the GPU resources when it goes out of scope
    dptr::DevicePtr<buffer_t> a(v_a.data(), v_a.size());

    dim3 blockSize(256, 1, 1);
    dim3 gridSize(ceil(static_cast<float>(a.getCount())/blockSize.x),
                  1,
                  1);

    // Launch a dummy kernel on the data
    // The CUDA kernel receives the raw pointer to the GPU resource with obtained with *a.get()*
    // The kernel internally operates with the data as if the resource was directly allocated with cudaMalloc
    dummyKernel<<<gridSize, blockSize>>>(a.get(), a.getCount());

    // Download the data to the CPU
    cudaMemcpy(v_a.data(), a.get(), a.getBytes(), cudaMemcpyDeviceToHost);

} // At this point (at the scope exit) the CUDA resource associated
  // with the DevicePtr object will automatically be freed

void runKernelExample2(std::vector<buffer_t> &v_a)
{
    // Create new DevicePtr
    // The object takes care of allocating the necessary resources in the GPU
    // The object will take care of deallocating the GPU resources when it goes out of scope
    dptr::DevicePtr<buffer_t> a(v_a.size());

    // Copy data into the DevicePtr resource
    // We have direct access with the raw GPU pointer,
    cudaMemcpy(a.get(), v_a.data(), v_a.size() * sizeof(buffer_t), cudaMemcpyHostToDevice);

    dim3 blockSize(256, 1, 1);
    dim3 gridSize(ceil(static_cast<float>(a.getCount())/blockSize.x),
                  1,
                  1);

    // Launch a dummy kernel on the data
    // The CUDA kernel receives the raw pointer to the GPU resource with obtained with *a.get()*
    // The kernel internally operates with the data as if the resource was directly allocated with cudaMalloc
    dummyKernel<<<gridSize, blockSize>>>(a.get(), a.getCount());

    // Download the data to the CPU
    cudaMemcpy(v_a.data(), a.get(), a.getBytes(), cudaMemcpyDeviceToHost);

} // At this point (at the scope exit) the CUDA resource associated
  // with the DevicePtr object will automatically be freed


int main(int argc, char **argv)
{
    uint32_t count = 10;
    std::vector<buffer_t> v_a(count);
    fillVector(v_a);

    runKernelExample1(v_a);

    runKernelExample2(v_a);

    printVector(v_a);

    cudaDeviceReset();
    return 1;
}
