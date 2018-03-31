#include <iostream>

#include "../DevicePtr.hpp"

using namespace dptr;

__global__
void mapKernel(float *p)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    p[idx]++;
}

void function()
{
    float h_buffer[10] = {1,2,3,4,5,6,7,8,9,10};

    DevicePtr<float> buffer(10);

    cudaMemcpy(buffer.get(), h_buffer, buffer.getBytes(), cudaMemcpyHostToDevice);

    mapKernel<<<1,10>>>(buffer.get());

    cudaMemcpy(h_buffer, buffer.get(), buffer.getBytes(), cudaMemcpyDeviceToHost);

}

int main(int argc, char** argv)
{
    std::cout << "example" << std::endl;

    function();

    cudaDeviceReset();
    return 1;
}
