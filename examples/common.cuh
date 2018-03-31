#include <vector>
#include <iostream>
#include <limits>

///////////////////////////////////////////////////////////
template<typename T>
void fillVector(std::vector<T> &vec)
{
    for (auto it = vec.begin(); it != vec.end(); it++)
        *it = static_cast<T>(rand()) / RAND_MAX;
}

///////////////////////////////////////////////////////////
template<typename T>
void printVector(std::vector<T> &vec)
{
    for (auto it = vec.begin(); it != vec.end(); it++)
        std::cout << "value: " << *it << std::endl;
}

///////////////////////////////////////////////////////////
template<typename T>
__global__ void dummyKernel(T *buff, uint64_t count)
{
    uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < count)
        buff[idx] += static_cast<T>(1);
}
