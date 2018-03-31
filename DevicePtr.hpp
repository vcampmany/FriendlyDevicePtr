#include <iostream>
#include <inttypes.h>
#include <cuda_runtime.h>

#ifndef DEVICE_PTR_HPP_
#define DEVICE_PTR_HPP_

namespace dptr
{

#define CUDA_SAFE(err) if (err != 0) std::cerr << "CUDA error: " << err << ", Description " << cudaGetErrorString(err) << std::endl;

template<typename T>
class DevicePtr
{
private:
    T* m_dptr;

    uint64_t m_count;

    uint64_t m_bytes;

    void freePtr();

public:
    DevicePtr<T>();
    DevicePtr<T>(uint64_t count);
    DevicePtr<T>(T *h_pointer, uint64_t count);

    ~DevicePtr<T>();

    T* get();

    T* release();

    void reset(T* d_pointer, uint64_t count);

    uint64_t getCount();

    uint64_t getBytes();
};

///////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////
template<typename T>
DevicePtr<T>::DevicePtr() :
                        m_dptr(nullptr), m_count(0), m_bytes(0)
{ }

///////////////////////////////////////////////////////////
template<typename T>
DevicePtr<T>::DevicePtr(uint64_t count) :
                        m_count(count)
{
    m_bytes = count * sizeof(T);
    CUDA_SAFE(cudaMalloc(&m_dptr, count * sizeof(T)));
}

///////////////////////////////////////////////////////////
template<typename T>
DevicePtr<T>::DevicePtr(T *h_pointer, uint64_t count) :
                        m_dptr(nullptr), m_count(count)
{
    m_bytes = count * sizeof(T);
    CUDA_SAFE(cudaMalloc(&m_dptr, m_bytes));
    CUDA_SAFE(cudaMemcpy(m_dptr, h_pointer, m_bytes, cudaMemcpyHostToDevice));
}

///////////////////////////////////////////////////////////
template<typename T>
DevicePtr<T>::~DevicePtr()
{
    freePtr();
}

///////////////////////////////////////////////////////////
template<typename T>
T* DevicePtr<T>::get()
{
    return m_dptr;
}

///////////////////////////////////////////////////////////
template<typename T>
T* DevicePtr<T>::release()
{
    T * p = m_dptr;
    m_dptr = nullptr;
    m_count = 0;
    m_bytes = 0;
    return p;
}

///////////////////////////////////////////////////////////
template<typename T>
void DevicePtr<T>::reset(T* d_pointer, uint64_t count)
{
    freePtr();
    m_dptr = d_pointer;
    m_count = count;
    m_bytes = count * sizeof(T);
}

///////////////////////////////////////////////////////////
template<typename T>
uint64_t DevicePtr<T>::getCount()
{
    return m_count;
}

///////////////////////////////////////////////////////////
template<typename T>
uint64_t DevicePtr<T>::getBytes()
{
    return m_bytes;
}

///////////////////////////////////////////////////////////
template<typename T>
void DevicePtr<T>::freePtr()
{
    cudaFree(m_dptr);
}

} /* end namespace */

#endif /* DEVICE_PTR_HPP_ */
