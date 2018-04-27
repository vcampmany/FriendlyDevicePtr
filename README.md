# FriendlyDevicePtr

The `DevicePtr` class takes care of managing the memory of CUDA pointers. It eases the allocation process of CUDA memory and leads to cleaner and shorter code. The user doesn't need to take care of any deallocation reducing the risk of memory leaks. It works with the same principle as the C++ `unique_ptr`, a memory region must just be referenced by only one pointer.

## Requirements
The class is self-contained in the `DevicePtr.hpp` file and it has no dependencies. The only requirement is to have a CUDA installation.

## Examples
The following example shows the most straightforward usage of the class:
```
sdt::vector<float> a;
...
{
    dptr::DevicePtr<float> d_a(a.data(), a.size());
    ...
    kernel<<<grid, block>>>(d_a.get(), d_a.getCount());
    cudaMemcpy(a.data(), d_a.get(), d_a.getBytes(), cudaMemcpyDeviceToHost);
    ...
} // The CUDA memory will be deallocated here, at scope exit

processVector(a); 
...
```
In the `examples` directory one can find several examples with more use cases and detailed explanations. To compile the examples simply run:
```
cd `project_root`
make all
```

## Contributions
Contributions, suggestions, bug reporting are very welcome :blush:
