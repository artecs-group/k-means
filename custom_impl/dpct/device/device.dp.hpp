#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <vector>
#include <tuple>
#include <string>

#define RED_ATTRS_PACK_NVIDIA 128
/*
DPCT1083:4: The size of local memory in the migrated code may be different from
the original code. Check that the allocated memory size in the migrated code is
correct.
*/
#define RED_DIMS_PACK_NVIDIA 4
#define ASSIGN_BLOCK_SIZE_NVIDIA 128

// CUDA GPU selector
class CudaGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string DriverVersion = Device.get_info<sycl::info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                return 1;

            return 0;
        }
};

class Device {
    public:
        Device(int _k, int _dims, int n_attrs, std::vector<float>& h_attrs);
        ~Device();
        void run_k_means(int iterations);
        void save_solution(std::vector<float>& h_mean);

    private:
        sycl::queue _queue;
        float *attributes{nullptr}, *mean{nullptr};
        unsigned int *counts{nullptr}, *assigments{nullptr};
        int dims{0}, k{0}, attributes_size{0}, attributes_bytes{0}, mean_bytes{0}, 
            count_bytes{0}, threads{0}, work_items{0}, blocks{0};

        sycl::queue _select_device();
        void _sync();
        void _assign_clusters();
        void _reduction();
        void _compute_mean();
        std::tuple<int,int,int> _get_block_threads(int elements);
};

#endif