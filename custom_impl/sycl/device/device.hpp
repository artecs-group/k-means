#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <vector>
#include <tuple>
#include <string>
#include <CL/sycl.hpp>

#define ATTRIBUTE_SIZE 2458285
#define DIMS           68
#define K              4
#define ITERATIONS     20

#define RED_ATTRS_PACK_NVIDIA 128
#define RED_DIMS_PACK_NVIDIA 4
#define RED_ATTRS_PACK 16
#define RED_DIMS_PACK_IGPU 4
#define RED_GROUP_SIZE_IGPU 256
#define RED_SIMD_WIDTH 4

#define ASSIGN_BLOCK_SIZE_NVIDIA 128
#define ASSIGN_PACK 3
#define ASSIGN_GROUP_SIZE_CPU 1
#define ASSIGN_GROUP_SIZE_IGPU 256
#define ASSIGN_SIMD_WIDTH 4

using namespace cl::sycl;

// CUDA GPU selector
class CudaGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string DriverVersion = Device.get_info<cl::sycl::info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                return 1;

            return 0;
        }
};

// Intel GPU
class IntelGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string vendor = Device.get_info<cl::sycl::info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
                return 1;

            return 0;
        }
};


class Device {
    public:
        Device(std::vector<float>& h_attrs);
        ~Device();
        void run_k_means();
        void save_solution(std::vector<float>& h_mean);

    private:
        cl::sycl::queue _queue;

        float *attributes{nullptr}, *mean{nullptr};
        unsigned int *counts{nullptr}, *assigments{nullptr};
        int attribute_bytes{0}, mean_bytes{0}, count_bytes{0}, group_size{0}, work_items{0}, groups{0};

        cl::sycl::queue _get_queue();
        void _sync();
        void _assign_clusters();
        void _assign_clusters_nvidia();
        void _cpu_reduction();
        void _intel_gpu_reduction();
        void _nvidia_reduction();
        void _compute_mean();
        std::tuple<int,int,int> _get_group_work_items(int elements);
};

#endif
