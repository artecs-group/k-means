#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <vector>
#include <tuple>
#include <string>
#include <CL/sycl.hpp>

using namespace cl::sycl;

// CUDA GPU selector
class CudaGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &device) const override {
            const std::string vendor = device.get_info<cl::sycl::info::device::vendor>();

            if (device.is_gpu() && (vendor.find("NVIDIA") != std::string::npos))
                return 1;

            return 0;
        }
};

// Intel GPU
class IntelGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &device) const override {
            const std::string vendor = device.get_info<cl::sycl::info::device::vendor>();

            if (device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
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

        float *attributes{nullptr}, *mean{nullptr}, *meanPrivate{nullptr};
        unsigned int *counts{nullptr}, *assigments{nullptr};
        int attribute_bytes{0}, mean_bytes{0}, count_bytes{0}, group_size{0}, work_items{0}, groups{0};

        cl::sycl::queue _get_queue();
        void _sync();
        void _assign_clusters_simd();
        void _assign_clusters_nvidia();
        void _assign_clusters_portable();
        void _cpu_reduction();
        void _intel_gpu_reduction();
        void _nvidia_reduction();
        void _portable_reduction();
        void _compute_mean();
        std::tuple<int,int,int> _get_group_work_items(int elements);
};

#endif
