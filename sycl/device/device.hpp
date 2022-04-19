#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <vector>
#include <tuple>
#include <string>
#include <CL/sycl.hpp>

#define THREADS_EU 1
#define EUs_SUBSLICE_INTEL_GEN9 8
#define EUs_SUBSLICE_INTEL_GEN12 16
#define EUs_SUBSLICE_NVIDIA_PASCAL 128
#define CPU_PACKAGES 100

using namespace cl::sycl;

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

// Intel GPU
class IntelGpuSelector : public cl::sycl::device_selector {
    public:
        int operator()(const cl::sycl::device &Device) const override {
            const std::string vendor = Device.get_info<sycl::info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
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

        float *attributes{nullptr}, *mean{nullptr}, *sum{nullptr};
        unsigned int* counts{nullptr};
        int* assigments{nullptr};
        int dims{0}, k{0}, attribute_size{0}, attribute_bytes{0}, mean_bytes{0}, sum_size{0}, sum_bytes{0}, 
            count_bytes{0}, attribute_size_pad{0}, group_size{0}, work_items{0}, groups{0};

        sycl::queue _get_queue();
        void _sync();
        void _cpu_assign_clusters();
        void _gpu_assign_clusters();
        void _cpu_reduction();
        void _gpu_reduction();
        void _compute_mean();
        std::tuple<int,int,int> _get_group_work_items(int elements);
};

#endif