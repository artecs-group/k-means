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
        void _assign_clusters();
        void _manage_cpu_reduction();
        void _manage_gpu_reduction();

        template <typename T>
        void _gpu_reduce(T* vec, size_t dims, size_t dim_offset);

        template <typename T>
        void _cpu_reduce(T* vec, size_t dims, size_t dim_offset);

        void _compute_mean();
        std::tuple<int,int,int> _get_group_work_items(int elements);
};

#endif