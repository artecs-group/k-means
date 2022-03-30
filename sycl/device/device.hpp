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
        float *point_x{nullptr}, *point_y{nullptr}, *mean_x{nullptr}, *mean_y{nullptr}, 
            *sum_x{nullptr}, *sum_y{nullptr}, *res_x{nullptr}, *res_y{nullptr};
        int* assigments{nullptr}, *counts{nullptr}, *reduction_keys{nullptr}, *res_keys{nullptr}, *res_count{nullptr};
        int k{0}, point_size{0}, point_bytes{0}, mean_bytes{0}, sum_size{0}, sum_bytes{0}, 
            count_bytes{0}, group_size{0}, work_items{0}, groups{0};

        Device(int _k, std::vector<float>& h_x, std::vector<float>& h_y);
        ~Device();
        void run_k_means(int iterations);
        void save_solution(std::vector<float>& h_mean_x, std::vector<float>& h_mean_y);
    private:
        sycl::queue _queue;

        sycl::queue _get_queue();
        void _sync();
        void _assign_clusters();
        void _compute_mean();
        std::tuple<int,int,int> _get_group_work_items(int elements);
        void _init_keys(int* reduction_keys, int size, int range);
};


float squared_l2_distance(float x_1, float y_1, float x_2, float y_2);

#endif