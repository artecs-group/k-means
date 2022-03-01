
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
        float *point_x{nullptr}, *point_y{nullptr}, *mean_x{nullptr}, *mean_y{nullptr}, *sum_x{nullptr}, *sum_y{nullptr};
        int* counts{nullptr};
        int k{0}, point_size{0}, point_bytes{0}, mean_bytes{0}, sum_size{0}, sum_bytes{0}, count_bytes{0}, point_size_pad{0};

        Device(int _k, std::vector<float>& h_x, std::vector<float>& h_y);
        ~Device();
        void sync();
        void fine_reduce();
        void coarse_reduce();
        void save_solution(std::vector<float>& h_mean_x, std::vector<float>& h_mean_y);
    private:
        sycl::queue _queue;

        sycl::queue _get_queue();
        std::tuple<int,int,int> _get_group_work_items(int elements);
};


float squared_l2_distance(float x_1, float y_1, float x_2, float y_2);

#endif