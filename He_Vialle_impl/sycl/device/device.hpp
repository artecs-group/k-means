#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace cl::sycl;

template <int unroll_factor>
class Trans;

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

sycl::queue get_queue(void);
template <int unroll_factor>
void transpose(T_real *odata, const T_real *idata, size_t m , size_t n);
void device_sync(void);
void device_init(void);
void device_finish(void);
void set_data_device(void);
void get_result_host(void);
void run_k_means(void);
