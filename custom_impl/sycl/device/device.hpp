#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <vector>
#include <tuple>
#include <string>
#include <CL/sycl.hpp>

#define ATTRIBUTE_SIZE 2458285
#define DIMS           68
#define K              256
#define ITERATIONS     20

#define RED_ATTRS_PACK_NVIDIA 128
#define RED_DIMS_PACK_NVIDIA 4
#define RED_ATTRS_PACK 64
#define RED_SIMD_WIDTH 8

#define ASSIGN_BLOCK_SIZE_NVIDIA 128
#define ASSIGN_PACK 16
#define ASSIGN_GROUP_SIZE_CPU 1
#define ASSIGN_SIMD_WIDTH 1

using namespace cl::sycl;

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
        void _assign_clusters();
        void _cpu_reduction();
        void _intel_gpu_reduction();
        void _nvidia_reduction();
        void _common_reduction();
        void _compute_mean();
        std::tuple<int,int,int> _get_group_work_items(int elements);
};

#endif
