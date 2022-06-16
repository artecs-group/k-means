#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <vector>
#include <tuple>
#include <string>

#define ATTRIBUTE_SIZE 2458285
#define DIMS           68
#define K              4
#define ITERATIONS     20 

#define RED_ATTRS_PACK_NVIDIA    128
#define RED_DIMS_PACK_NVIDIA     4
#define ASSIGN_BLOCK_SIZE_NVIDIA 128

class Device {
    public:
        Device(std::vector<float>& h_attrs);
        ~Device();
        void run_k_means();
        void save_solution(std::vector<float>& h_mean);

    private:
        cudaDeviceProp _gpu_props;
        float *attributes{nullptr}, *mean{nullptr};
        unsigned int *counts{nullptr}, *assigments{nullptr};
        int attributes_bytes{0}, mean_bytes{0}, count_bytes{0}, threads{0}, work_items{0}, blocks{0};

        void _select_device();
        void _sync();
        void _assign_clusters();
        void _reduction();
        void _compute_mean();
        std::tuple<int,int,int> _get_block_threads(int elements);
};

#endif