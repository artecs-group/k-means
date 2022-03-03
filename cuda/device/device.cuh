#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <vector>
#include <tuple>
#include <string>

class Device {
    public:
        float *point_x{nullptr}, *point_y{nullptr}, *mean_x{nullptr}, *mean_y{nullptr}, 
            *sum_x{nullptr}, *sum_y{nullptr}, *counts{nullptr};
        int* assigments{nullptr};
        int k{0}, point_size{0}, point_bytes{0}, mean_bytes{0}, sum_size{0}, sum_bytes{0}, 
            count_bytes{0}, point_size_pad{0}, threads{0}, work_items{0}, blocks{0};

        Device(int _k, std::vector<float>& h_x, std::vector<float>& h_y);
        ~Device();
        void run_k_means(int iterations);
        void save_solution(std::vector<float>& h_mean_x, std::vector<float>& h_mean_y);
    private:
        cudaDeviceProp _gpu_props;
        void _select_device();
        void _sync();
        void _assign_clusters();
        void _manage_reduction();
        void _reduce(float* vec);
        void _compute_mean();
        std::tuple<int,int,int> _get_block_threads(int elements);
};

#endif