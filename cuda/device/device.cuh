#ifndef _DEVICE_K_MEANS_
#define _DEVICE_K_MEANS_

#include <vector>
#include <tuple>
#include <string>

class Device {
    public:
        Device(int _k, int _dims, int n_attrs, std::vector<float>& h_attrs);
        ~Device();
        void run_k_means(int iterations);
        void save_solution(std::vector<float>& h_mean);

    private:
        cudaDeviceProp _gpu_props;
        float *attributes{nullptr}, *mean{nullptr}, *sum{nullptr};
        unsigned int* counts{nullptr};
        int* assigments{nullptr};
        int dims{0}, k{0}, attributes_size{0}, attributes_bytes{0}, mean_bytes{0}, sum_size{0}, sum_bytes{0}, 
            count_bytes{0}, attributes_size_pad{0}, threads{0}, work_items{0}, blocks{0};

        void _select_device();
        void _sync();
        void _assign_clusters();
        void _manage_reduction();

        template <typename T>
        void _reduce(T* vec, size_t _dims, size_t dim_offset);
        void _compute_mean();
        std::tuple<int,int,int> _get_block_threads(int elements);
};

#endif