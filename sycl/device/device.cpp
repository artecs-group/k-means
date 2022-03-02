#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include "./device.hpp"

Device::Device(int _k, std::vector<float>& h_x, std::vector<float>& h_y): k(_k){
    _queue = _get_queue();
    
    point_size     = h_x.size();
    std::tie (groups, group_size, work_items) = _get_group_work_items(point_size);

    point_size_pad = point_size + (work_items - point_size);
    point_bytes    = point_size * sizeof(float);
    mean_bytes     = k * sizeof(float);
    sum_size       = k * point_size;
    sum_bytes      = sum_size * sizeof(float);
    count_bytes    = sum_size * sizeof(float);

    point_x    = malloc_device<float>(point_size_pad * sizeof(float), _queue);
    point_y    = malloc_device<float>(point_size_pad * sizeof(float), _queue);
    mean_x     = malloc_device<float>(mean_bytes, _queue);
    mean_y     = malloc_device<float>(mean_bytes, _queue);
    sum_x      = malloc_device<float>(sum_bytes, _queue);
    sum_y      = malloc_device<float>(sum_bytes, _queue);
    counts     = malloc_device<float>(count_bytes, _queue);
    assigments = malloc_device<int>(point_size_pad * sizeof(int), _queue);

    // init pad values
    _queue.memset(point_x, 0, point_size_pad * sizeof(float));
    _queue.memset(point_y, 0, point_size_pad * sizeof(float));
    _sync();
    _queue.memcpy(point_x, h_x.data(), point_bytes);
    _queue.memcpy(point_y, h_y.data(), point_bytes);

    //shuffle points to random choose points
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, h_x.size() - 1);
    std::vector<float> h_mean_x, h_mean_y;
    for(int i{0}; i < k; i++) {
        int idx = indices(rng);
        h_mean_x.push_back(h_x[idx]);
        h_mean_y.push_back(h_y[idx]);
    }

    _queue.memcpy(mean_x, h_mean_x.data(), mean_bytes);
    _queue.memcpy(mean_y, h_mean_y.data(), mean_bytes);
    _queue.memset(sum_x, 0, sum_bytes);
    _queue.memset(sum_y, 0, sum_bytes);
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(assigments, 0, point_size_pad * sizeof(int)); // potential bug: try init to -1
    _sync();
}


Device::~Device() {
	if(point_x != nullptr)    free(point_x, _queue);
	if(point_y != nullptr)    free(point_y, _queue);
	if(mean_x != nullptr)     free(mean_x, _queue);
	if(mean_y != nullptr)     free(mean_y, _queue);
	if(sum_x != nullptr)      free(sum_x, _queue);
	if(sum_y != nullptr)      free(sum_y, _queue);
	if(counts != nullptr)     free(counts, _queue);
    if(assigments != nullptr) free(assigments, _queue);
}


sycl::queue Device::_get_queue() {
#if defined(INTEL_GPU_DEVICE)
	IntelGpuSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CudaGpuSelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue queue{selector};
	std::cout << "Running on \"" << queue.get_device().get_info<sycl::info::device::name>() << "\" under SYCL." << std::endl;
    return queue;
}


void Device::run_k_means(int iterations) {
    for (size_t i{0}; i < iterations; ++i) {
        _assign_clusters();
        _manage_reduction();
        _compute_mean();
    }
    _sync();
}


/*
    Case 1) elements <= max_group_size 
            * group_size = elements
            * work_items = elements
            * groups     = 1
    
    Case 2) elements > max_group_size
            * group_size = max_group_size
            * work_items = elements + group_size - (elements % group_size)
            * groups     = work_items / group_size

*/
std::tuple<int,int,int> Device::_get_group_work_items(int elements) {
#if defined(CPU_DEVICE)
// temporal fix, keeping the whole number crashes the execution
    const int max_group  = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>() >> 2;
#else
	const int max_group  = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
#endif
    const int group_size = (elements <= max_group) ? elements : max_group;
    const int work_items = (elements <= max_group) ? elements : elements + group_size - (elements % group_size);
    const int groups     = (elements <= max_group) ? 1 : work_items / group_size;

    return std::tuple<int,int,int>(groups, group_size, work_items);
}


void Device::_sync() {
    _queue.wait();
}


void Device::_assign_clusters() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(point_size);

    _queue.submit([&](handler& h) {
        int point_s = this-> point_size;
        int k = this->k;
        float* point_x = this->point_x;
        float* point_y = this->point_y;
        float* mean_x = this->mean_x;
        float* mean_y = this->mean_y;
        float* sum_x = this->sum_x;
        float* sum_y = this->sum_y;
        float* counts = this->counts;
        int* assigments = this->assigments;

        h.parallel_for<class assign_clusters>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);

            // Load once here.
            const float x_value = point_x[global_index];
            const float y_value = point_y[global_index];

            float best_distance = FLT_MAX;
            int best_cluster = -1;
            for (int cluster = 0; cluster < k; ++cluster) {
                const float distance = 
                    squared_l2_distance(x_value, y_value, mean_x[cluster], mean_y[cluster]);
                
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }
            assigments[global_index] = best_cluster;

            for(int cluster{0}; cluster < k; cluster++) {
                sum_x[point_s * cluster + global_index]  = (best_cluster == cluster) ? x_value : 0;
                sum_y[point_s * cluster + global_index]  = (best_cluster == cluster) ? y_value : 0;
                counts[point_s * cluster + global_index] = (best_cluster == cluster) ? 1 : 0;
            }
        });
    });
}


void Device::_reduce(float* vec) {
    _queue.submit([&](handler& h) {
        int point_size = this->point_size;
        int group_size = this->group_size;
        int k = this->k;
        int s_size = group_size * sizeof(float);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_data(s_size, h);

        h.parallel_for<class reduce>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
            const int local_index  = item.get_local_id(0);
            const int x            = local_index;

            for (int cluster = 0; cluster < k; ++cluster) {
                // load by cluster
                shared_data[x] = vec[point_size * cluster + global_index];
                item.barrier(sycl::access::fence_space::local_space);

                // apply reduction
                for (int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                    if (local_index < stride)
                        shared_data[x] += shared_data[x + stride];

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (local_index == 0) {  
                    const int cluster_index = point_size * cluster + item.get_group_linear_id();
                    vec[cluster_index]      = shared_data[x];
                }
            }
        });
    });
}


void Device::_manage_reduction() {
    int elements{point_size};
    std::tie (groups, group_size, work_items) = _get_group_work_items(elements);

    while (elements > k) {
        _reduce(this->sum_x);
        _reduce(this->sum_y);
        _reduce(this->counts);
        _sync();
        elements = groups*k;
        std::tie (groups, group_size, work_items) = _get_group_work_items(elements);
    }
}


void Device::_compute_mean() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(k);
    _queue.submit([&](handler& h) {
        int point_size = this->point_size;
        float* counts = this->counts;
        float* mean_x = this->mean_x;
        float* mean_y = this->mean_y;
        float* sum_x = this->sum_x;
        float* sum_y = this->sum_y;

        h.parallel_for<class compute_mean>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
            const int count        = (int)(1 < counts[point_size * global_index]) ? counts[point_size * global_index] : 1;
            mean_x[global_index]   = sum_x[point_size * global_index] / count;
            mean_y[global_index]   = sum_y[point_size * global_index] / count;
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean_x, std::vector<float>& h_mean_y) {
    _queue.memcpy(h_mean_x.data(), mean_x, mean_bytes);
    _queue.memcpy(h_mean_y.data(), mean_y, mean_bytes);
    _sync();
}


float squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
    float a = x_1 - x_2;
    float b = y_1 - y_2;
    return a*a + b*b;
}
