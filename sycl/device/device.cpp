#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include "./device.hpp"

Device::Device(int _k, std::vector<float>& h_x, std::vector<float>& h_y): k(_k){
    _queue = _get_queue();
    
    point_size     = h_x.size();
    point_bytes    = point_size * sizeof(float);
    mean_bytes     = k * sizeof(float);
    sum_size       = k * point_size;
    sum_bytes      = sum_size * sizeof(float);
    count_bytes    = sum_size * sizeof(int);

    point_x        = malloc_device<float>(point_size * sizeof(float), _queue);
    point_y        = malloc_device<float>(point_size * sizeof(float), _queue);
    mean_x         = malloc_device<float>(mean_bytes, _queue);
    mean_y         = malloc_device<float>(mean_bytes, _queue);
    sum_x          = malloc_shared<float>(sum_bytes, _queue);
    sum_y          = malloc_shared<float>(sum_bytes, _queue);
    counts         = malloc_shared<int>(count_bytes, _queue);
    assigments     = malloc_device<int>(point_size * sizeof(int), _queue);
    reduction_keys = malloc_device<int>(sum_size * sizeof(int), _queue);
    res_keys       = malloc_device<int>(k * sizeof(int), _queue);
    res_count      = malloc_device<int>(k * sizeof(int), _queue);
    res_x          = malloc_device<float>(k * sizeof(float), _queue);
    res_y          = malloc_device<float>(k * sizeof(float), _queue);

    _init_keys(reduction_keys, sum_size, point_size);

    // init pad values
    _queue.memset(point_x, 0, point_size * sizeof(float));
    _queue.memset(point_y, 0, point_size * sizeof(float));
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
    _queue.memset(assigments, 0, point_size * sizeof(int)); // potential bug: try init to -1
    _sync();
}


void Device::_init_keys(int* reduction_keys, int size, int range) {
    auto policy = oneapi::dpl::execution::make_device_policy(this->_queue);
    auto counting_begin = oneapi::dpl::counting_iterator<int>{0};

    // let keys_buf contain {0, 0, ..., 1, 1, ..., k-1, k-1, ..., k, k}
    std::transform(policy, counting_begin, counting_begin + size, reduction_keys,
        [range](auto i) { return i / range; });
}


Device::~Device() {
	if(point_x != nullptr)        free(point_x, _queue);
	if(point_y != nullptr)        free(point_y, _queue);
	if(mean_x != nullptr)         free(mean_x, _queue);
	if(mean_y != nullptr)         free(mean_y, _queue);
	if(sum_x != nullptr)          free(sum_x, _queue);
	if(sum_y != nullptr)          free(sum_y, _queue);
	if(counts != nullptr)         free(counts, _queue);
    if(assigments != nullptr)     free(assigments, _queue);
    if(reduction_keys != nullptr) free(reduction_keys, _queue);
    if(res_keys != nullptr)       free(res_keys, _queue);
    if(res_count != nullptr)      free(res_count, _queue);
    if(res_x != nullptr)          free(res_x, _queue);
    if(res_y != nullptr)          free(res_y, _queue);
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
    auto policy1 = oneapi::dpl::execution::make_device_policy<class reduce_x>(this->_queue);
    auto policy2 = oneapi::dpl::execution::make_device_policy<class reduce_y>(policy1);
    auto policy3 = oneapi::dpl::execution::make_device_policy<class reduce_count>(policy1);
    for (size_t i{0}; i < iterations; ++i) {
        // 1º Assign each point to a cluster
        _assign_clusters();

        // 2º apply reductions over vectors, to calculate how much points are by cluster 
        oneapi::dpl::reduce_by_segment(policy1, this->reduction_keys, this->reduction_keys + this->sum_size, 
            this->sum_x, this->res_keys, this->res_x, std::equal_to<float>(),std::plus<float>());

        oneapi::dpl::reduce_by_segment(policy2, this->reduction_keys, this->reduction_keys + this->sum_size, 
            this->sum_y, this->res_keys, this->res_y, std::equal_to<float>(),std::plus<float>());

        oneapi::dpl::reduce_by_segment(policy3, this->reduction_keys, this->reduction_keys + this->sum_size, 
            this->counts, this->res_keys, this->res_count, std::equal_to<int>(),std::plus<int>());

        // 3º Calculate the new means for each cluster
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
        int* counts = this->counts;
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


void Device::_compute_mean() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(k);
    _queue.submit([&](handler& h) {
        int* counts = this->res_count;
        float* mean_x = this->mean_x;
        float* mean_y = this->mean_y;
        float* sum_x = this->res_x;
        float* sum_y = this->res_y;

        h.parallel_for<class compute_mean>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
            const int count        = (1 < counts[global_index]) ? counts[global_index] : 1;
            mean_x[global_index]   = sum_x[global_index] / count;
            mean_y[global_index]   = sum_y[global_index] / count;
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
