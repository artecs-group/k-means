#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include "./device.hpp"

Device::Device(int _k, int _dims, int n_attrs, std::vector<float>& h_attrs): k(_k), dims(_dims){
    _queue = _get_queue();
    
    attribute_size     = n_attrs;
    std::tie (groups, group_size, work_items) = _get_group_work_items(attribute_size);

    attribute_size_pad = attribute_size + (work_items - attribute_size);
    attribute_bytes    = attribute_size * dims * sizeof(float);
    mean_bytes         = k * dims * sizeof(float);
    sum_size           = k * attribute_size;
    sum_bytes          = sum_size * dims * sizeof(float);
    count_bytes        = sum_size * sizeof(float);

    attributes = malloc_device<float>(attribute_size_pad * dims * sizeof(float), _queue);
    mean       = malloc_device<float>(mean_bytes, _queue);
    sum        = malloc_device<float>(sum_bytes, _queue);
    counts     = malloc_device<float>(count_bytes, _queue);
    assigments = malloc_device<int>(attribute_size_pad * sizeof(int), _queue);

    // init pad values
    _queue.memset(attributes, 0, attribute_size_pad * dims * sizeof(float));
    _sync();
    _queue.memcpy(attributes, h_attrs.data(), attribute_bytes);

    //shuffle attributes to random choose attributes
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, attribute_size - 1);
    std::vector<float> h_mean;
    for(int i{0}; i < k; i++) {
        int idx = indices(rng);
        for(int j{0}; j < dims; j++)
            h_mean.push_back(h_attrs[idx + j * attribute_size]);
    }

    _queue.memcpy(mean, h_mean.data(), mean_bytes);
    _queue.memset(sum, 0, sum_bytes);
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(assigments, 0, attribute_size_pad * sizeof(int)); // potential bug: try init to -1
    _sync();
}


Device::~Device() {
	if(attributes != nullptr)    free(attributes, _queue);
	if(mean != nullptr)     free(mean, _queue);
	if(sum != nullptr)      free(sum, _queue);
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
    std::tie (groups, group_size, work_items) = _get_group_work_items(attribute_size);

    _queue.submit([&](handler& h) {
        int attr_size = this-> attribute_size;
        int k = this->k;
        int dims = this->dims;
        float* attrs = this->attributes;
        float* mean = this->mean;
        float* sum = this->sum;
        float* counts = this->counts;
        int* assigments = this->assigments;

        h.parallel_for<class assign_clusters>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);

            float best_distance{FLT_MAX};
            int best_cluster{-1};
            float distance{0};
            for (int cluster = 0; cluster < k; ++cluster) {
                for(int d{0}; d < dims; d++)
                    distance += squared_l2_distance(attrs[(d * attr_size) + global_index], mean[(cluster * dims) + d]);
                
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }
            assigments[global_index] = best_cluster;

            for(int cluster{0}; cluster < k; cluster++) {
                for(int d{0}; d < dims; d++) {
                    int val_id = (d * attr_size) + global_index;
                    int sum_id = attr_size * cluster * dims + val_id;
                    sum[sum_id]  = (best_cluster == cluster) ? attrs[val_id] : 0;
                }
                counts[attr_size * cluster + global_index] = (best_cluster == cluster) ? 1 : 0;
            }
        });
    });
}


void Device::_reduce(float* vec) {
    _queue.submit([&](handler& h) {
        int dims = this->dims;
        int attrs_size = this->attribute_size;
        int group_size = this->group_size;
        int k = this->k;
        int s_size = group_size * sizeof(float);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_data(s_size, h);

        h.parallel_for<class reduce>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
            const int local_index  = item.get_local_id(0);
            const int x            = local_index;

            for (int cluster = 0; cluster < k; ++cluster) {
                for(int d{0}; d < dims; d++) {
                    // load by cluster
                    shared_data[x] = vec[(attrs_size * dims * cluster) + (d * attr_size) + global_index];
                    item.barrier(sycl::access::fence_space::local_space);

                    // apply reduction
                    for (int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                        if (local_index < stride)
                            shared_data[x] += shared_data[x + stride];

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (local_index == 0) {  
                        int id = (attrs_size * cluster * dims) + (d * attr_size) + item.get_group_linear_id();
                        vec[id]      = shared_data[x];
                    }
                }
            }
        });
    });
}


void Device::_manage_reduction() {
    int elements{attribute_size};
    std::tie (groups, group_size, work_items) = _get_group_work_items(elements);

    while (elements > k) {
        _reduce(this->sum);
        _reduce(this->counts);
        _sync();
        elements = groups*k;
        std::tie (groups, group_size, work_items) = _get_group_work_items(elements);
    }
}


void Device::_compute_mean() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(k);
    _queue.submit([&](handler& h) {
        int dims = this->dims;
        int attrs_size = this->attribute_size;
        float* counts = this->counts;
        float* mean = this->mean;
        float* sum = this->sum;

        h.parallel_for<class compute_mean>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
            const int count        = (int)(1 < counts[attrs_size * global_index]) ? counts[attrs_size * global_index] : 1;
            for(int d{0}; d < dims; d++) {
                int id = (global_index * attrs_size * dims) + (d * attrs_size);
                mean[global_index * dims + d] = sum[id] / count;
            }
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean_x, std::vector<float>& h_mean_y) {
    _queue.memcpy(h_mean.data(), mean, mean_bytes);
    _sync();
}


float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}
