#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include "./device.hpp"


float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


Device::Device(int _k, int _dims, int n_attrs, std::vector<float>& h_attrs): k(_k), dims(_dims){
    _queue = _get_queue();
    
    attribute_size     = n_attrs;
    std::tie (groups, group_size, work_items) = _get_group_work_items(attribute_size);

    attribute_size_pad = attribute_size + (work_items - attribute_size);
    attribute_bytes    = attribute_size * dims * sizeof(float);
    mean_bytes         = k * dims * sizeof(float);
    sum_size           = k * dims * attribute_size;
    sum_bytes          = sum_size * sizeof(float);
    count_bytes        = k * attribute_size * sizeof(unsigned int);

    attributes = malloc_device<float>(attribute_size_pad * dims * sizeof(float), _queue);
    mean       = malloc_device<float>(mean_bytes, _queue);
    sum        = malloc_device<float>(sum_bytes, _queue);
    counts     = malloc_device<unsigned int>(count_bytes, _queue);
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
	if(attributes != nullptr) free(attributes, _queue);
	if(mean != nullptr)       free(mean, _queue);
	if(sum != nullptr)        free(sum, _queue);
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
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float t_assign{0}, t_reduction{0}, t_mean{0};
    for (size_t i{0}; i < iterations; ++i) {
        start = std::chrono::high_resolution_clock::now();
        _assign_clusters(); 
        _sync();
        end = std::chrono::high_resolution_clock::now();
        t_assign += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
#if defined(CPU_DEVICE)
        _cpu_reduction();
#else
        _gpu_reduction();
#endif
        _sync();
        end = std::chrono::high_resolution_clock::now();
        t_reduction += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        _compute_mean();
        _sync();
        end = std::chrono::high_resolution_clock::now();
        t_mean += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    }
    double total = t_assign + t_reduction + t_mean;
    std::cout << std::endl << "Kernel time: " << std::endl
              << "  * Assign Clusters = " << t_assign << " (s) -> " << t_assign/total*100 << "%" << std::endl
              << "  * Reduction       = " << t_reduction << " (s) -> " << t_reduction/total*100 << "%" << std::endl
              << "  * Mean            = " << t_mean << " (s) -> " << t_mean/total*100 << "%" << std::endl;
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
        int attrs_size = this-> attribute_size;
        int k = this->k;
        int dims = this->dims;
        float* attrs = this->attributes;
        float* mean = this->mean;
        float* sum = this->sum;
        unsigned int* counts = this->counts;
        int* assigments = this->assigments;

        h.parallel_for<class assign_clusters>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);

            float best_distance{FLT_MAX};
            int best_cluster{-1};
            float distance{0};
            for (int cluster = 0; cluster < k; ++cluster) {
                for(int d{0}; d < dims; d++)
                    distance += squared_l2_distance(attrs[(d * attrs_size) + global_index], mean[(cluster * dims) + d]);
                
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
                distance = 0;
            }
            assigments[global_index] = best_cluster;

            int val_id{0}, sum_id{0};
            for(int cluster{0}; cluster < k; cluster++) {
                for(int d{0}; d < dims; d++) {
                    val_id       = (d * attrs_size) + global_index;
                    sum_id       = attrs_size * cluster * dims + val_id;
                    sum[sum_id]  = (best_cluster == cluster) ? attrs[val_id] : 0;
                }
                counts[attrs_size * cluster + global_index] = (best_cluster == cluster) ? 1 : 0;
            }
        });
    });
}


void Device::_gpu_reduction() {
#if defined(INTEL_GPU_DEVICE)
	const size_t CUs       = THREADS_EU * EUs_SUBSLICE_INTEL_GEN9;
#elif defined(NVIDIA_DEVICE)
	const size_t CUs       = THREADS_EU * EUs_SUBSLICE_NVIDIA_PASCAL;
#elif defined(CPU_DEVICE)	
	const size_t CUs       = THREADS_EU * _queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
#endif
    const int attrs_per_CU = attribute_size / CUs;
    const int remaining    = attribute_size % CUs;

    _queue.submit([&](handler& h) {
        int attrs_size       = this->attribute_size;
        int k                = this->k;
        int dims             = this->dims;
        float* sums          = this->sum;
        unsigned int* counts = this->counts;
        int s_size           = CUs * sizeof(float);
        int c_size           = CUs * sizeof(unsigned int);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_sum(s_size, h);
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_count(c_size, h);

        h.parallel_for(nd_range(range(k, dims, CUs), range(1, 1, CUs)), [=](nd_item<3> item){
            const int cluster        = item.get_global_id(0);
            const int d              = item.get_global_id(1);
            const int local_idx      = item.get_local_id(2);
            const int attr_start_idx = attrs_per_CU * local_idx;
            const int n_attrs        = (local_idx == CUs-1) ? attrs_per_CU + remaining : attrs_per_CU;
            int sum{0}, counter{0};

            // load all elements by thread
            sum = 0;
            counter = 0;
            for(int i{attr_start_idx}; i < attr_start_idx + n_attrs; i++) {
                sum     += sums[(attrs_size * cluster * dims) + d * attrs_size + i];
                counter += counts[attrs_size * cluster + i];
            }
            
            shared_sum[local_idx]   = sum;
            shared_count[local_idx] = counter;

            // tree reduction
            for (int stride = item.get_local_range(2) >> 1; stride > 0; stride >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if (local_idx < stride) {
                    shared_sum[local_idx]   += shared_sum[local_idx + stride];
                    shared_count[local_idx] += shared_count[local_idx + stride];
                }
            }

            if (local_idx == 0) {
                sums[(attrs_size * cluster * dims) + d * attrs_size] = shared_sum[0];
                counts[attrs_size * cluster]                         = shared_count[0];
            }
        });
    });
}


void Device::_cpu_reduction() {
    const size_t CUs       = _queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    const int attrs_per_CU = attribute_size / CUs;
    const int remaining    = attribute_size % CUs;

    _queue.submit([&](handler &h) {
        int attrs_size = this->attribute_size;
        int k = this->k;
        int dims = this->dims;
        float* sum = this->sum;
        unsigned int* count = this->counts;

        h.parallel_for(nd_range(range(CUs), range(1)), [=](nd_item<1> item){
            const int global_idx     = item.get_global_id(0);
            const int attr_start_idx = attrs_per_CU * global_idx;
            const int n_attrs        = (global_idx == CUs-1) ? attrs_per_CU + remaining : attrs_per_CU;
            int sum_id{0}, count_id{0};

            for (int cluster{0}; cluster < k; cluster++) {
                count_id = (attrs_size * cluster) + attr_start_idx;

                for(int i{1}; i < n_attrs; i++) {
                    count[count_id] += count[count_id + i];

                    for(int d{0}; d < dims; d++) {
                        sum_id       = (attrs_size * cluster * dims) + d * attrs_size + attr_start_idx;
                        sum[sum_id] += sum[sum_id + i];
                    }
                }
            }
        });
    });
    _sync();

    // serial reduction
    // reduce all previous reductions
    _queue.submit([&](handler &h) {
        int attrs_size = this->attribute_size;
        int k = this->k;
        int dims = this->dims;
        unsigned int* counts = this->counts;
        float* sum = this->sum;

        h.single_task([=]() {
            int cluster_id{0}, dim_id{0};

            for(int cluster{0}; cluster < k; cluster++){
                cluster_id = (attrs_size * cluster);
                for(int i{attrs_per_CU}; i < attrs_size; i += attrs_per_CU) {
                    counts[cluster_id] += counts[cluster_id + i];
                    
                    for(int d{0}; d < dims; d++) {
                        dim_id       = (attrs_size * cluster * dims) + d * attrs_size;
                        sum[dim_id] += sum[dim_id + i];
                    }
                }
            }
        });
    });
}


void Device::_compute_mean() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(k);
    _queue.submit([&](handler& h) {
        int dims = this->dims;
        int attrs_size = this->attribute_size;
        unsigned int* counts = this->counts;
        float* mean = this->mean;
        float* sum = this->sum;

        h.parallel_for<class compute_mean>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
            const int count        = (1 < counts[attrs_size * global_index]) ? counts[attrs_size * global_index] : 1;
            for(int d{0}; d < dims; d++) {
                int id = (global_index * attrs_size * dims) + (d * attrs_size);
                mean[global_index * dims + d] = sum[id] / count;
            }
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean) {
    _queue.memcpy(h_mean.data(), mean, mean_bytes);
    _sync();
}
