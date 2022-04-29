#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <dpct/dpct.hpp>
#include "./device.hpp"


inline float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


Device::Device(int _k, int _dims, int length, std::vector<float>& h_attrs): k(_k), dims(_dims){
    _queue = _get_queue();
    const int group_size = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();

    attribute_size     = length;
    attribute_bytes    = attribute_size * dims * sizeof(float);
    mean_bytes         = k * dims * sizeof(float);
    count_bytes        = k * sizeof(unsigned int);

    attributes    = malloc_device<float>(attribute_bytes, _queue);
    mean_package  = malloc_device<float>(group_size * mean_bytes, _queue);
    mean          = malloc_device<float>(mean_bytes, _queue);
    counts        = malloc_device<unsigned int>(count_bytes, _queue);
    count_package = malloc_device<unsigned int>(group_size * count_bytes, _queue);
    assigments    = malloc_device<int>(attribute_size * sizeof(int), _queue);

    _queue.memcpy(attributes, h_attrs.data(), attribute_bytes);

    //shuffle attributes to random choose attributes
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, attribute_size - 1);
    std::vector<float> h_mean;
    for(int i{0}; i < k; i++) {
        int idx = indices(rng);
        for(int j{0}; j < dims; j++)
            h_mean.push_back(h_attrs[idx * dims + j]);
    }

    _queue.memcpy(mean, h_mean.data(), mean_bytes);
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(assigments, 0, attribute_size * sizeof(int));

    _sync();
}


Device::~Device() {
	if(attributes != nullptr) free(attributes, _queue);
	if(mean != nullptr)       free(mean, _queue);
	if(counts != nullptr)     free(counts, _queue);
    if(assigments != nullptr) free(assigments, _queue);
    if(mean_package != nullptr) free(mean_package, _queue);
    if(count_package != nullptr) free(count_package, _queue);
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
    constexpr int B          = 2;
    const int simd_width     = 8; //check that simd_width < dims
    const int simd_reminder  = dims % simd_width;
    const int group_size     = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    const int attrs_per_pckg = attribute_size / ASSIGN_PACKAGES;
    const int remainder_pckg = attribute_size % ASSIGN_PACKAGES;

    _queue.submit([&](handler& h) {
        int k = this->k;
        int dims = this->dims;
        float* attrs = this->attributes;
        float* mean = this->mean;
        int* assigments = this->assigments;

        using global_ptr = sycl::multi_ptr<float, sycl::access::address_space::global_space>;

        h.parallel_for<class assign_clusters>(nd_range(range(ASSIGN_PACKAGES), range(1)), [=](nd_item<1> item){
            int global_idx = item.get_global_id(0);
            int offset     = attrs_per_pckg * global_idx;
            int length     = (global_idx == ASSIGN_PACKAGES-1) ? attrs_per_pckg + remainder_pckg : attrs_per_pckg;

            sycl::vec<float, simd_width> v_attr, v_mean;
            sycl::vec<float, simd_width> result[B];
            float best_distance{FLT_MAX};
            int best_cluster{-1};
            float distance{0};

            for(int i = offset; i < offset + length; i++) {
                best_distance = FLT_MAX;
                best_cluster  = -1;

                for (int cluster = 0; cluster < k; cluster++) {
                    //vector var initialization
                    //for (size_t i = 0; i < simd_width; i++)
                    for (size_t j = 0; j < B; j++)
                        result[j] = {0};

                    // calculate simd squared_l2_distance
                    for(int d{0}; d < dims / simd_width; d += B) {
                        for(int j{0}; j < B; j++) {
                            v_attr.load(0, global_ptr(&attrs[(i * dims) + d + j]));
                            v_mean.load(0, global_ptr(&mean[(cluster * dims) + d + j]));
                            v_attr    = v_attr - v_mean;
                            result[j] = result[j] + v_attr * v_attr;
                        }
                    }

                    for (size_t i = 1; i < B; i++)
                        result[0] += result[i];
                    
                    // reduce simd lane in scalar
                    for(int i{0}; i < simd_width; i++)
                        distance += result[0][i];

                    // calculate remaining values
                    for(int d{dims - simd_reminder}; d < dims; d++)
                        distance += squared_l2_distance(attrs[(i * dims) + d], mean[(cluster * dims) + d]);

                    best_distance = sycl::min<float>(distance, best_distance);
                    best_cluster  = distance == best_distance ? cluster : best_cluster;
                    distance      = 0;
                }
                assigments[i] = best_cluster;
            }
        });
    });
}


void Device::_gpu_reduction() {
    const int group_size     = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    const int attrs_per_pckg = attribute_size / REDUCTION_PACKAGES;
    const int remainder_pckg = attribute_size % REDUCTION_PACKAGES;

    // clean matrices
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(mean, 0, mean_bytes);
    _sync();

    _queue.submit([&](handler& h) {
        int attrs_size              = this->attribute_size;
        int k                       = this->k;
        int dims                    = this->dims;
        float* attrs                = this->attributes;
        float* mean                 = this->mean;
        int* assigments             = this->assigments;
        unsigned int* counts        = this->counts;
        unsigned int* count_package = this->count_package;
        float* mean_package         = this->mean_package;

        h.parallel_for<class gpu_red>(nd_range(range(REDUCTION_PACKAGES*group_size), range(group_size)), [=](nd_item<1> item){
            const int grp_idx      = item.get_group(0);
            const int global_idx   = item.get_global_id(0);
            const int local_idx    = item.get_local_id(0);
            const int offset_gr    = attrs_per_pckg * grp_idx;
            const int length_gr    = (global_idx == REDUCTION_PACKAGES-1) ? attrs_per_pckg + remainder_pckg : attrs_per_pckg;
            const int attrs_per_wi = length_gr / group_size;
            const int remainder_wi = length_gr % group_size;
            const int offset       = offset_gr + attrs_per_wi * local_idx;
            const int length       = (local_idx == group_size-1) ? attrs_per_wi + remainder_wi : attrs_per_wi;

            // init memory each work item inits its own memory section
            for(int cluster{0}; cluster < k; cluster++) {
                count_package[local_idx * k + cluster] = 0;
                for(int d{0}; d < dims; d++)
                    mean_package[local_idx * k * dims + cluster * dims + d] = 0;
            }

            // perform work item private sum 
            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                count_package[local_idx * k + cluster]++;
                for(int d{0}; d < dims; d++)
                    mean_package[local_idx * k * dims + cluster * dims + d] += attrs[i * dims + d];
            }

            // perform work group local sum with a tree reduction
            for(int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if(local_idx < stride) {
                    for(int cluster{0}; cluster < k; cluster++) {
                        count_package[local_idx * k + cluster] += count_package[(local_idx + stride) * k + cluster];
                        for(int d{0}; d < dims; d++)
                            mean_package[local_idx * k * dims + cluster * dims + d] += mean_package[(local_idx + stride) * k * dims + cluster * dims + d];
                    }   
                }
            }

            // perform global sum
            if(local_idx == 0) {
                for(int cluster{0}; cluster < k; cluster++) {
                    sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&counts[cluster])).fetch_add(count_package[cluster]);
                    for(int d{0}; d < dims; d++)
                        dpct::atomic_fetch_add(&mean[cluster * dims + d], mean_package[cluster * dims + d]); 
                }
            }
        });
    });
}


void Device::_cpu_reduction() {
    const int simd_width    = 8; //check that simd_width < dims
    const int simd_reminder = dims % simd_width;
    const int attrs_per_CU  = attribute_size / REDUCTION_PACKAGES;
    const int remaining     = attribute_size % REDUCTION_PACKAGES;

    // clean matrices
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(mean, 0, mean_bytes);
    _sync();

    _queue.submit([&](handler &h) {
        int attrs_size      = this->attribute_size;
        int k               = this->k;
        int dims            = this->dims;
        unsigned int* count = this->counts;
        float* attrs        = this->attributes;
        float* mean         = this->mean;
        int* assigments     = this->assigments;
        int p_size          = k * dims * sizeof(float);
        int c_size          = k * sizeof(unsigned int);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> package(p_size, h);
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::local> p_count(c_size, h);

        h.parallel_for<class cpu_red>(nd_range(range(REDUCTION_PACKAGES), range(1)), [=](nd_item<1> item){
            const int global_idx = item.get_global_id(0);
            const int offset     = attrs_per_CU * global_idx;
            const int length     = (global_idx == REDUCTION_PACKAGES-1) ? attrs_per_CU + remaining : attrs_per_CU;
            sycl::vec<float, simd_width> v_pckg, result;

            using global_ptr = sycl::multi_ptr<float, sycl::access::address_space::global_space>;
            using local_ptr  = sycl::multi_ptr<float, sycl::access::address_space::local_space>;

            for(int i{0}; i < k; i++) {
                p_count[i] = 0.0;
                for(int j{0}; j < dims; j++)
                    package[i*dims + j] = 0.0;
            }

            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                p_count[cluster]++;
                for(int j{simd_width}; j < dims; j += simd_width) {
                    int d = j - simd_width;
                    v_pckg.load(0, global_ptr(&attrs[i * dims + d]));
                    result = v_pckg;
                    v_pckg.load(0, local_ptr(&package[cluster * dims + d]));
                    result += v_pckg;
                    result.store(0, local_ptr(&package[cluster * dims + d]));
                }
                // calculate remaining dims
                for(int d{dims - simd_reminder}; d < dims; d++)
                    package[cluster * dims + d] += attrs[i * dims + d];
            }

            for(int cluster{0}; cluster < k; cluster++) {
                sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&count[cluster])).fetch_add(p_count[cluster]);
                for(int d{0}; d < dims; d++)
                    dpct::atomic_fetch_add(&mean[cluster * dims + d], package[cluster * dims + d]); 
            }
        });
    });
}


void Device::_compute_mean() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(k);
    _queue.submit([&](handler& h) {
        int dims = this->dims;
        unsigned int* counts = this->counts;
        float* mean = this->mean;

        h.parallel_for<class compute_mean>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
            const int count        = (0 < counts[global_index]) ? counts[global_index] : 1;
            for(int d{0}; d < dims; d++)
                mean[global_index * dims + d] /= count;
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean) {
    _queue.memcpy(h_mean.data(), mean, mean_bytes);
    _sync();
}
