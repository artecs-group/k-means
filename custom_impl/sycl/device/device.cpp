#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <unordered_set>
#include "./device.hpp"

// #if defined(DPCPP)
// // this definitions could change over time and moved to other place
// #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
// using namespace sycl::ext::oneapi::experimental::cuda;
// using namespace sycl::ext::oneapi::experimental;
// #endif

inline float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


Device::Device(std::vector<float>& h_attrs){
    _queue = _get_queue();

    attribute_bytes    = ATTRIBUTE_SIZE * DIMS * sizeof(float);
    mean_bytes         = K * DIMS * sizeof(float);
    count_bytes        = K * sizeof(unsigned int);

    attributes    = malloc_device<float>(attribute_bytes, _queue);
    mean          = malloc_device<float>(mean_bytes, _queue);
    counts        = malloc_device<unsigned int>(count_bytes, _queue);
    assigments    = malloc_device<unsigned int>(ATTRIBUTE_SIZE * sizeof(unsigned int), _queue);

    _queue.memcpy(attributes, h_attrs.data(), attribute_bytes);

    //shuffle attributes to random choose attributes
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, ATTRIBUTE_SIZE - 1);
    std::vector<float> h_mean;
    std::unordered_set<unsigned int> idxs;
    unsigned int idx{0};
    for(int i{0}; i < K; i++) {
        do { idx = indices(rng); } while(idxs.find(idx) != idxs.end());
        idxs.insert(idx);
        for(int d{0}; d < DIMS; d++) {
#if defined(NVIDIA_DEVICE)
            h_mean.push_back(h_attrs[d * ATTRIBUTE_SIZE + idx]);
#else
            h_mean.push_back(h_attrs[idx * DIMS + d]);
#endif
        }
    }

    _queue.memcpy(mean, h_mean.data(), mean_bytes);
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(assigments, 0, ATTRIBUTE_SIZE * sizeof(unsigned int));
    _sync();
}


Device::~Device() {
	if(attributes != nullptr)    free(attributes, _queue);
	if(mean != nullptr)          free(mean, _queue);
	if(counts != nullptr)        free(counts, _queue);
    if(assigments != nullptr)    free(assigments, _queue);
}


cl::sycl::queue Device::_get_queue() {
#if defined(INTEL_GPU_DEVICE)
	IntelGpuSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CudaGpuSelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	cl::sycl::queue queue{selector};
	std::cout << "Running on \"" << queue.get_device().get_info<cl::sycl::info::device::name>() << "\" under SYCL." << std::endl;
    return queue;
}


void Device::run_k_means() {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float t_assign{0}, t_reduction{0}, t_mean{0};
    for (size_t i{0}; i < ITERATIONS; i++) {
        start = std::chrono::high_resolution_clock::now();
#if defined(NVIDIA_DEVICE)
        _assign_clusters_nvidia();
#else
        _assign_clusters();
#endif
        _sync();
        end = std::chrono::high_resolution_clock::now();
        t_assign += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
#if defined(CPU_DEVICE)
        _cpu_reduction();
#elif defined(NVIDIA_DEVICE)
        _nvidia_reduction();
#else
        _intel_gpu_reduction();
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
#if defined(CPU_DEVICE)
    constexpr int group_size = ASSIGN_GROUP_SIZE_CPU;
#else //GPU
    constexpr int group_size = ASSIGN_GROUP_SIZE_IGPU;
#endif
    constexpr int B              = 2;
    constexpr int simd_width     = ASSIGN_SIMD_WIDTH; //check that simd_width < DIMS
    constexpr int simd_remainder = DIMS % simd_width;
    const int attrs_per_pckg     = ATTRIBUTE_SIZE / ASSIGN_PACK;
    const int remainder_pckg     = ATTRIBUTE_SIZE % ASSIGN_PACK;

    _queue.submit([&](handler& h) {
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;

        using global_ptr = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::global_space>;

        h.parallel_for<class assign_clusters>(nd_range(range(ASSIGN_PACK*group_size), range(group_size)), [=](nd_item<1> item){
            int offset       = attrs_per_pckg * item.get_group(0);
            int length       = (item.get_group(0) == ASSIGN_PACK-1) ? attrs_per_pckg + remainder_pckg : attrs_per_pckg;
            int attrs_per_wi = length / group_size;
            int remainder_wi = length % group_size;
            offset           = offset + attrs_per_wi * item.get_local_id(0);
            length           = (item.get_local_id(0) == group_size-1) ? attrs_per_wi + remainder_wi : attrs_per_wi;
            
            cl::sycl::vec<float, simd_width> v_attr, v_mean;
            cl::sycl::vec<float, simd_width> result[B];
            float best_distance{FLT_MAX};
            int best_cluster{0};
            float distance{0};

            for(int i = offset; i < offset + length; i++) {
                best_distance = FLT_MAX;
                best_cluster  = 0;

                for (int cluster = 0; cluster < K; cluster++) {
                    distance = 0;
                    //vector var initialization
                    for (size_t j = 0; j < B; j++)
                        result[j] = {0};

                    // calculate simd squared_l2_distance
                    for(int d{0}; d < DIMS - simd_remainder; d += simd_width*B) {
                        for(int j{0}; j < simd_width*B; j += simd_width) {
                            int res_idx = j / simd_width;
                            v_attr.load(0, global_ptr(&attrs[(i * DIMS) + d + j]));
                            v_mean.load(0, global_ptr(&mean[(cluster * DIMS) + d + j]));
                            v_attr          = v_attr - v_mean;
                            result[res_idx] = result[res_idx] + v_attr * v_attr;
                        }
                    }

                    for (size_t i = 1; i < B; i++)
                        result[0] += result[i];
                    
                    // reduce simd lane in scalar
                    for(int i{0}; i < simd_width; i++)
                        distance += result[0][i];

                    // calculate remaining values
                    for(int d{DIMS - simd_remainder}; d < DIMS; d++)
                        distance += squared_l2_distance(attrs[(i * DIMS) + d], mean[(cluster * DIMS) + d]);

                    if(distance < best_distance) {
                        best_distance = distance;
                        best_cluster  = cluster;
                    }
                }
                assigments[i] = best_cluster;
            }
        });
    });
}


void Device::_assign_clusters_nvidia() {
    constexpr int block_size = ASSIGN_BLOCK_SIZE_NVIDIA;
    const int blocks         = ATTRIBUTE_SIZE / block_size + (ATTRIBUTE_SIZE % block_size == 0 ? 0 : 1);
    _queue.submit([&](handler& h) {
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;

        h.parallel_for<class assign_clusters_nvidia>(nd_range(range(blocks*block_size), range(block_size)), [=](nd_item<1> item){
#if defined(HIPSYCL)
            __hipsycl_if_target_cuda(
#endif
            const int global_idx = item.get_global_id(0);

            if(global_idx < ATTRIBUTE_SIZE) {
                float best_distance{FLT_MAX}, distance;
                int best_cluster{0};

                for (int cluster = 0; cluster < K; cluster++) {
                    distance = 0.0f;
                    for(int d = 0; d < DIMS; d++)
                        distance += squared_l2_distance(__ldg(&attrs[d * ATTRIBUTE_SIZE + global_idx]), __ldg(&mean[cluster * DIMS + d]));

                    if(distance < best_distance) {
                        best_distance = distance;
                        best_cluster  = cluster;
                    }
                }

                assigments[global_idx] = best_cluster;
            }
#if defined(HIPSYCL)
            );
#endif
        });
    });
}


void Device::_nvidia_reduction() {
    const int remainder_attr = ATTRIBUTE_SIZE % RED_ATTRS_PACK_NVIDIA;
    const int quotient_attr  = ATTRIBUTE_SIZE / RED_ATTRS_PACK_NVIDIA;
    const int attr_pckg      = quotient_attr + (remainder_attr == 0 ? 0 : 1);
    const int remainder_dims = DIMS % RED_DIMS_PACK_NVIDIA;
    const int quotient_dims  = DIMS / RED_DIMS_PACK_NVIDIA;
    const int dims_pckg      = quotient_dims + (remainder_dims == 0 ? 0 : 1);
    cl::sycl::range<2> group_size(RED_DIMS_PACK_NVIDIA, RED_ATTRS_PACK_NVIDIA);
    cl::sycl::range<2> groups(dims_pckg, attr_pckg);

    // clean matrices
    _queue.memset(mean, 0, mean_bytes);
    _queue.memset(counts, 0, count_bytes);
    _sync();

    _queue.submit([&](handler& h) {
        float* attrs                = this->attributes;
        float* mean                 = this->mean;
        unsigned int* assigments    = this->assigments;
        unsigned int* counts        = this->counts;
        size_t size_mean            = RED_DIMS_PACK_NVIDIA * RED_ATTRS_PACK_NVIDIA * sizeof(float);
        size_t size_label           = RED_ATTRS_PACK_NVIDIA * sizeof(unsigned int);

        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> mean_package(size_mean, h);
        cl::sycl::accessor<unsigned int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> label_package(size_label, h);

        h.parallel_for<class gpu_nvidia_red>(nd_range<2>(groups*group_size, group_size), [=](nd_item<2> item){
#if defined(HIPSYCL)
            __hipsycl_if_target_cuda(
#endif
            int gid_x   = item.get_group(1);
            int baseRow = item.get_group(0) * RED_DIMS_PACK_NVIDIA; // Base row of the block
            int row     = baseRow + item.get_local_id(0); // Row of child thread
            int baseCol = gid_x * RED_ATTRS_PACK_NVIDIA; // Base column of the block
            int col     = baseCol + item.get_local_id(1); // Column of child thread
            int cltIdx  = item.get_local_id(0) * RED_ATTRS_PACK_NVIDIA + item.get_local_id(1); // 1D cluster index
            
            // Add one element per group from the remaining elements
            int offset = (gid_x < remainder_attr ? ((quotient_attr + 1) * gid_x) : (quotient_attr * gid_x + remainder_attr));
            int length = (gid_x < remainder_attr ? (quotient_attr + 1) : quotient_attr);

            // Load the values and cluster labels of instances into shared memory
            if (col < (offset + length) && row < DIMS) {
                mean_package[item.get_local_id(0) * RED_DIMS_PACK_NVIDIA + item.get_local_id(1)] = __ldg(&attrs[row * ATTRIBUTE_SIZE + col]);
                if (item.get_local_id(0) == 0)
                    label_package[item.get_local_id(1)] = __ldg(&assigments[col]);
            }
            item.barrier(cl::sycl::access::fence_space::local_space);

            // Compute partial evolution of centroid related to cluster number 'cltIdx'
            if (cltIdx < K) {  // Required condition: K <= RED_ATTRS_PACK_NVIDIA * RED_DIMS_PACK_NVIDIA <= 1024
                float sum[RED_DIMS_PACK_NVIDIA] = {0};
                unsigned int count = 0;

                // Accumulate contributions to cluster number 'cltIdx'
                // the second for condition is set for the last block to avoid out of bounds
                for (int x{0}; x < RED_ATTRS_PACK_NVIDIA && (baseCol + x) < (offset + length); x++) {
                    if (label_package[x] == cltIdx) {
                        count++;
                        for (int y{0}; y < RED_DIMS_PACK_NVIDIA && (baseRow + y) < DIMS; y++)
                            sum[y] += mean_package[y * RED_ATTRS_PACK_NVIDIA + x];
                    }
                }

                // Add block contribution to global mem
                if (count != 0) {
                    if (item.get_group(0) == 0) {
                        cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> count_ref(counts[cltIdx]);
                        count_ref.fetch_add(count);
                    }
                    int dmax = (item.get_group(0) == quotient_dims ? remainder_dims : RED_DIMS_PACK_NVIDIA);
                    for (int j{0}; j < dmax; j++) {  //number of dimensions managed by block
                        cl::sycl::atomic_ref<float, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> mean_ref(mean[cltIdx * DIMS + (baseRow + j)]);
                        mean_ref.fetch_add(sum[j]);
                    }
                }
            }
#if defined(HIPSYCL)
            );
#endif
        });
    });
}


void Device::_intel_gpu_reduction() {
    const int attrs_per_pckg = ATTRIBUTE_SIZE / RED_ATTRS_PACK;
    const int remainder_pckg = ATTRIBUTE_SIZE % RED_ATTRS_PACK;
    const int dims_remainder = DIMS % RED_DIMS_PACK_IGPU;
    const int dims_pckg      = DIMS / RED_DIMS_PACK_IGPU + (dims_remainder == 0 ? 0 : 1);
    const int att_group_size = RED_GROUP_SIZE_IGPU;

    cl::sycl::range<2> group_size(att_group_size, 1);
    cl::sycl::range<2> groups(RED_ATTRS_PACK, dims_pckg);
    
    constexpr int simd_width     = RED_SIMD_WIDTH; //check that simd_width <= RED_DIMS_PACK_IGPU
    constexpr int simd_remainder = DIMS % simd_width;

    // clean matrices
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(mean, 0, mean_bytes);
    _sync();

    _queue.submit([&](handler& h) {
        float* attrs                = this->attributes;
        float* mean                 = this->mean;
        unsigned int* assigments    = this->assigments;
        unsigned int* counts        = this->counts;
        size_t size_mean            = att_group_size * K * RED_DIMS_PACK_IGPU * sizeof(float);
        size_t size_count           = att_group_size * K * sizeof(float);

        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> mean_package(size_mean, h);
        cl::sycl::accessor<unsigned int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> count_package(size_count, h);
        using global_ptr = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::global_space>;
        using cte_local_ptr = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::local_space>;
        using local_ptr = cl::sycl::multi_ptr<float, cl::sycl::access::address_space::local_space>;

        h.parallel_for<class gpu_red>(nd_range<2>(group_size*groups, group_size), [=](nd_item<2> item){
            int local_idx    = item.get_local_id(0);
            int offset       = attrs_per_pckg * item.get_group(0);
            int length       = (item.get_group(0) == RED_ATTRS_PACK-1) ? attrs_per_pckg + remainder_pckg : attrs_per_pckg;
            int attrs_per_wi = length / att_group_size;
            int remainder_wi = length % att_group_size;
            offset           = offset + attrs_per_wi * local_idx;
            length           = (local_idx == att_group_size-1) ? attrs_per_wi + remainder_wi : attrs_per_wi;
            int dim_offset   = RED_DIMS_PACK_IGPU * item.get_group(1);
            int dim_length   = dim_offset + (item.get_group(1) != dims_pckg-1 ? RED_DIMS_PACK_IGPU : dims_remainder);
            cl::sycl::vec<float, simd_width> v_pckg, result;

            // init shared memory
            for (size_t cluster{0}; cluster < K; cluster++) {
                count_package[local_idx * K + cluster] = 0;
                
                for(size_t d = dim_offset; d < dim_length && d < DIMS; d++)
                    mean_package[local_idx * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d] = 0;
            }

            // perform work item private sum with explicit vectorization
            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                count_package[local_idx * K + cluster]++;
                
                int d = dim_offset;
                for(; (d < DIMS - simd_remainder) && (d <= dim_length - simd_width); d += simd_width) {
                    int pckg_id = local_idx * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d;
                    result.load(0, global_ptr(&attrs[i * DIMS + d]));
                    v_pckg.load(0, cte_local_ptr(&mean_package[pckg_id]));
                    result += v_pckg;
                    result.store(0, local_ptr(&mean_package[pckg_id]));
                }
                // calculate remaining DIMS
                for(; (d < DIMS) && (d < dim_length); d++)
                    mean_package[local_idx * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d] += attrs[i * DIMS + d];
            }

            // perform work group local sum with a tree reduction and explicit vectorization
            for(int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                item.barrier(cl::sycl::access::fence_space::local_space);
                if(local_idx < stride) {
                    for(int cluster{0}; cluster < K; cluster++) {
                        count_package[local_idx * K + cluster] += count_package[(local_idx + stride) * K + cluster];
                        
                        int d = dim_offset;
                        for(; (d < DIMS - simd_remainder) && (d <= dim_length - simd_width); d += simd_width) {
                            result.load(0, cte_local_ptr(&mean_package[local_idx * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d]));
                            v_pckg.load(0, cte_local_ptr(&mean_package[(local_idx + stride) * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d]));
                            result += v_pckg;
                            result.store(0, local_ptr(&mean_package[local_idx * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d]));
                        }
                        // calculate remaining DIMS
                        for(; (d < DIMS) && (d < dim_length); d++)
                            mean_package[local_idx * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d] += mean_package[(local_idx + stride) * K * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d];
                    }   
                }
            }

            // perform global sum
            if(local_idx == 0) {
                for(int cluster{0}; cluster < K; cluster++) {
                    cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> count_ref(counts[cluster]);
                    count_ref.fetch_add(count_package[cluster]);
                    for(int d{0}; d < DIMS; d++){
                        cl::sycl::atomic_ref<float, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> mean_ref(mean[cluster * DIMS + d]);
                        mean_ref.fetch_add(mean_package[cluster * RED_DIMS_PACK_IGPU + d]);
                    }
                }
            }
        });
    });
}


void Device::_cpu_reduction() {
    constexpr int simd_width     = RED_SIMD_WIDTH; //check that simd_width < DIMS
    constexpr int simd_remainder = DIMS % simd_width;
    const int attrs_per_CU       = ATTRIBUTE_SIZE / RED_ATTRS_PACK;
    const int remaining          = ATTRIBUTE_SIZE % RED_ATTRS_PACK;

    // clean matrices
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(mean, 0, mean_bytes);
    _sync();

    _queue.submit([&](handler &h) {
        unsigned int* count      = this->counts;
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;
        int p_size               = K * DIMS * sizeof(float);
        int c_size               = K * sizeof(unsigned int);
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> package(p_size, h);
        cl::sycl::accessor<unsigned int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> p_count(c_size, h);
        using global_ptr = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::global_space>;
        using cte_local_ptr  = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::local_space>;
        using local_ptr  = cl::sycl::multi_ptr<float, cl::sycl::access::address_space::local_space>;

        h.parallel_for<class cpu_red>(nd_range(range(RED_ATTRS_PACK), range(1)), [=](nd_item<1> item){
            const int global_idx = item.get_global_id(0);
            const int offset     = attrs_per_CU * global_idx;
            const int length     = (global_idx == RED_ATTRS_PACK-1) ? attrs_per_CU + remaining : attrs_per_CU;
            cl::sycl::vec<float, simd_width> v_pckg, result;

            for(int i{0}; i < K; i++) {
                p_count[i] = 0.0;
                for(int j{0}; j < DIMS; j++)
                    package[i*DIMS + j] = 0.0;
            }

            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                p_count[cluster]++;
                
                for(int d{0}; d < DIMS - simd_remainder; d += simd_width) {
                    result.load(0, global_ptr(&attrs[i * DIMS + d]));
                    v_pckg.load(0, cte_local_ptr(&package[cluster * DIMS + d]));
                    result += v_pckg;
                    result.store(0, local_ptr(&package[cluster * DIMS + d]));
                }
                // calculate remaining DIMS
                for(int d{DIMS - simd_remainder}; d < DIMS; d++)
                    package[cluster * DIMS + d] += attrs[i * DIMS + d];
            }

            for(int cluster{0}; cluster < K; cluster++) {
                cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> count_ref(count[cluster]);
                count_ref.fetch_add(p_count[cluster]);
                for(int d{0}; d < DIMS; d++) {
                    cl::sycl::atomic_ref<float, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> mean_ref(mean[cluster * DIMS + d]);
                    mean_ref.fetch_add(package[cluster * DIMS + d]);
                } 
            }
        });
    });
}


void Device::_compute_mean() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(K);
    _queue.submit([&](handler& h) {
        unsigned int* counts = this->counts;
        float* mean = this->mean;

        h.parallel_for<class compute_mean>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index   = item.get_global_id(0);
            const unsigned int count = counts[global_index] > 0 ? counts[global_index] : 1; 
            for(int d{0}; d < DIMS; d++)
                mean[global_index * DIMS + d] /= count;
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean) {
    _queue.memcpy(h_mean.data(), mean, mean_bytes);
    _sync();
}
