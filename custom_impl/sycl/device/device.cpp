#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <unordered_set>
#include <dpct/dpct.hpp>
#include "./device.hpp"


inline float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


Device::Device(int _k, int _dims, int length, std::vector<float>& h_attrs): k(_k), dims(_dims){
    _queue = _get_queue();

    attribute_size     = length;
    attribute_bytes    = attribute_size * dims * sizeof(float);
    mean_bytes         = k * dims * sizeof(float);
    count_bytes        = k * sizeof(unsigned int);

    attributes    = malloc_device<float>(attribute_bytes, _queue);
    mean          = malloc_device<float>(mean_bytes, _queue);
    counts        = malloc_device<unsigned int>(count_bytes, _queue);
    assigments    = malloc_device<unsigned int>(attribute_size * sizeof(unsigned int), _queue);

    _queue.memcpy(attributes, h_attrs.data(), attribute_bytes);

    //shuffle attributes to random choose attributes
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, attribute_size - 1);
    std::vector<float> h_mean;
    std::unordered_set<unsigned int> idxs;
    unsigned int idx{0};
    for(int i{0}; i < k; i++) {
        do { idx = indices(rng); } while(idxs.find(idx) != idxs.end());
        idxs.insert(idx);
        for(int d{0}; d < dims; d++) {
#if defined(NVIDIA_DEVICE)
            h_mean.push_back(h_attrs[d * attribute_size + idx]);
#else
            h_mean.push_back(h_attrs[idx * dims + d]);
#endif
        }
    }

    _queue.memcpy(mean, h_mean.data(), mean_bytes);
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(assigments, 0, attribute_size * sizeof(unsigned int));
    _sync();
}


Device::~Device() {
	if(attributes != nullptr)    free(attributes, _queue);
	if(mean != nullptr)          free(mean, _queue);
	if(counts != nullptr)        free(counts, _queue);
    if(assigments != nullptr)    free(assigments, _queue);
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
    for (size_t i{0}; i < iterations; i++) {
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
    constexpr int B          = 2;
    const int simd_width     = ASSIGN_SIMD_WIDTH; //check that simd_width < dims
    const int simd_remainder = dims % simd_width;
    const int attrs_per_pckg = attribute_size / ASSIGN_PACK;
    const int remainder_pckg = attribute_size % ASSIGN_PACK;

    _queue.submit([&](handler& h) {
        int k                    = this->k;
        int dims                 = this->dims;
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;

        using global_ptr = sycl::multi_ptr<float, sycl::access::address_space::global_space>;

        h.parallel_for<class assign_clusters>(nd_range(range(ASSIGN_PACK*group_size), range(group_size)), [=](nd_item<1> item){
            int offset       = attrs_per_pckg * item.get_group(0);
            int length       = (item.get_group(0) == ASSIGN_PACK-1) ? attrs_per_pckg + remainder_pckg : attrs_per_pckg;
            int attrs_per_wi = length / group_size;
            int remainder_wi = length % group_size;
            offset           = offset + attrs_per_wi * item.get_local_id(0);
            length           = (item.get_local_id(0) == group_size-1) ? attrs_per_wi + remainder_wi : attrs_per_wi;
            
            sycl::vec<float, simd_width> v_attr, v_mean;
            sycl::vec<float, simd_width> result[B];
            float best_distance{FLT_MAX};
            int best_cluster{0};
            float distance{0};

            for(int i = offset; i < offset + length; i++) {
                best_distance = FLT_MAX;
                best_cluster  = 0;

                for (int cluster = 0; cluster < k; cluster++) {
                    distance = 0;
                    //vector var initialization
                    for (size_t j = 0; j < B; j++)
                        result[j] = {0};

                    // calculate simd squared_l2_distance
                    for(int d{0}; d < dims - simd_remainder; d += simd_width*B) {
                        for(int j{0}; j < simd_width*B; j += simd_width) {
                            int res_idx = j / simd_width;
                            v_attr.load(0, global_ptr(&attrs[(i * dims) + d + j]));
                            v_mean.load(0, global_ptr(&mean[(cluster * dims) + d + j]));
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
                    for(int d{dims - simd_remainder}; d < dims; d++)
                        distance += squared_l2_distance(attrs[(i * dims) + d], mean[(cluster * dims) + d]);

                    bool update   = distance < best_distance;
                    best_distance = update ? distance : best_distance;
                    best_cluster  = update ? cluster : best_cluster;
                }
                assigments[i] = best_cluster;
            }
        });
    });
}


void Device::_assign_clusters_nvidia() {
    constexpr int block_size = ASSIGN_BLOCK_SIZE_NVIDIA;
    const int blocks         = attribute_size / block_size + (attribute_size % block_size == 0 ? 0 : 1);
    _queue.submit([&](handler& h) {
        int attribute_size       = this->attribute_size;
        int k                    = this->k;
        int dims                 = this->dims;
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;

        h.parallel_for<class assign_clusters_nvidia>(nd_range(range(blocks*block_size), range(block_size)), [=](nd_item<1> item){
            int global_idx = item.get_global_id(0);
            float best_distance{FLT_MAX};
            int best_cluster{0};
            float distance{0};

            if(global_idx >= attribute_size)
                return;

            for (int cluster = 0; cluster < k; cluster++) {
                for(int d{0}; d < dims; d++)
                    distance += squared_l2_distance(attrs[d * attribute_size + global_idx], mean[cluster * dims + d]);

                bool min = distance < best_distance;
                best_distance = min ? distance : best_distance;
                best_cluster  = distance < best_distance ? cluster : best_cluster;
                distance      = 0;
            }
            assigments[global_idx] = best_cluster;
        });
    });
}


void Device::_nvidia_reduction() {
    const int remainder_attr = attribute_size % RED_ATTRS_PACK_NVIDIA;
    const int quotient_attr  = attribute_size / RED_ATTRS_PACK_NVIDIA;
    const int attr_pckg      = quotient_attr + (remainder_attr == 0 ? 0 : 1);
    const int remainder_dims = dims % RED_DIMS_PACK_NVIDIA;
    const int quotient_dims  = dims / RED_DIMS_PACK_NVIDIA;
    const int dims_pckg      = quotient_dims + (remainder_dims == 0 ? 0 : 1);
    sycl::range<2> group_size(RED_DIMS_PACK_NVIDIA, RED_ATTRS_PACK_NVIDIA);
    sycl::range<2> groups(dims_pckg, attr_pckg);

    // clean matrices
    _queue.memset(mean, 0, mean_bytes);
    _queue.memset(counts, 0, count_bytes);
    _sync();

    _queue.submit([&](handler& h) {
        int attribute_size          = this->attribute_size;
        int k                       = this->k;
        int dims                    = this->dims;
        float* attrs                = this->attributes;
        float* mean                 = this->mean;
        unsigned int* assigments    = this->assigments;
        unsigned int* counts        = this->counts;
        size_t size_mean            = RED_DIMS_PACK_NVIDIA * RED_ATTRS_PACK_NVIDIA * sizeof(float);
        size_t size_label           = RED_ATTRS_PACK_NVIDIA * sizeof(unsigned int);

        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> mean_package(size_mean, h);
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::local> label_package(size_label, h);

        h.parallel_for<class gpu_nvidia_red>(nd_range<2>(groups*group_size, group_size), [=](nd_item<2> item){
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
            if (col < (offset + length) && row < dims) {
                mean_package[item.get_local_id(0) * RED_DIMS_PACK_NVIDIA + item.get_local_id(1)] = attrs[row * attribute_size + col];
                if (item.get_local_id(0) == 0)
                    label_package[item.get_local_id(1)] = assigments[col];
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Compute partial evolution of centroid related to cluster number 'cltIdx'
            if (cltIdx < k) {  // Required condition: k <= RED_ATTRS_PACK_NVIDIA * RED_DIMS_PACK_NVIDIA <= 1024
                float sum[RED_DIMS_PACK_NVIDIA] = {0};
                unsigned int count = 0;

                // Accumulate contributions to cluster number 'cltIdx'
                // the second for condition is set for the last block to avoid out of bounds
                for (int x{0}; x < RED_ATTRS_PACK_NVIDIA && (baseCol + x) < (offset + length); x++) {
                    if (label_package[x] == cltIdx) {
                        count++;
                        for (int y{0}; y < RED_DIMS_PACK_NVIDIA && (baseRow + y) < dims; y++)
                            sum[y] += mean_package[y * RED_ATTRS_PACK_NVIDIA + x];
                    }
                }

                // Add block contribution to global mem
                if (count != 0) {
                    if (item.get_group(0) == 0)
                        sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&counts[cltIdx])).fetch_add(count);
                    int dmax = (item.get_group(0) == quotient_dims ? remainder_dims : RED_DIMS_PACK_NVIDIA);
                    for (int j{0}; j < dmax; j++)  //number of dimensions managed by block
                        dpct::atomic_fetch_add(&mean[cltIdx * dims + (baseRow + j)], sum[j]);
                }
            }
        });
    });
}


void Device::_intel_gpu_reduction() {
    const int attrs_per_pckg = attribute_size / RED_ATTRS_PACK;
    const int remainder_pckg = attribute_size % RED_ATTRS_PACK;
    const int dims_remainder = dims % RED_DIMS_PACK_IGPU;
    const int dims_pckg      = dims / RED_DIMS_PACK_IGPU + (dims_remainder == 0 ? 0 : 1);
    const int att_group_size = RED_GROUP_SIZE_IGPU;

    sycl::range<2> group_size(att_group_size, 1);
    sycl::range<2> groups(RED_ATTRS_PACK, dims_pckg);
    
    const int simd_width     = RED_SIMD_WIDTH; //check that simd_width <= RED_DIMS_PACK_IGPU
    const int simd_remainder = dims % simd_width;

    // clean matrices
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(mean, 0, mean_bytes);
    _sync();

    _queue.submit([&](handler& h) {
        int k                       = this->k;
        int dims                    = this->dims;
        float* attrs                = this->attributes;
        float* mean                 = this->mean;
        unsigned int* assigments    = this->assigments;
        unsigned int* counts        = this->counts;
        size_t size_mean            = att_group_size * k * RED_DIMS_PACK_IGPU * sizeof(float);
        size_t size_count           = att_group_size * k * sizeof(float);

        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> mean_package(size_mean, h);
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::local> count_package(size_count, h);
        using global_ptr = sycl::multi_ptr<float, sycl::access::address_space::global_space>;
        using local_ptr = sycl::multi_ptr<float, sycl::access::address_space::local_space>;

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
            sycl::vec<float, simd_width> v_pckg, result;

            // init shared memory
            for (size_t cluster{0}; cluster < k; cluster++) {
                count_package[local_idx * k + cluster] = 0;
                
                for(size_t d = dim_offset; d < dim_length && d < dims; d++)
                    mean_package[local_idx * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d] = 0;
            }

            // perform work item private sum with explicit vectorization
            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                count_package[local_idx * k + cluster]++;
                
                int d = dim_offset;
                for(; (d < dims - simd_remainder) && (d <= dim_length - simd_width); d += simd_width) {
                    int pckg_id = local_idx * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d;
                    result.load(0, global_ptr(&attrs[i * dims + d]));
                    v_pckg.load(0, local_ptr(&mean_package[pckg_id]));
                    result += v_pckg;
                    result.store(0, local_ptr(&mean_package[pckg_id]));
                }
                // calculate remaining dims
                for(; (d < dims) && (d < dim_length); d++)
                    mean_package[local_idx * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d] += attrs[i * dims + d];
            }

            // perform work group local sum with a tree reduction and explicit vectorization
            for(int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                item.barrier(sycl::access::fence_space::local_space);
                if(local_idx < stride) {
                    for(int cluster{0}; cluster < k; cluster++) {
                        count_package[local_idx * k + cluster] += count_package[(local_idx + stride) * k + cluster];
                        
                        int d = dim_offset;
                        for(; (d < dims - simd_remainder) && (d <= dim_length - simd_width); d += simd_width) {
                            result.load(0, local_ptr(&mean_package[local_idx * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d]));
                            v_pckg.load(0, local_ptr(&mean_package[(local_idx + stride) * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d]));
                            result += v_pckg;
                            result.store(0, local_ptr(&mean_package[local_idx * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d]));
                        }
                        // calculate remaining dims
                        for(; (d < dims) && (d < dim_length); d++)
                            mean_package[local_idx * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d] += mean_package[(local_idx + stride) * k * RED_DIMS_PACK_IGPU + cluster * RED_DIMS_PACK_IGPU + d];
                    }   
                }
            }

            // perform global sum
            if(local_idx == 0) {
                for(int cluster{0}; cluster < k; cluster++) {
                    sycl::atomic<unsigned int>(sycl::global_ptr<unsigned int>(&counts[cluster])).fetch_add(count_package[cluster]);
                    for(int d{0}; d < dims; d++)
                        dpct::atomic_fetch_add(&mean[cluster * dims + d], mean_package[cluster * RED_DIMS_PACK_IGPU + d]); 
                }
            }
        });
    });
}


void Device::_cpu_reduction() {
    const int simd_width    = RED_SIMD_WIDTH; //check that simd_width < dims
    const int simd_remainder = dims % simd_width;
    const int attrs_per_CU  = attribute_size / RED_ATTRS_PACK;
    const int remaining     = attribute_size % RED_ATTRS_PACK;

    // clean matrices
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(mean, 0, mean_bytes);
    _sync();

    _queue.submit([&](handler &h) {
        int k                    = this->k;
        int dims                 = this->dims;
        unsigned int* count      = this->counts;
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;
        int p_size               = k * dims * sizeof(float);
        int c_size               = k * sizeof(unsigned int);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> package(p_size, h);
        sycl::accessor<unsigned int, 1, sycl::access::mode::read_write, sycl::access::target::local> p_count(c_size, h);
        using global_ptr = sycl::multi_ptr<float, sycl::access::address_space::global_space>;
        using local_ptr  = sycl::multi_ptr<float, sycl::access::address_space::local_space>;

        h.parallel_for<class cpu_red>(nd_range(range(RED_ATTRS_PACK), range(1)), [=](nd_item<1> item){
            const int global_idx = item.get_global_id(0);
            const int offset     = attrs_per_CU * global_idx;
            const int length     = (global_idx == RED_ATTRS_PACK-1) ? attrs_per_CU + remaining : attrs_per_CU;
            sycl::vec<float, simd_width> v_pckg, result;

            for(int i{0}; i < k; i++) {
                p_count[i] = 0.0;
                for(int j{0}; j < dims; j++)
                    package[i*dims + j] = 0.0;
            }

            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                p_count[cluster]++;
                
                for(int d{0}; d < dims - simd_remainder; d += simd_width) {
                    result.load(0, global_ptr(&attrs[i * dims + d]));
                    v_pckg.load(0, local_ptr(&package[cluster * dims + d]));
                    result += v_pckg;
                    result.store(0, local_ptr(&package[cluster * dims + d]));
                }
                // calculate remaining dims
                for(int d{dims - simd_remainder}; d < dims; d++)
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
            const int count        = sycl::max<unsigned int>(counts[global_index], 1); 
            for(int d{0}; d < dims; d++)
                mean[global_index * dims + d] /= count;
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean) {
    _queue.memcpy(h_mean.data(), mean, mean_bytes);
    _sync();
}
