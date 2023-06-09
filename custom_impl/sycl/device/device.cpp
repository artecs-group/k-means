#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <unordered_set>
#include "./device.hpp"

inline float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


Device::Device(std::vector<float>& h_attrs){
    _queue = _get_queue();

    attribute_bytes    = ATTRIBUTE_SIZE * DIMS * sizeof(float);
    mean_bytes         = CLUSTERS * DIMS * sizeof(float);
    count_bytes        = CLUSTERS * sizeof(unsigned int);

    attributes    = malloc_device<float>(attribute_bytes, _queue);
    mean          = malloc_device<float>(mean_bytes, _queue);
#if defined(SYCL_IGPU)
    meanPrivate   = malloc_device<float>(mean_bytes * RED_GROUP_SIZE_IGPU * RED_ATTRS_PACK, _queue);
#endif
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
    for(int i{0}; i < CLUSTERS; i++) {
        do { idx = indices(rng); } while(idxs.find(idx) != idxs.end());
        idxs.insert(idx);
        for(int d{0}; d < DIMS; d++) {
#if defined(SYCL_NGPU) || defined(SYCL_PORTABLE)
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
#if defined(SYCL_IGPU)
    if(meanPrivate != nullptr)   free(meanPrivate, _queue);
#endif
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
#if defined(SYCL_NGPU)
        _assign_clusters_nvidia();
#elif defined(SYCL_PORTABLE)
        _assign_clusters_portable();
#else
        _assign_clusters_simd();
#endif
        _sync();
        end = std::chrono::high_resolution_clock::now();
        t_assign += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
#if defined(SYCL_CPU)
        _cpu_reduction();
#elif defined(SYCL_NGPU)
        _nvidia_reduction();
#elif defined(SYCL_IGPU)
        _intel_gpu_reduction();
#else
        _portable_reduction();
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
    const int max_group  = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
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


void Device::_assign_clusters_simd() {
#if defined(CPU_DEVICE)
    constexpr int group_size = ASSIGN_GROUP_SIZE_CPU;
#else //GPU
    const int group_size = ASSIGN_GROUP_SIZE_IGPU;
#endif
    constexpr int B              = 2;
    constexpr int simd_width     = ASSIGN_SIMD_WIDTH; //check that simd_width < DIMS
    constexpr int simd_remainder = DIMS % simd_width;
    const int attrs_per_pckg     = ATTRIBUTE_SIZE / ASSIGN_PACK;
    const int remainder_pckg     = ATTRIBUTE_SIZE % ASSIGN_PACK;

    _queue.submit([&](handler& h) {
        const float* attrs       = this->attributes;
        const float* mean        = this->mean;
        unsigned int* assigments = this->assigments;

        using global_ptr = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::global_space>;

        h.parallel_for<class assign_clusters_simd>(nd_range(range(ASSIGN_PACK*group_size), range(group_size)), [=](nd_item<1> item){
            int offset       = attrs_per_pckg * item.get_group(0);
            int length       = (item.get_group(0) == ASSIGN_PACK-1) ? attrs_per_pckg + remainder_pckg : attrs_per_pckg;
            const int attrs_per_wi = length / group_size;
            const int remainder_wi = length % group_size;
            offset           = offset + attrs_per_wi * item.get_local_id(0);
            length           = (item.get_local_id(0) == group_size-1) ? attrs_per_wi + remainder_wi : attrs_per_wi;
            
            cl::sycl::vec<float, simd_width> v_attr, v_mean;
            cl::sycl::vec<float, simd_width> simd_result[B];
            float best_distance{FLT_MAX};
            int best_cluster{0};
            float distance{0};

            for(int i = offset; i < offset + length; i++) {
                best_distance = FLT_MAX;
                best_cluster  = 0;

                for (int cluster = 0; cluster < CLUSTERS; cluster++) {
                    distance = 0;
                    //vector var initialization
                    for (size_t j = 0; j < B; j++)
                        simd_result[j] = {0};

                    // calculate simd squared_l2_distance
                    for(int d{0}; d < DIMS - simd_remainder; d += simd_width*B) {
                        for(int j{0}; j < simd_width*B; j += simd_width) {
                            int res_idx = j / simd_width;
                            v_attr.load(0, global_ptr(&attrs[(i * DIMS) + d + j]));
                            v_mean.load(0, global_ptr(&mean[(cluster * DIMS) + d + j]));
                            v_attr = v_attr - v_mean;
                            simd_result[res_idx] = simd_result[res_idx] + v_attr * v_attr;
                        }
                    }

                    for (size_t i = 1; i < B; i++)
                        simd_result[0] += simd_result[i];
                    
                    // reduce simd lane in scalar
                    for(int i{0}; i < simd_width; i++)
                        distance += simd_result[0][i];

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
#if defined(NVIDIA_DEVICE)
    constexpr int block_size = ASSIGN_BLOCK_SIZE_NVIDIA;
#else
    const int block_size = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
#endif
    const int blocks         = ATTRIBUTE_SIZE / block_size + (ATTRIBUTE_SIZE % block_size == 0 ? 0 : 1);
    _queue.submit([&](handler& h) {
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;

        h.parallel_for<class assign_clusters_nvidia>(nd_range(range(blocks*block_size), range(block_size)), [=](nd_item<1> item){
#if defined(HIPSYCL) && defined(NVIDIA_DEVICE)
            __hipsycl_if_target_cuda(
#endif
            const int global_idx = item.get_global_id(0);

            if(global_idx < ATTRIBUTE_SIZE) {
                float best_distance{FLT_MAX}, distance;
                int best_cluster{0};

                for (int cluster = 0; cluster < CLUSTERS; cluster++) {
                    distance = 0.0f;
                    for(int d = 0; d < DIMS; d++)
                        distance += squared_l2_distance(attrs[d * ATTRIBUTE_SIZE + global_idx], mean[cluster * DIMS + d]);

                    if(distance < best_distance) {
                        best_distance = distance;
                        best_cluster  = cluster;
                    }
                }

                assigments[global_idx] = best_cluster;
            }
#if defined(HIPSYCL) && defined(NVIDIA_DEVICE)
            );
#endif
        });
    });
}


void Device::_assign_clusters_portable() {
#if defined(CPU_DEVICE)
    constexpr int group_size = ASSIGN_GROUP_SIZE_CPU;
#elif defined(NVIDIA_DEVICE)
    constexpr int group_size = ASSIGN_BLOCK_SIZE_NVIDIA;
#else //intel GPU
    constexpr int group_size = ASSIGN_GROUP_SIZE_IGPU;
#endif

    constexpr int groups = ATTRIBUTE_SIZE / group_size + (ATTRIBUTE_SIZE % group_size == 0 ? 0 : 1);

    _queue.submit([&](handler& h) {
        float* attrs             = this->attributes;
        float* mean              = this->mean;
        unsigned int* assigments = this->assigments;

        h.parallel_for<class assign_clusters_portable>(nd_range(range(groups*group_size), range(group_size)), [=](nd_item<1> item){
            const int global_idx = item.get_global_id(0);

            if(global_idx < ATTRIBUTE_SIZE) {
                float best_distance{FLT_MAX}, distance;
                int best_cluster{0};

                for (int cluster = 0; cluster < CLUSTERS; cluster++) {
                    distance = 0.0f;
                    for(int d = 0; d < DIMS; d++)
                        distance += squared_l2_distance(attrs[d * ATTRIBUTE_SIZE + global_idx], mean[cluster * DIMS + d]);

                    if(distance < best_distance) {
                        best_distance = distance;
                        best_cluster  = cluster;
                    }
                }

                assigments[global_idx] = best_cluster;
            }
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

        cl::sycl::accessor<unsigned int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> label_package(size_label, h);
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> mean_package(size_mean, h);

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
                mean_package[item.get_local_id(0) * RED_DIMS_PACK_NVIDIA + item.get_local_id(1)] = attrs[row * ATTRIBUTE_SIZE + col];
                if (item.get_local_id(0) == 0)
                    label_package[item.get_local_id(1)] = assigments[col];
            }
            item.barrier(cl::sycl::access::fence_space::local_space);

            // Compute partial evolution of centroid related to cluster number 'cltIdx'
            if (cltIdx < CLUSTERS) {  // Required condition: CLUSTERS <= RED_ATTRS_PACK_NVIDIA * RED_DIMS_PACK_NVIDIA <= 1024
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
    constexpr int attrs_per_pckg = ATTRIBUTE_SIZE / RED_ATTRS_PACK;
    constexpr int remainder_pckg = ATTRIBUTE_SIZE % RED_ATTRS_PACK;
    constexpr int g_size = RED_GROUP_SIZE_IGPU;

    const cl::sycl::range<1> group_size(g_size);
    const cl::sycl::range<1> groups(RED_ATTRS_PACK);
    
    constexpr int simd_width     = RED_SIMD_WIDTH;
    constexpr int simd_remainder = DIMS % simd_width;

    // clean matrices
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(mean, 0, mean_bytes);
    _queue.memset(meanPrivate, 0, mean_bytes * g_size * RED_ATTRS_PACK);
    _sync();

    _queue.submit([&](handler& h) {
        const float* attrs             = this->attributes;
        float* mean                    = this->mean;
        float* meanPrivate             = this->meanPrivate;
        const unsigned int* assigments = this->assigments;
        unsigned int* counts           = this->counts;

        //sycl::local_accessor<unsigned int, 1> countLocal(CLUSTERS, h);
        cl::sycl::accessor<unsigned int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> countLocal(CLUSTERS * sizeof(unsigned int), h);
        using cte_global_ptr = sycl::multi_ptr<const float, sycl::access::address_space::global_space>;
        using global_ptr = sycl::multi_ptr<float, sycl::access::address_space::global_space>;

        h.parallel_for<class igpu_reduction>(nd_range<1>(group_size*groups, group_size), [=](nd_item<1> item){
            const int local_idx    = item.get_local_id(0);
            int offset             = attrs_per_pckg * item.get_group(0);
            int length             = (item.get_group(0) == RED_ATTRS_PACK-1) ? attrs_per_pckg + remainder_pckg : attrs_per_pckg;
            const int attrs_per_wi = length / item.get_local_range(0);
            const int remainder_wi = length % item.get_local_range(0);
            offset                 = offset + attrs_per_wi * local_idx;
            length                 = (local_idx == item.get_local_range(0)-1) ? attrs_per_wi + remainder_wi : attrs_per_wi;
            const int mean_group_range = item.get_group(0) * item.get_local_range(0) * CLUSTERS * DIMS;
            const int mean_idx     = mean_group_range + local_idx * CLUSTERS * DIMS;
            cl::sycl::vec<float, simd_width> simd_mean, simd_result;

            // init shared memory
            if(local_idx == 0) {
                for (size_t cluster{0}; cluster < CLUSTERS; cluster++)
                    countLocal[cluster] = 0;
            }

            // perform work item private sum with explicit vectorization
            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::work_group, cl::sycl::access::address_space::local_space> count_ref(countLocal[cluster]);
                count_ref.fetch_add(1);
                
                int d{0};
                for(; d < DIMS - simd_remainder; d += simd_width) {
                    int pckg_id = mean_idx + cluster * DIMS + d;
                    simd_result.load(0, cte_global_ptr(&attrs[i * DIMS + d]));
                    simd_mean.load(0, cte_global_ptr(&meanPrivate[pckg_id]));
                    simd_result += simd_mean;
                    simd_result.store(0, global_ptr(&meanPrivate[pckg_id]));
                }
                // calculate remaining DIMS
                for(; d < DIMS; d++) {
                    int pckg_id = mean_idx + cluster * DIMS + d;
                    meanPrivate[pckg_id] += attrs[i * DIMS + d];
                }
            }

            // perform work group local sum with a tree reduction and explicit vectorization
            for(int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                item.barrier(cl::sycl::access::fence_space::local_space);
                if(local_idx < stride) {
                    for(int cluster{0}; cluster < CLUSTERS; cluster++) {
                        int d{0};
                        for(; d < DIMS - simd_remainder; d += simd_width) {
                            int pckg_id1 = mean_idx + cluster * DIMS + d;
                            int pckg_id2 = mean_group_range + (local_idx + stride) * CLUSTERS * DIMS + cluster * DIMS + d;
                            simd_result.load(0, cte_global_ptr(&meanPrivate[pckg_id1]));
                            simd_mean.load(0, cte_global_ptr(&meanPrivate[pckg_id2]));
                            simd_result += simd_mean;
                            simd_result.store(0, global_ptr(&meanPrivate[pckg_id1]));
                        }
                        // calculate remaining DIMS
                        for(; d < DIMS; d++){
                            int pckg_id1 = mean_idx + cluster * DIMS + d;
                            int pckg_id2 = mean_group_range + (local_idx + stride) * CLUSTERS * DIMS + cluster * DIMS + d;
                            meanPrivate[pckg_id1] += meanPrivate[pckg_id2];
                        }
                    }   
                }
            }

            // perform global sum
            if(local_idx == 0) {
                for(int cluster{0}; cluster < CLUSTERS; cluster++) {
                    cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> count_ref(counts[cluster]);
                    count_ref.fetch_add(countLocal[cluster]);
                    for(int d{0}; d < DIMS; d++){
                        cl::sycl::atomic_ref<float, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> mean_ref(mean[cluster * DIMS + d]);
                        mean_ref.fetch_add(meanPrivate[mean_group_range + cluster * DIMS + d]);
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
        const float* attrs             = this->attributes;
        float* mean              = this->mean;
        const unsigned int* assigments = this->assigments;
        int p_size               = CLUSTERS * DIMS * sizeof(float);
        int c_size               = CLUSTERS * sizeof(unsigned int);

        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> package(p_size, h);
        cl::sycl::accessor<unsigned int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> p_count(c_size, h);

        using global_ptr = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::global_space>;
        using cte_private_ptr  = cl::sycl::multi_ptr<const float, cl::sycl::access::address_space::private_space>;
        using private_ptr  = cl::sycl::multi_ptr<float, cl::sycl::access::address_space::private_space>;

        h.parallel_for<class cpu_red>(nd_range(range(RED_ATTRS_PACK), range(1)), [=](nd_item<1> item){
            const int global_idx = item.get_global_id(0);
            const int offset     = attrs_per_CU * global_idx;
            const int length     = (global_idx == RED_ATTRS_PACK-1) ? attrs_per_CU + remaining : attrs_per_CU;
            cl::sycl::vec<float, simd_width> simd_mean, simd_result;

            for(int i{0}; i < CLUSTERS; i++) {
                p_count[i] = 0.0;
                for(int j{0}; j < DIMS; j++)
                    package[i*DIMS + j] = 0.0;
            }

            for (int i{offset}; i < offset + length; i++) {
                int cluster = assigments[i];
                p_count[cluster]++;
                
                for(int d{0}; d < DIMS - simd_remainder; d += simd_width) {
                    simd_result.load(0, global_ptr(&attrs[i * DIMS + d]));
                    simd_mean.load(0, cte_private_ptr(&package[cluster * DIMS + d]));
                    simd_result += simd_mean;
                    simd_result.store(0, private_ptr(&package[cluster * DIMS + d]));
                }
                // calculate remaining DIMS
                for(int d{DIMS - simd_remainder}; d < DIMS; d++)
                    package[cluster * DIMS + d] += attrs[i * DIMS + d];
            }

            for(int cluster{0}; cluster < CLUSTERS; cluster++) {
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


void Device::_portable_reduction() {
#if defined(INTEL_GPU_DEVICE)
    constexpr int ATTRS_PACK = RED_ATTRS_PACK_IGPU;
    constexpr int DIMS_PACK = RED_DIMS_PACK_IGPU;
#elif defined(NVIDIA_DEVICE)
    constexpr int ATTRS_PACK = RED_ATTRS_PACK_NVIDIA;
    constexpr int DIMS_PACK = RED_DIMS_PACK_NVIDIA;
#elif defined(CPU_DEVICE)
    constexpr int ATTRS_PACK = RED_ATTRS_PACK_CPU;
    constexpr int DIMS_PACK = RED_DIMS_PACK_CPU;
#endif
    constexpr int remainder_attr = ATTRIBUTE_SIZE % ATTRS_PACK;
    constexpr int quotient_attr  = ATTRIBUTE_SIZE / ATTRS_PACK;
    constexpr int attr_pckg      = quotient_attr + (remainder_attr == 0 ? 0 : 1);
    constexpr int remainder_dims = DIMS % DIMS_PACK;
    constexpr int quotient_dims  = DIMS / DIMS_PACK;
    constexpr int dims_pckg      = quotient_dims + (remainder_dims == 0 ? 0 : 1);
    cl::sycl::range<2> group_size(DIMS_PACK, ATTRS_PACK);
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
        size_t size_mean            = DIMS_PACK * ATTRS_PACK * sizeof(float);
        size_t size_label           = ATTRS_PACK * sizeof(unsigned int);

        cl::sycl::accessor<unsigned int, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> label_package(size_label, h);
        cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> mean_package(size_mean, h);

        h.parallel_for<class portable_reduction>(nd_range<2>(groups*group_size, group_size), [=](nd_item<2> item){
            int gid_x   = item.get_group(1);
            int baseRow = item.get_group(0) * DIMS_PACK;
            int row     = baseRow + item.get_local_id(0);
            int baseCol = gid_x * ATTRS_PACK;
            int col     = baseCol + item.get_local_id(1);
            int cltIdx  = item.get_local_id(0) * ATTRS_PACK + item.get_local_id(1);
            
            int offset = (gid_x < remainder_attr ? ((quotient_attr + 1) * gid_x) : (quotient_attr * gid_x + remainder_attr));
            int length = (gid_x < remainder_attr ? (quotient_attr + 1) : quotient_attr);

            if (col < (offset + length) && row < DIMS) {
                mean_package[item.get_local_id(0) * DIMS_PACK + item.get_local_id(1)] = attrs[row * ATTRIBUTE_SIZE + col];
                if (item.get_local_id(0) == 0)
                    label_package[item.get_local_id(1)] = assigments[col];
            }
            item.barrier(cl::sycl::access::fence_space::local_space);

            if (cltIdx < CLUSTERS) {
                float sum[DIMS_PACK] = {0};
                unsigned int count = 0;

                for (int x{0}; x < ATTRS_PACK && (baseCol + x) < (offset + length); x++) {
                    if (label_package[x] == cltIdx) {
                        count++;
                        for (int y{0}; y < DIMS_PACK && (baseRow + y) < DIMS; y++)
                            sum[y] += mean_package[y * ATTRS_PACK + x];
                    }
                }

                if (count != 0) {
                    if (item.get_group(0) == 0) {
                        cl::sycl::atomic_ref<unsigned int, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> count_ref(counts[cltIdx]);
                        count_ref.fetch_add(count);
                    }
                    int dmax = (item.get_group(0) == quotient_dims ? remainder_dims : DIMS_PACK);
                    for (int j{0}; j < dmax; j++) {
                        cl::sycl::atomic_ref<float, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space> mean_ref(mean[cltIdx * DIMS + (baseRow + j)]);
                        mean_ref.fetch_add(sum[j]);
                    }
                }
            }
        });
    });
}


void Device::_compute_mean() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(CLUSTERS);
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
