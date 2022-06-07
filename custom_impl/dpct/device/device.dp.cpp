#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <unordered_set>
#include "./device.dp.hpp"

inline float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


Device::Device(int _k, int _dims, int length, std::vector<float> &h_attrs)
    : k(_k), dims(_dims) {
    _queue = _select_device();
    
    attributes_size     = length;
    attributes_bytes    = attributes_size * dims * sizeof(float);
    mean_bytes          = k * dims * sizeof(float);
    count_bytes         = k * sizeof(unsigned int);

    attributes = (float *)sycl::malloc_device(attributes_bytes, _queue);
    mean = (float *)sycl::malloc_device(mean_bytes, _queue);
    counts = (unsigned int *)sycl::malloc_device(count_bytes, _queue);
    assigments = sycl::malloc_device<unsigned int>(attributes_size, _queue);

    _queue.memcpy(attributes, h_attrs.data(), attributes_bytes);

    //shuffle attributess to random choose attributess
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, attributes_size - 1);
    std::vector<float> h_mean;
    std::unordered_set<unsigned int> idxs;
    unsigned int idx{0};
    for(int i{0}; i < k; i++) {
        do { idx = indices(rng); } while(idxs.find(idx) != idxs.end());
        idxs.insert(idx);
        for(int d{0}; d < dims; d++)
            h_mean.push_back(h_attrs[d * attributes_size + idx]);
    }

    _queue.memcpy(mean, h_mean.data(), mean_bytes);
    _queue.memset(counts, 0, count_bytes);
    _queue.memset(assigments, 0, attributes_size * sizeof(unsigned int));
    _sync();
}


Device::~Device() {
    if (attributes != nullptr) sycl::free(attributes, _queue);
    if (mean != nullptr) sycl::free(mean, _queue);
    if (counts != nullptr) sycl::free(counts, _queue);
    if (assigments != nullptr) sycl::free(assigments, _queue);
}


sycl::queue Device::_select_device() {
	CudaGpuSelector selector{};

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
        _reduction();
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
            * threads = elements
            * work_items = elements
            * blocks     = 1
    
    Case 2) elements > max_group_size
            * threads = max_group_size
            * work_items = elements + threads - (elements % threads)
            * blocks     = work_items / threads

*/
std::tuple<int,int,int> Device::_get_block_threads(int elements) {
    const int max_group = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    const int threads = (elements <= max_group) ? elements : max_group;
    const int work_items = (elements <= max_group) ? elements : elements + threads - (elements % threads);
    const int blocks     = (elements <= max_group) ? 1 : work_items / threads;

    return std::tuple<int,int,int>(blocks, threads, work_items);
}


void Device::_sync() {
    _queue.wait();
}


void Device::_assign_clusters() {
    constexpr int threads = ASSIGN_BLOCK_SIZE_NVIDIA;
    const int blocks      = attributes_size / threads + (attributes_size % threads == 0 ? 0 : 1);

    _queue.submit([&](sycl::handler &cgh) {
        auto attrs_size = this->attributes_size;
        auto k = this->k;
        auto dims = this->dims;
        auto attrs = this->attributes;
        auto mean = this->mean;
        auto assigments = this->assigments;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                               sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {

            const int global_idx =
                item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);
            float best_distance{FLT_MAX};
            int best_cluster{0};
            float distance{0};

            if(global_idx >= attrs_size)
                return;

            for (int cluster = 0; cluster < k; cluster++) {
                for(int d{0}; d < dims; d++)
                    distance += squared_l2_distance(attrs[d * attrs_size + global_idx], mean[cluster * dims + d]);

                bool min = distance < best_distance;
                best_distance = min ? distance : best_distance;
                best_cluster  = distance < best_distance ? cluster : best_cluster;
                distance      = 0;
            }
            assigments[global_idx] = best_cluster;
        });
    });
}

void Device::_reduction() {
    const int remainder_attr = attributes_size % RED_ATTRS_PACK_NVIDIA;
    const int quotient_attr  = attributes_size / RED_ATTRS_PACK_NVIDIA;
    const int attr_pckg      = quotient_attr + (remainder_attr == 0 ? 0 : 1);
    const int remainder_dims = dims % RED_DIMS_PACK_NVIDIA;
    const int quotient_dims  = dims / RED_DIMS_PACK_NVIDIA;
    const int dims_pckg      = quotient_dims + (remainder_dims == 0 ? 0 : 1);

    sycl::range<3> threads(1, 1, 1), blocks(1, 1, 1);
    threads[2] = RED_DIMS_PACK_NVIDIA;
    threads[1] = RED_ATTRS_PACK_NVIDIA;
    blocks[2] = dims_pckg;
    blocks[1] = attr_pckg;

    size_t size_mean  = RED_DIMS_PACK_NVIDIA * RED_ATTRS_PACK_NVIDIA * sizeof(float);
    size_t size_label = RED_ATTRS_PACK_NVIDIA * sizeof(unsigned int);

    _queue.memset(mean, 0, mean_bytes).wait();
    _queue.memset(counts, 0, count_bytes).wait();
    _sync();

    _queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<float, 1, sycl::access_mode::read_write, sycl::access::target::local>
            mean_package(size_mean, cgh);

        sycl::accessor<unsigned int, 1, sycl::access_mode::read_write, sycl::access::target::local>
            label_package(size_label, cgh);

        auto attrs_size = this->attributes_size;
        auto k = this->k;
        auto dims = this->dims;
        auto attrs = this->attributes;
        auto mean = this->mean;
        auto assigments = this->assigments;
        auto counts = this->counts;

        cgh.parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {

            int gid_x = item_ct1.get_group(1);
            int baseRow =
                item_ct1.get_group(2) * RED_DIMS_PACK_NVIDIA; // Base row of the block
            int row = baseRow + item_ct1.get_local_id(2); // Row of child thread
            int baseCol = gid_x * RED_ATTRS_PACK_NVIDIA; // Base column of the block
            int col = baseCol + item_ct1.get_local_id(1); // Column of child thread
            int cltIdx = item_ct1.get_local_id(2) * RED_ATTRS_PACK_NVIDIA +
                        item_ct1.get_local_id(1); // 1D cluster index

            // Add one element per group from the remaining elements
            int offset = (gid_x < remainder_attr ? ((quotient_attr + 1) * gid_x) : (quotient_attr * gid_x + remainder_attr));
            int length = (gid_x < remainder_attr ? (quotient_attr + 1) : quotient_attr);

            // Load the values and cluster labels of instances into shared memory
            if (col < (offset + length) && row < dims) {
                mean_package[item_ct1.get_local_id(2) * RED_DIMS_PACK_NVIDIA +
                            item_ct1.get_local_id(1)] = attrs[row * attrs_size + col];
                if (item_ct1.get_local_id(2) == 0)
                    label_package[item_ct1.get_local_id(1)] = assigments[col];
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);

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
                    if (item_ct1.get_group(2) == 0)
                        sycl::atomic<unsigned int>(
                            sycl::global_ptr<unsigned int>(&counts[cltIdx]))
                            .fetch_add(count);
                    int dmax =
                        (item_ct1.get_group(2) == quotient_dims ? remainder_dims
                                                                : RED_DIMS_PACK_NVIDIA);
                    for (int j{0}; j < dmax; j++)  //number of dimensions managed by block
                        dpct::atomic_fetch_add(&mean[cltIdx * dims + (baseRow + j)],
                                            sum[j]);
                }
            }

        });
    });
}


void Device::_compute_mean() {
    std::tie (blocks, threads, work_items) = _get_block_threads(k);

    _queue.submit([&](sycl::handler &cgh) {
        auto dims = this->dims;
        auto mean = this->mean;
        auto counts = this->counts;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blocks) *
                                               sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
            const int global_index =
                item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                item_ct1.get_local_id(2);
            const int count        = (1 < counts[global_index]) ? counts[global_index] : 1;
            for(int d{0}; d < dims; d++)
                mean[global_index * dims + d] /= count;
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean) {
    _queue.memcpy(h_mean.data(), mean, mean_bytes).wait();
    _sync();
}
