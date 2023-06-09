#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <unordered_set>
#include "./device.cuh"

__device__ inline float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


__global__ void assign_clusters(const float* __restrict__ attrs, const float* __restrict__ mean, 
    unsigned int* __restrict__ assigments)
{ 
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(global_idx < ATTRIBUTE_SIZE) {
        float best_distance{FLT_MAX}, distance;
        int best_cluster{0};

        for (int cluster = 0; cluster < K; cluster++) {
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
}


__global__ void reduction(size_t sh_offset, int remainder_attr, int quotient_attr, int remainder_dims,
    int quotient_dims, const float* __restrict__ attrs, float* __restrict__ mean, 
    const unsigned int* __restrict__ assigments, unsigned int* __restrict__ counts)
{
    extern __shared__ float mean_package[];
    unsigned int* label_package = (unsigned int*) &mean_package[sh_offset];

    int gid_x   = blockIdx.y;
    int baseRow = blockIdx.x * RED_DIMS_PACK_NVIDIA; // Base row of the block
    int row     = baseRow + threadIdx.x; // Row of child thread
    int baseCol = gid_x * RED_ATTRS_PACK_NVIDIA; // Base column of the block
    int col     = baseCol + threadIdx.y; // Column of child thread
    int cltIdx  = threadIdx.x * RED_ATTRS_PACK_NVIDIA + threadIdx.y; // 1D cluster index

    // Add one element per group from the remaining elements
    int offset = (gid_x < remainder_attr ? ((quotient_attr + 1) * gid_x) : (quotient_attr * gid_x + remainder_attr));
    int length = (gid_x < remainder_attr ? (quotient_attr + 1) : quotient_attr);

    // Load the values and cluster labels of instances into shared memory
    if (col < (offset + length) && row < DIMS) {
        mean_package[threadIdx.x * RED_DIMS_PACK_NVIDIA + threadIdx.y] = attrs[row * ATTRIBUTE_SIZE + col];
        if (threadIdx.x == 0)
            label_package[threadIdx.y] = assigments[col];
    }
    __syncthreads();

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
            if (blockIdx.x == 0)
                atomicAdd(&counts[cltIdx], count);
            int dmax = (blockIdx.x == quotient_dims ? remainder_dims : RED_DIMS_PACK_NVIDIA);
            for (int j{0}; j < dmax; j++)  //number of dimensions managed by block
                atomicAdd(&mean[cltIdx * DIMS + (baseRow + j)], sum[j]);
        }
    }
}


__global__ void compute_mean(float* __restrict__ mean, const unsigned int* __restrict__ counts)
{
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = counts[global_index];
    
    if (count < 1)
        count = 1;

    for(int d{0}; d < DIMS; d++)
        mean[global_index * DIMS + d] /= count;
}


Device::Device(std::vector<float>& h_attrs) {
    _select_device();
    
    attributes_bytes    = ATTRIBUTE_SIZE * DIMS * sizeof(float);
    mean_bytes          = K * DIMS * sizeof(float);
    count_bytes         = K * sizeof(unsigned int);

    cudaMalloc(&attributes, attributes_bytes);
    cudaMalloc(&mean, mean_bytes);
    cudaMalloc(&counts, count_bytes);
    cudaMalloc(&assigments, ATTRIBUTE_SIZE * sizeof(unsigned int));

    cudaMemcpy(attributes, h_attrs.data(), attributes_bytes, cudaMemcpyHostToDevice);

    //shuffle attributess to random choose attributess
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, ATTRIBUTE_SIZE - 1);
    std::vector<float> h_mean;
    std::unordered_set<unsigned int> idxs;
    unsigned int idx{0};
    for(int i{0}; i < K; i++) {
        do { idx = indices(rng); } while(idxs.find(idx) != idxs.end());
        idxs.insert(idx);
        for(int d{0}; d < DIMS; d++)
            h_mean.push_back(h_attrs[d * ATTRIBUTE_SIZE + idx]);
    }

    cudaMemcpy(mean, h_mean.data(), mean_bytes, cudaMemcpyHostToDevice);
    cudaMemset(counts, 0, count_bytes);
    cudaMemset(assigments, 0, ATTRIBUTE_SIZE * sizeof(unsigned int));
    _sync();
}


Device::~Device() {
	if(attributes != nullptr) cudaFree(attributes);
	if(mean != nullptr)       cudaFree(mean);
	if(counts != nullptr)     cudaFree(counts);
    if(assigments != nullptr) cudaFree(assigments);
}


void Device::_select_device() {
    int device_id{0};
    cudaGetDeviceProperties(&_gpu_props, device_id);
    std::cout << "Running on \"" << _gpu_props.name << "\" under CUDA." << std::endl;
}


void Device::run_k_means() {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float t_assign{0}, t_reduction{0}, t_mean{0};

    for (size_t i{0}; i < ITERATIONS; ++i) {
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
	const int max_group  = _gpu_props.maxThreadsPerBlock;
    const int threads = (elements <= max_group) ? elements : max_group;
    const int work_items = (elements <= max_group) ? elements : elements + threads - (elements % threads);
    const int blocks     = (elements <= max_group) ? 1 : work_items / threads;

    return std::tuple<int,int,int>(blocks, threads, work_items);
}


void Device::_sync() {
    cudaDeviceSynchronize();
}


void Device::_assign_clusters() {
    constexpr int threads = ASSIGN_BLOCK_SIZE_NVIDIA;
    const int blocks      = ATTRIBUTE_SIZE / threads + (ATTRIBUTE_SIZE % threads == 0 ? 0 : 1);

    assign_clusters<<<blocks, threads>>>(attributes, mean, assigments);
}


void Device::_reduction() {
    const int remainder_attr = ATTRIBUTE_SIZE % RED_ATTRS_PACK_NVIDIA;
    const int quotient_attr  = ATTRIBUTE_SIZE / RED_ATTRS_PACK_NVIDIA;
    const int attr_pckg      = quotient_attr + (remainder_attr == 0 ? 0 : 1);
    const int remainder_dims = DIMS % RED_DIMS_PACK_NVIDIA;
    const int quotient_dims  = DIMS / RED_DIMS_PACK_NVIDIA;
    const int dims_pckg      = quotient_dims + (remainder_dims == 0 ? 0 : 1);

    dim3 threads, blocks;
    threads.x = RED_DIMS_PACK_NVIDIA;
    threads.y = RED_ATTRS_PACK_NVIDIA;
    blocks.x  = dims_pckg;
    blocks.y  = attr_pckg;

    size_t size_mean  = RED_DIMS_PACK_NVIDIA * RED_ATTRS_PACK_NVIDIA * sizeof(float);
    size_t size_label = RED_ATTRS_PACK_NVIDIA * sizeof(unsigned int);

    cudaMemset(mean, 0, mean_bytes);
    cudaMemset(counts, 0, count_bytes);
    _sync();
	
    reduction<<<blocks, threads, size_mean + size_label>>>(
        RED_DIMS_PACK_NVIDIA * RED_ATTRS_PACK_NVIDIA,
        remainder_attr,
        quotient_attr,
        remainder_dims,
        quotient_dims,
        attributes,
        mean,
        assigments,
        counts
    );
}


void Device::_compute_mean() {
    std::tie (blocks, threads, work_items) = _get_block_threads(K);
    compute_mean<<<blocks, threads>>>(mean, counts);
}


void Device::save_solution(std::vector<float>& h_mean) {
    cudaMemcpy(h_mean.data(), mean, mean_bytes, cudaMemcpyDeviceToHost);
    _sync();
}
