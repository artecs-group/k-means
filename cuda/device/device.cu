#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include <chrono>
#include "./device.cuh"

__device__ float squared_l2_distance(float x_1, float x_2) {
    float a = x_1 - x_2;
    return a*a;
}


__global__ void assign_clusters(int attrs_size, int k, int dims,
    float* __restrict__ attrs, float* __restrict__ mean, float* __restrict__ sum,
    unsigned int* __restrict__ counts, int* __restrict__ assigments)
{ 
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;

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

    for(int cluster{0}; cluster < k; cluster++) {
        for(int d{0}; d < dims; d++) {
            int val_id = (d * attrs_size) + global_index;
            int sum_id = attrs_size * cluster * dims + val_id;
            sum[sum_id]  = (best_cluster == cluster) ? attrs[val_id] : 0;
        }
        counts[attrs_size * cluster + global_index] = (best_cluster == cluster) ? 1 : 0;
    }
}


template <typename T>
__global__ void reduce(size_t attrs_size, size_t k, size_t dims, size_t dim_offset, T* __restrict__ vec)
{
    // cast in order to keep the T type
    extern __shared__ __align__(sizeof(T)) unsigned char smem[];
    T* shared_data = reinterpret_cast<T*>(smem);

    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_index  = threadIdx.x;
    const int x            = local_index;

    for (int cluster = 0; cluster < k; cluster++) {
        // load by cluster
        shared_data[x] = vec[(attrs_size * cluster * dims) + global_index + dim_offset];

        // tree reduction
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            __syncthreads();
            if (local_index < stride)
                shared_data[x] += shared_data[x + stride];
        }

        if (local_index == 0) {  
            int cluster_index  = (attrs_size * cluster * dims) + blockIdx.x + dim_offset;
            vec[cluster_index] = shared_data[x];
        }
    }
}


__global__ void compute_mean(int attrs_size, int dims, float* __restrict__ mean, 
    float* __restrict__ sum, unsigned int* __restrict__ counts)
{
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count        = (1 < counts[attrs_size * global_index]) ? counts[attrs_size * global_index] : 1;
    for(int d{0}; d < dims; d++) {
        int id = (global_index * attrs_size * dims) + (d * attrs_size);
        mean[global_index * dims + d] = sum[id] / count;
    }
}


Device::Device(int _k, int _dims, int n_attrs, std::vector<float>& h_attrs): k(_k), dims(_dims){
    _select_device();
    
    attributes_size     = n_attrs;
    std::tie (blocks, threads, work_items) = _get_block_threads(attributes_size);

    attributes_size_pad = attributes_size + (work_items - attributes_size);
    attributes_bytes    = attributes_size * dims * sizeof(float);
    mean_bytes          = k * dims * sizeof(float);
    sum_size            = k * dims * attributes_size;
    sum_bytes           = sum_size * sizeof(float);
    count_bytes         = k * attributes_size * sizeof(unsigned int);

    cudaMalloc(&attributes, attributes_size_pad * sizeof(float));
    cudaMalloc(&mean, mean_bytes);
    cudaMalloc(&sum, sum_bytes);
    cudaMalloc(&counts, count_bytes);
    cudaMalloc(&assigments, attributes_size_pad * sizeof(int));

    // init pad values
    cudaMemset(attributes, 0, attributes_size_pad * dims * sizeof(float));
    _sync();
    cudaMemcpy(attributes, h_attrs.data(), attributes_bytes, cudaMemcpyHostToDevice);

    //shuffle attributess to random choose attributess
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::uniform_int_distribution<size_t> indices(0, attributes_size - 1);
    std::vector<float> h_mean;
    for(int i{0}; i < k; i++) {
        int idx = indices(rng);
        for(int j{0}; j < dims; j++)
            h_mean.push_back(h_attrs[idx + j * attributes_size]);
    }

    cudaMemcpy(mean, h_mean.data(), mean_bytes, cudaMemcpyHostToDevice);
    cudaMemset(sum, 0, sum_bytes);
    cudaMemset(counts, 0, count_bytes);
    cudaMemset(assigments, 0, attributes_size_pad * sizeof(int)); // potential bug: try init to -1
    _sync();
}


Device::~Device() {
	if(attributes != nullptr) cudaFree(attributes);
	if(mean != nullptr)       cudaFree(mean);
	if(sum != nullptr)        cudaFree(sum);
	if(counts != nullptr)     cudaFree(counts);
    if(assigments != nullptr) cudaFree(assigments);
}


void Device::_select_device() {
    int device_id{0};
    cudaGetDeviceProperties(&_gpu_props, device_id);
    std::cout << "Running on \"" << _gpu_props.name << "\" under CUDA." << std::endl;
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
        _manage_reduction();
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
    std::tie (blocks, threads, work_items) = _get_block_threads(attributes_size);

    assign_clusters<<<blocks, threads>>>(
        attributes_size,
        k,
        dims,
        attributes,
        mean,
        sum,
        counts,
        assigments
    );
}


template <typename T>
void Device::_reduce(T* vec, size_t _dims, size_t dim_offset) {
    int shared_size = threads * sizeof(T);
    reduce<<<blocks, threads, shared_size>>>(attributes_size, k, _dims, dim_offset, vec);
}


void Device::_manage_reduction() {
    int elements{attributes_size};
    std::tie (blocks, threads, work_items) = _get_block_threads(elements);

    // iterate 'till all elements are equals to number of clusters, 
    // which means that all the reductions are done.
    while (elements > k) {
        // reduce each dimension
        for(int d{0}; d < dims; d++)
            _reduce<float>(this->sum, dims, d*attributes_size);
    
        _reduce<unsigned int>(this->counts, 1, 0);
        _sync();

        // re-calculate how many elements will need for next iteration
        // each group will produce a partial solution of each cluster
        elements = blocks*k;
        std::tie (blocks, threads, work_items) = _get_block_threads(elements);
    }
}


void Device::_compute_mean() {
    std::tie (blocks, threads, work_items) = _get_block_threads(k);
    compute_mean<<<blocks, threads>>>(
        attributes_size,
        dims,
        mean,
        sum,
        counts
    );
}


void Device::save_solution(std::vector<float>& h_mean) {
    cudaMemcpy(h_mean.data(), mean, mean_bytes, cudaMemcpyDeviceToHost);
    _sync();
}
