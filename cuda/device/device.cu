#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include "./device.cuh"

__device__ float squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
    float a = x_1 - x_2;
    float b = y_1 - y_2;
    return a*a + b*b;
}


__global__ void assign_clusters(int point_size, int k, 
    float* __restrict__ point_x, float* __restrict__ point_y, float* __restrict__ mean_x, 
    float* __restrict__ mean_y, float* __restrict__ sum_x, float* __restrict__ sum_y,
    float* __restrict__ counts, int* __restrict__ assigments)
{ 
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // Load once here.
    const float x_value = point_x[global_index];
    const float y_value = point_y[global_index];

    float best_distance = FLT_MAX;
    int best_cluster = -1;
    for (int cluster = 0; cluster < k; ++cluster) {
        const float distance = squared_l2_distance(x_value, y_value, mean_x[cluster], mean_y[cluster]);
        
        if (distance < best_distance) {
            best_distance = distance;
            best_cluster = cluster;
        }
    }
    assigments[global_index] = best_cluster;

    for(int cluster{0}; cluster < k; cluster++) {
        sum_x[point_size * cluster + global_index]  = (best_cluster == cluster) ? x_value : 0;
        sum_y[point_size * cluster + global_index]  = (best_cluster == cluster) ? y_value : 0;
        counts[point_size * cluster + global_index] = (best_cluster == cluster) ? 1 : 0;
    }
}


__global__ void reduce(int point_size, int k, float* __restrict__ vec)
{
    extern __shared__ float shared_data[];
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_index  = threadIdx.x;
    const int x            = local_index;

    for (int cluster = 0; cluster < k; ++cluster) {
        // load by cluster
        shared_data[x] = vec[point_size * cluster + global_index];
        __syncthreads();

        // apply reduction
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (local_index < stride)
                shared_data[x] += shared_data[x + stride];

            __syncthreads();
        }

        if (local_index == 0) {  
            const int cluster_index = point_size * cluster + blockIdx.x;
            vec[cluster_index]      = shared_data[x];
        }
    }
}


__global__ void compute_mean(int point_size, float* __restrict__ mean_x, 
    float* __restrict__ mean_y, float* __restrict__ sum_x, float* __restrict__ sum_y,
    float* __restrict__ counts)
{
    const int global_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count        = (int)(1 < counts[point_size * global_index]) ? counts[point_size * global_index] : 1;
    mean_x[global_index]   = sum_x[point_size * global_index] / count;
    mean_y[global_index]   = sum_y[point_size * global_index] / count;
}


Device::Device(int _k, std::vector<float>& h_x, std::vector<float>& h_y): k(_k){
    _select_device();
    
    point_size     = h_x.size();
    std::tie (blocks, threads, work_items) = _get_block_threads(point_size);

    point_size_pad = point_size + (work_items - point_size);
    point_bytes    = point_size * sizeof(float);
    mean_bytes     = k * sizeof(float);
    sum_size       = k * point_size;
    sum_bytes      = sum_size * sizeof(float);
    count_bytes    = sum_size * sizeof(float);

    cudaMalloc(&point_x, point_size_pad * sizeof(float));
    cudaMalloc(&point_y, point_size_pad * sizeof(float));
    cudaMalloc(&mean_x, mean_bytes);
    cudaMalloc(&mean_y, mean_bytes);
    cudaMalloc(&sum_x, sum_bytes);
    cudaMalloc(&sum_y, sum_bytes);
    cudaMalloc(&counts, count_bytes);
    cudaMalloc(&assigments, point_size_pad * sizeof(int));

    // init pad values
    cudaMemset(point_x, 0, point_size_pad * sizeof(float));
    cudaMemset(point_y, 0, point_size_pad * sizeof(float));
    _sync();
    cudaMemcpy(point_x, h_x.data(), point_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(point_y, h_y.data(), point_bytes, cudaMemcpyHostToDevice);

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

    cudaMemcpy(mean_x, h_mean_x.data(), mean_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(mean_y, h_mean_y.data(), mean_bytes, cudaMemcpyHostToDevice);
    cudaMemset(sum_x, 0, sum_bytes);
    cudaMemset(sum_y, 0, sum_bytes);
    cudaMemset(counts, 0, count_bytes);
    cudaMemset(assigments, 0, point_size_pad * sizeof(int)); // potential bug: try init to -1
    _sync();
}


Device::~Device() {
	if(point_x != nullptr)    cudaFree(point_x);
	if(point_y != nullptr)    cudaFree(point_y);
	if(mean_x != nullptr)     cudaFree(mean_x);
	if(mean_y != nullptr)     cudaFree(mean_y);
	if(sum_x != nullptr)      cudaFree(sum_x);
	if(sum_y != nullptr)      cudaFree(sum_y);
	if(counts != nullptr)     cudaFree(counts);
    if(assigments != nullptr) cudaFree(assigments);
}


void Device::_select_device() {
    int device_id{0};
    cudaGetDeviceProperties(&_gpu_props, device_id);
    std::cout << "Running on \"" << _gpu_props.name << "\" under CUDA." << std::endl;
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
    std::tie (blocks, threads, work_items) = _get_block_threads(point_size);

    assign_clusters<<<blocks, threads>>>(
        point_size,
        k,
        point_x,
        point_y,
        mean_x,
        mean_y,
        sum_x,
        sum_y,
        counts,
        assigments
    );
}


void Device::_reduce(float* vec) {
    int shared_size = threads * sizeof(float);
    reduce<<<blocks, threads, shared_size>>>(point_size, k, vec);
}


void Device::_manage_reduction() {
    int elements{point_size};
    std::tie (blocks, threads, work_items) = _get_block_threads(elements);

    while (elements > k) {
        _reduce(this->sum_x);
        _reduce(this->sum_y);
        _reduce(this->counts);
        _sync();
        elements = blocks*k;
        std::tie (blocks, threads, work_items) = _get_block_threads(elements);
    }
}


void Device::_compute_mean() {
    std::tie (blocks, threads, work_items) = _get_block_threads(k);
    compute_mean<<<blocks, threads>>>(
        point_size,
        mean_x,
        mean_y,
        sum_x,
        sum_y,
        counts
    );
}


void Device::save_solution(std::vector<float>& h_mean_x, std::vector<float>& h_mean_y) {
    cudaMemcpy(h_mean_x.data(), mean_x, mean_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mean_y.data(), mean_y, mean_bytes, cudaMemcpyDeviceToHost);
    _sync();
}
