#include <iostream>
#include <random>
#include <algorithm>
#include <cfloat>
#include "./device.hpp"

Device::Device(int _k, std::vector<float>& h_x, std::vector<float>& h_y): k(_k){
    _queue = _get_queue();
    
    point_size     = h_x.size();
    std::tie (groups, group_size, work_items) = _get_group_work_items(point_size);

    point_size_pad = point_size + (work_items - point_size);
    point_bytes    = point_size * sizeof(float);
    mean_bytes     = k * sizeof(float);
    sum_size       = k * groups;
    sum_bytes      = sum_size * sizeof(float);
    count_bytes    = sum_size * sizeof(int);

    point_x = malloc_device<float>(point_size_pad * sizeof(float), _queue);
    point_y = malloc_device<float>(point_size_pad * sizeof(float), _queue);
    mean_x  = malloc_device<float>(mean_bytes, _queue);
    mean_y  = malloc_device<float>(mean_bytes, _queue);
    sum_x   = malloc_device<float>(sum_bytes, _queue);
    sum_y   = malloc_device<float>(sum_bytes, _queue);
    counts  = malloc_device<int>(count_bytes, _queue);

    // init pad values
    _queue.memset(point_x, 0, point_size_pad * sizeof(float));
    _queue.memset(point_y, 0, point_size_pad * sizeof(float));
    _sync();
    _queue.memcpy(point_x, h_x.data(), point_bytes);
    _queue.memcpy(point_y, h_y.data(), point_bytes);
    _sync();

    //init means
    std::mt19937 rng(std::random_device{}());
    rng.seed(0);
    std::shuffle(h_x.begin(), h_x.end(), rng);
    std::shuffle(h_y.begin(), h_y.end(), rng);
    _queue.memcpy(mean_x, h_x.data(), mean_bytes);
    _queue.memcpy(mean_y, h_y.data(), mean_bytes);

    _queue.memset(sum_x, 0, sum_bytes);
    _queue.memset(sum_y, 0, sum_bytes);
    _queue.memset(counts, 0, count_bytes);
    _sync();
}


Device::~Device() {
	if(point_x != nullptr) free(point_x, _queue);
	if(point_y != nullptr) free(point_y, _queue);
	if(mean_x != nullptr) free(mean_x, _queue);
	if(mean_y != nullptr) free(mean_y, _queue);
	if(sum_x != nullptr) free(sum_x, _queue);
	if(sum_y != nullptr) free(sum_y, _queue);
	if(counts != nullptr) free(counts, _queue);
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
        _fine_reduce();
        _sync();
        _middle_reduce();
        _sync();
        _coarse_reduce();
        _sync();
    }
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
    const int max_group  = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    const int group_size = (elements <= max_group) ? elements : max_group;
    const int work_items = (elements <= max_group) ? elements : elements + group_size - (elements % group_size);
    const int groups     = (elements <= max_group) ? 1 : work_items / group_size;

    return std::tuple<int,int,int>(groups, group_size, work_items);
}


void Device::_sync() {
    _queue.wait();
}


void Device::_fine_reduce() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(point_size);

    _queue.submit([&](handler& h) {
        // * 3 for x, y and counts.
        int point_s = this-> point_size;
        int k = this->k;
        float* point_x = this->point_x;
        float* point_y = this->point_y;
        float* mean_x = this->mean_x;
        float* mean_y = this->mean_y;
        float* sum_x = this->sum_x;
        float* sum_y = this->sum_y;
        int* counts = this->counts;

        int s_size = 3 * group_size * sizeof(float);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_data(s_size, h);

        h.parallel_for<class fine_reduce>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int global_index = item.get_global_id(0);
			const int local_index = item.get_local_id(0);

            // Load the mean values into shared memory.
            if (local_index < k) {
                shared_data[local_index] = mean_x[local_index];
                shared_data[k + local_index] = mean_y[local_index];
            }
            item.barrier(sycl::access::fence_space::local_space);

            // Load once here.
            const float x_value = point_x[global_index];
            const float y_value = point_y[global_index];

            float best_distance = FLT_MAX;
            int best_cluster = -1;
            for (int cluster = 0; cluster < k; ++cluster) {
                const float distance = 
                    squared_l2_distance(x_value, y_value, shared_data[cluster], shared_data[k + cluster]);
                
                if (distance < best_distance) {
                    best_distance = distance;
                    best_cluster = cluster;
                }
            }
            item.barrier(sycl::access::fence_space::local_space);
            
            // Reduction phase
            const int x     = local_index;
            const int y     = local_index + item.get_local_range(0);
            const int count = local_index + item.get_local_range(0) + item.get_local_range(0);

            for (int cluster = 0; cluster < k; ++cluster) {
                shared_data[x]     = (best_cluster == cluster) ? x_value : 0;
                shared_data[y]     = (best_cluster == cluster) ? y_value : 0;
                shared_data[count] = (best_cluster == cluster) ? 1 : 0;
                item.barrier(sycl::access::fence_space::local_space);

                // Reduction for this cluster.
                for (int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                    if (local_index < stride) {
                        shared_data[x]     += shared_data[x + stride];
                        shared_data[y]     += shared_data[y + stride];
                        shared_data[count] += shared_data[count + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                // Now shared_data[0] holds the sum for x.
                if (local_index == 0) {
                    const int cluster_index = item.get_group(0) * k + cluster;
                    sum_x[cluster_index]    = shared_data[x];
                    sum_y[cluster_index]    = shared_data[y];
                    counts[cluster_index]   = shared_data[count];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
        });
    });
}


void Device::_middle_reduce() {
    std::tie (groups, group_size, work_items) = _get_group_work_items(groups*k);

    // iterate till we just need one group for the coarse reduction
    while (groups > 1) {
        _queue.submit([&](handler& h) {
            int k = this->k;
            float* sum_x = this->sum_x;
            float* sum_y = this->sum_y;

            // * 2 for x and y. Will have k * blocks threads for the coarse reduction.
            int s_size = 2 * groups*k * sizeof(float);
            sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_data(s_size, h);

            h.parallel_for<class middle_reduce>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
                const int global_index = item.get_global_id(0);
                const int local_index  = item.get_local_id(0);
                const int x            = local_index;
                const int y            = local_index + item.get_local_range(0);

                shared_data[x] = sum_x[global_index];
                shared_data[y] = sum_y[global_index];
                item.barrier(sycl::access::fence_space::local_space);

                for (int cluster = 0; cluster < k; ++cluster) {
                    for (int stride = item.get_local_range(0) >> 1; stride > 0; stride >>= 1) {
                        if (local_index < stride) {
                            shared_data[x] += shared_data[x + stride];
                            shared_data[y] += shared_data[y + stride];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (local_index == 0) {
                        const int cluster_index = item.get_group(0) * k + cluster;
                        sum_x[cluster_index]    = shared_data[x];
                        sum_y[cluster_index]    = shared_data[y];
                    }
                }
            });
        });
        std::tie (groups, group_size, work_items) = _get_group_work_items(groups*k);
    }
     
}


void Device::_coarse_reduce() {
    // GT1030 -> gs = 392, wi = 392, g = 1

    // UHD630 -> gs = 256, wi = 100096, g = 391; size = 100000
    // UHD630 -> gs = 256, wi = 1792,   g = 7;   size = 4*7 = 1564
    // UHD630 -> gs = 28,  wi = 28,     g = 1;   size = 4*7 = 28

    _queue.submit([&](handler& h) {
        int k = this->k;
        int* counts = this->counts;
        float* mean_x = this->mean_x;
        float* mean_y = this->mean_y;
        float* sum_x = this->sum_x;
        float* sum_y = this->sum_y;

        // * 2 for x and y. Will have k * blocks threads for the coarse reduction.
        int s_size = 2 * groups*k * sizeof(float);
        sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> shared_data(s_size, h);

        h.parallel_for<class coarse_reduce>(nd_range(range(work_items), range(group_size)), [=](nd_item<1> item){
            const int index = item.get_local_id(0);
            const int y = item.get_local_id(0) + item.get_local_range(0);

            shared_data[index] = sum_x[index];
            shared_data[y]     = sum_y[index];
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = item.get_local_range(0) >> 1; stride >= k; stride >>= 1) {
                if (index < stride) {
                    shared_data[index] += shared_data[index + stride];
                    shared_data[y]     += shared_data[y + stride];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (index < k) {
                const int count = max(1, counts[index]);
                mean_x[index] = shared_data[index] / count;
                mean_y[index] = shared_data[y] / count;
                sum_x[index] = 0;
                sum_y[index] = 0;
                counts[index] = 0;
            }
        });
    });
}


void Device::save_solution(std::vector<float>& h_mean_x, std::vector<float>& h_mean_y) {
    _queue.memcpy(h_mean_x.data(), mean_x, mean_bytes);
    _queue.memcpy(h_mean_y.data(), mean_y, mean_bytes);
    _sync();
}


float squared_l2_distance(float x_1, float y_1, float x_2, float y_2) {
    float a = x_1 - x_2;
    float b = y_1 - y_2;
    return a*a + b*b;
}
