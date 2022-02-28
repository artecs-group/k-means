#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <stdio.h>
#include <random>
#include "oneapi/dal.hpp"
#include <CL/sycl.hpp>


int main(int argc, const char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: k_means <data-file> <k> <number-points> [iterations]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string file_name = argv[1];
    const auto k = std::atoi(argv[2]);
    const auto n_points = std::atoi(argv[3]);
    const auto number_of_iterations = (argc == 5) ? std::atoi(argv[4]) : 300;

    const auto queue = sycl::queue{sycl::gpu_selector{}};
    const auto data = oneapi::dal::read<oneapi::dal::table>(queue, oneapi::dal::csv::data_source{file_name});

    const auto kmeans_init_desc = oneapi::dal::kmeans_init::descriptor<float, oneapi::dal::kmeans_init::method::random_dense>().set_cluster_count(k);
    const auto result_init = oneapi::dal::compute(queue, kmeans_init_desc, data);
    const auto kmeans_desc = oneapi::dal::kmeans::descriptor<float>{}
        .set_cluster_count(k)
        .set_max_iteration_count(number_of_iterations);

    const auto start    = std::chrono::high_resolution_clock::now();
    const auto result   = train(kmeans_desc, data, result_init.get_centroids());
    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    
    std::cerr << "Took: " << duration.count() << "s" << std::endl;
    print_table("centroids", result.get_model().get_centroids());

    return 0;
}
