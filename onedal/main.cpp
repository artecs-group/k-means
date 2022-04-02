/*
 * Compile: dpcpp --gcc-toolchain=/usr -std=c++17 -pedantic -Wall -Wextra -Werror -Wno-unused-parameter -fsycl-device-code-split=per_kernel -I./source -I"${ONEAPI_ROOT}/dal/latest/include" main.cpp -o main -L"${ONEAPI_ROOT}/dal/latest/lib/intel64" -L"${ONEAPI_ROOT}/tbb/latest/env/../lib/intel64/gcc4.8" -lonedal_dpc -lonedal_core -lonedal_thread "${ONEAPI_ROOT}/dal/latest/lib/intel64"/libonedal_sycl.a -ltbb -ltbbmalloc -lpthread -lOpenCL -ldl
*/

#include <CL/sycl.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal.hpp"


/*
 * For clusters printing
*/
std::ostream &operator<<(std::ostream &stream, const oneapi::dal::table &table) {
    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    if (table.get_row_count() <= 10) {
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    else {
        for (std::int64_t i = 0; i < 5; i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
        std::cout << "..." << (table.get_row_count() - 10) << " lines skipped..." << std::endl;
        for (std::int64_t i = table.get_row_count() - 5; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    return stream;
}


int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cerr << "usage: k_means <data-file> <k> <number-points> [iterations]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::string file_name = argv[1];
    const auto k = std::atoi(argv[2]);
    const auto number_of_iterations = (argc == 4) ? std::atoi(argv[3]) : 300;

    auto queue = sycl::queue{sycl::cpu_selector{}};
    std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    const auto data = oneapi::dal::read<oneapi::dal::table>(queue, oneapi::dal::csv::data_source{file_name});

    const auto kmeans_init_desc = oneapi::dal::kmeans_init::descriptor<float, oneapi::dal::kmeans_init::method::random_dense>().set_cluster_count(k);
    const auto result_init = oneapi::dal::compute(queue, kmeans_init_desc, data);
    const auto kmeans_desc = oneapi::dal::kmeans::descriptor<float>{}
        .set_cluster_count(k)
        .set_max_iteration_count(number_of_iterations);

    std::cout << "Starting computation ..." << std::endl;

    const auto start    = std::chrono::high_resolution_clock::now();
    const auto result   = oneapi::dal::train(queue, kmeans_desc, data, result_init.get_centroids());
    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    
    std::cout << "Took: " << duration.count() << "s" << std::endl
              << "Iteration count: " << result.get_iteration_count() << std::endl;

    std::cout << "Centroids:\n" << result.get_model().get_centroids() << std::endl;

    return 0;
}
