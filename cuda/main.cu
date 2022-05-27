#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "device/device.cuh"

int main(int argc, const char* argv[]) {
    if (argc < 5) {
        std::cerr << "usage: k_means <data-file> <clusters> <dimensions> <number-points> [iterations]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto clusters             = std::atoi(argv[2]);
    const auto dims                 = std::atoi(argv[3]);
    const auto n_points             = std::atoi(argv[4]);
    const auto number_of_iterations = (argc == 6) ? std::atoi(argv[5]) : 300;

    std::vector<float> h_attrs(n_points*dims);
    std::ifstream stream(argv[1]);
    std::string line;
    for(int i{0}; std::getline(stream, line) && i < n_points; i++) {
        std::istringstream line_stream(line);
        float val;
        for(int j{0}; j < dims; j++) {
            line_stream >> val;
            h_attrs[j*n_points + i] = val;
        }
    }

    Device device = Device(clusters, dims, n_points, h_attrs);

    const auto start = std::chrono::high_resolution_clock::now();
    device.run_k_means(number_of_iterations);
    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cout << "Total time = " << duration.count() << "s" << std::endl;
    std::cout << "Clusters   = " << clusters << std::endl
              << "Dimensions = " << dims << std::endl
              << "Atributes  = " << n_points << std::endl
              << "Iterations = " << number_of_iterations << std::endl;
              
    std::vector<float> mean(clusters*dims, 0);
    device.save_solution(mean);

    // std::cout << std::endl << "Clusters:" << std::endl;
    // for (size_t cluster{0}; cluster < clusters; ++cluster) {
    //     for(size_t d{0}; d < dims; d++)
    //         std::cout << mean[cluster * dims + d] << " ";
    //     std::cout << std::endl;
    // }

    return 0;
}
