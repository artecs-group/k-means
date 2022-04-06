#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "device/device.hpp"

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
            h_attrs[i + n_points*j] = val;
        }
    }

    Device device = Device(clusters, dims, n_points, h_attrs);

    const auto start = std::chrono::high_resolution_clock::now();
    device.run_k_means(number_of_iterations);
    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cerr << "Took: " << duration.count() << "s" << std::endl;

    std::vector<float> mean(clusters*dims, 0);
    device.save_solution(mean);

    for (size_t cluster = 0; cluster < clusters; ++cluster)
        std::cout << mean_x[cluster] << " " << mean_y[cluster] << std::endl;

    return 0;
}
