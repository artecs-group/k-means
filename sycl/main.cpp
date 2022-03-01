#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "device/device.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 4) {
        std::cerr << "usage: k_means <data-file> <k> <number-points> [iterations]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const auto k = std::atoi(argv[2]);
    const auto n_points = std::atoi(argv[3]);
    const auto number_of_iterations = (argc == 5) ? std::atoi(argv[4]) : 300;

    std::vector<float> h_x;
    std::vector<float> h_y;
    std::ifstream stream(argv[1]);
    std::string line;
    for(int i{n_points}; std::getline(stream, line) && i > 0; i--) {
        std::istringstream line_stream(line);
        float x, y;
        uint16_t label;
        line_stream >> x >> y >> label;
        h_x.push_back(x);
        h_y.push_back(y);
    }

    Device device = Device(k, h_x, h_y);

    const auto start = std::chrono::high_resolution_clock::now();
    device.run_k_means(number_of_iterations);
    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cerr << "Took: " << duration.count() << "s" << std::endl;

    std::vector<float> mean_x(k, 0);
    std::vector<float> mean_y(k, 0);
    device.save_solution(mean_x, mean_y);

    for (size_t cluster = 0; cluster < k; ++cluster)
        std::cout << mean_x[cluster] << " " << mean_y[cluster] << std::endl;

    return 0;
}
