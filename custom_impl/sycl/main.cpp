#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "device/device.hpp"

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cerr << "usage: k_means <data-file>" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<float> h_attrs(ATTRIBUTE_SIZE*DIMS);
    std::ifstream stream(argv[1]);
    std::string line;
    for(int i{0}; std::getline(stream, line) && i < ATTRIBUTE_SIZE; i++) {
        std::istringstream line_stream(line);
        float val;
        for(int j{0}; j < DIMS; j++) {
            line_stream >> val;
#if defined(SYCL_NGPU) || defined(SYCL_PORTABLE)
            h_attrs[j*ATTRIBUTE_SIZE + i] = val;
#else
            h_attrs[i*DIMS + j] = val;
#endif     
            
        }
    }

    Device device = Device(h_attrs);

    const auto start = std::chrono::high_resolution_clock::now();
    device.run_k_means();
    const auto end      = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
    std::cout << "Total time = " << duration.count() << "s" << std::endl;
    std::cout << "Clusters   = " << CLUSTERS << std::endl
              << "Dimensions = " << DIMS << std::endl
              << "Atributes  = " << ATTRIBUTE_SIZE << std::endl
              << "Iterations = " << ITERATIONS << std::endl;
              
    std::vector<float> mean(CLUSTERS*DIMS, 0);
    device.save_solution(mean);

    // std::cout << std::endl << "Clusters:" << std::endl;
    // for (size_t cluster{0}; cluster < CLUSTERS; ++cluster) {
    //     for(size_t d{0}; d < DIMS; d++)
    //         std::cout << mean[cluster * DIMS + d] << " ";
    //     std::cout << std::endl;
    // }

    return 0;
}