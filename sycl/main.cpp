#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>

#include "main.hpp"
#include "init/init.hpp"
#include "device/device.hpp"


/*-----------------------------------------------------------------------------------------*/
/* Toplevel function                                                                       */
/*-----------------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    // Declare local variables for time recording
    std::chrono::time_point<std::chrono::steady_clock> tb_input, tf_input;
    std::chrono::time_point<std::chrono::steady_clock> tb_output, tf_output;
    std::chrono::time_point<std::chrono::steady_clock> tb_computation, tf_computation;
    std::chrono::time_point<std::chrono::steady_clock> tb_transfer_computation, tf_transfer_computation;
    std::chrono::time_point<std::chrono::steady_clock> tb_application, tf_application;

    // Initialisations
    tb_application = std::chrono::steady_clock::now();
    CommandLineParsing(argc,argv);
    PrintConfig();
    // Load input dataset
    fprintf(stdout,"- Loading data from text file into CPU RAM...\n");
    tb_input = std::chrono::steady_clock::now();
    InputDataset();
    tf_input = std::chrono::steady_clock::now();	
    fprintf(stdout,"- Parallel computation starts...\n");
    device_init();       // Init the device
    // Run computation 
    tb_transfer_computation = std::chrono::steady_clock::now();
    set_data_device();
    tb_computation = std::chrono::steady_clock::now();
    run_k_means();
    tf_computation = std::chrono::steady_clock::now();
    get_result_host();
    tf_transfer_computation = std::chrono::steady_clock::now();
    device_finish();
    tb_output = std::chrono::steady_clock::now();
    OutputResult();
    tf_output = std::chrono::steady_clock::now();
    tf_application = std::chrono::steady_clock::now();

    // Calculate the elapsed time
    Ts_input = std::chrono::duration<double>(tf_input - tb_input).count();
    Ts_output = std::chrono::duration<double>(tf_output - tb_output).count();
    Ts_computation = std::chrono::duration<double>(tf_computation - tb_computation).count();
    Ts_transfer_computation = std::chrono::duration<double>(tf_transfer_computation - tb_transfer_computation).count();
    Ts_transfer = std::chrono::duration<double>(Ts_transfer_computation - Ts_computation).count();
    Ts_application = std::chrono::duration<double>(tf_application - tb_application).count();

    // Results and performance printing
    PrintResultsAndPerf();

    // End of the parallel program
    return(EXIT_SUCCESS);
}
