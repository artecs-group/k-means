#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>

#include "main.h"
#include "init.h"
#include "gpu.h"


/*-----------------------------------------------------------------------------------------*/
/* Toplevel function                                                                       */
/*-----------------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    // Declare local variables for time recording
    double tb_input, tf_input;
    double tb_output, tf_output;
    double tb_computation, tf_computation;
    double tb_transfer_computation, tf_transfer_computation;
    double tb_application, tf_application;

    // Initialisations
    tb_application = std::chrono::high_resolution_clock::now();
    CommandLineParsing(argc,argv);
    omp_set_num_threads(NbThreads);
    PrintConfig();

    // Load input dataset
    fprintf(stdout,"- Loading data from text file into CPU RAM...\n");
    tb_input = std::chrono::high_resolution_clock::now();
    InputDataset();
    tf_input = std::chrono::high_resolution_clock::now();	

    fprintf(stdout,"- Parallel computation starts...\n");

    if (OnGPUFlag)
        gpuInit();       // Init the GPU device

    // Run computation on the GPU or on the CPU
    tb_transfer_computation = std::chrono::high_resolution_clock::now();
    gpuSetDataOnGPU();
    tb_computation = std::chrono::high_resolution_clock::now();
    gpuKmeans();
    tf_computation = std::chrono::high_resolution_clock::now();
    gpuGetResultOnCPU();
    tf_transfer_computation = std::chrono::high_resolution_clock::now();

    if (OnGPUFlag)
        gpuFinalize();   // Finalize GPU device usage

    tb_output = std::chrono::high_resolution_clock::now();
    OutputResult();
    tf_output = std::chrono::high_resolution_clock::now();

    tf_application = std::chrono::high_resolution_clock::now();

    // Calculate the elapsed time
    Ts_input = tf_input - tb_input;
    Ts_output = tf_output - tb_output;
    Ts_computation = tf_computation - tb_computation;
    Ts_transfer_computation = tf_transfer_computation - tb_transfer_computation;
    Ts_transfer = Ts_transfer_computation - Ts_computation;
    Ts_application = tf_application - tb_application;

    // Results and performance printing
    PrintResultsAndPerf();

    // End of the parallel program
    return(EXIT_SUCCESS);
}
