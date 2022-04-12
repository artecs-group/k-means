#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <float.h>

#include "main.h"
#include "init.h"
#include "gpu.h"


/*-----------------------------------------------------------------------------------------*/
/* K-means clustering on the CPU                                                           */
/*-----------------------------------------------------------------------------------------*/
void cpuKmeans(void)
{
    double tolerance;

    // Reset global variables to zeros
    NbIters = 0;
    track = 0;	
    Tms_init = 0.0f;
    Tms_compute_assign = 0.0f;
    Tms_update = 0.0f;

    #pragma omp parallel
    {	
	    // Declare variables for time recording
        double tb_init, tf_init, ts_init = 0.0;
        double tb_compute_assign, tf_compute_assign, ts_compute_assign = 0.0;
        double tb_update, tf_update, ts_update = 0.0; 

        // Declare a seed for rand_r() function
        unsigned int seed;

        /*---------------------------------------------------------------------------------*/
        /* Initial centroids selection                                                     */
        /*---------------------------------------------------------------------------------*/
        if (INPUT_INITIAL_CENTROIDS == "") {
            tb_init = omp_get_wtime();
            #pragma omp for
            for (int k = 0; k < NbClusters; k++) {
                seed = k;
                int idx = rand_r(&seed)/(T_real)RAND_MAX * NbPoints;  // rand_r() is multithread safe
                for (int j = 0; j < NbDims; j++)
                    centroid[k][j] = data[idx*NbDims + j];
            }
            tf_init = omp_get_wtime(); 
            ts_init += (tf_init - tb_init);
        }

            // To uncomment the following code if needed (output initial centroids into a text file)
            /*
            #pragma omp single 
            {
                fp = fopen(OUTPUT_INITIAL_CENTROIDS, "w");
                if (fp == NULL) {
                    printf("Fail to open file!\n");
                    exit(0);
                }
                for (int i = 0; i < NbClusters; i++) {
                    for (int j = 0; j < NbDims; j++)
                        fprintf(fp, "%f ", centroid[i][j]);
                    fprintf(fp, "\n");
                }
                fclose(fp);
            }*/

        /*---------------------------------------------------------------------------------*/
        /* Two-phase iterations                                                            */
        /*---------------------------------------------------------------------------------*/
        do {
            /*-----------------------------------------------------------------------------*/
            /* Compute point-centroid distances & Assign each point to its nearest centroid*/
            /*-----------------------------------------------------------------------------*/
            tb_compute_assign = omp_get_wtime();
            #pragma omp for reduction(+: track)
            for (int i = 0; i < NbPoints; i++) {
                int min = 0;
                T_real dist_sq, minDist_sq = FLT_MAX;
                for (int k = 0; k < NbClusters; k++) {
                    dist_sq = 0.0;
                    // Calculate the square of distance between instance i and centroid k across NbDims dimensions
                    for (int j = 0; j < NbDims; j ++)
                        dist_sq += (data[i*NbDims + j] - centroid[k][j])*(data[i*NbDims + j] - centroid[k][j]);
                    // Find and record the nearest centroid to instance i
                    bool a = (dist_sq < minDist_sq);
                    min = (a ? k : min);
                    minDist_sq = (a ? dist_sq : minDist_sq);
                }

                // Change the label if necessary and count this change into track
                if (label[i] != min) {
                    track++;
                    label[i] = min;
                }
            }
            tf_compute_assign = omp_get_wtime();
            ts_compute_assign += (tf_compute_assign - tb_compute_assign);


            /*-----------------------------------------------------------------------------*/
            /* Update centroids                                                            */
            /*-----------------------------------------------------------------------------*/
            tb_update = omp_get_wtime();
            #pragma omp for
            for(int k = 0; k < NbClusters; k++){
                count[k] = 0;
                for(int j = 0; j < NbDims; j++)
                centroid[k][j] = 0.0;
            }

            // In order to reduce the rounding error which happens when adding numbers of very different magnitudes,
            // we first divide the dataset into packages, then calculate the sum of points in each package, finally compute the sum of all packages.
            int quotient, remainder, offset, length;
            quotient = NbPoints/NbPackages;
            remainder = NbPoints%NbPackages;
            // Sum the contributions to each cluster
            #pragma omp for private(package) reduction(+: count, centroid)
            for(int p = 0; p < NbPackages; p++){   // Process by package
                offset = (p < remainder ? ((quotient + 1) * p) : (quotient * p + remainder));
                length = (p < remainder ? (quotient + 1) : quotient);
                // Reset "package" to zeros
                for(int k = 0; k < NbClusters; k++)
                    for(int j = 0; j < NbDims; j++)
                        package[k][j] = 0.0;

                // 1st step local reduction
                // - Count nb of instances in OpenMP reduction array
                // - Reduction in thread private array
                for(int i = offset; i < offset + length; i++){
                    int k = label[i];
                    count[k]++;
                    for(int j = 0; j < NbDims; j++)
                        package[k][j] += data[i*NbDims + j];
                }
                // 2nd step local reduction
                // - Reduction in local OpenMP reduction array
                for(int k = 0; k < NbClusters; k++){
                    for(int j = 0; j < NbDims; j++)
                        centroid[k][j] += package[k][j];
                }
            }   // 2nd step global reduction: final reduction by OpenMP in global "centroid" array

            // Final averaging to get new centroids
            #pragma omp for
            for(int k = 0; k < NbClusters; k++)   // Process by cluster
                if (count[k] != 0) 
                    for(int j = 0; j < NbDims; j++)
                        centroid[k][j] /= count[k];  // - Update global "centroid" array

            tf_update = omp_get_wtime();
            ts_update += (tf_update - tb_update);

            /*-----------------------------------------------------------------------------*/
            /* Calculate the variables for checking stopping criteria                      */
            /*-----------------------------------------------------------------------------*/
            #pragma omp single
            {
                NbIters++;     // Count the number of iterations
                tolerance = (double)track / NbPoints;
                track = 0; 
                //printf("Track = %llu  Tolerance = %lf\n", track, tolerance); 
            }
        } while (NbIters < MaxNbIters);
<<<<<<< HEAD
=======
        //} while (tolerance > TOL && NbIters < MaxNbIters);
>>>>>>> 9cf84fc83c36a1223a998dd1a21546c7bb715f69

        // Store the elapsed time in ms in global variables
        #pragma omp single
        {
            Tms_init = (float) ts_init*1E3;
            Tms_compute_assign = (float) ts_compute_assign*1E3;
            Tms_update = (float) ts_update*1E3;
        }
    }
}


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
    tb_application = omp_get_wtime();
    CommandLineParsing(argc,argv);
    omp_set_num_threads(NbThreads);
    PrintConfig();

    // Load input dataset
    fprintf(stdout,"- Loading data from text file into CPU RAM...\n");
    tb_input = omp_get_wtime();
    InputDataset();
    tf_input = omp_get_wtime();	

    fprintf(stdout,"- Parallel computation starts...\n");

    if (OnGPUFlag)
        gpuInit();       // Init the GPU device

    // Run computation on the GPU or on the CPU
    tb_transfer_computation = omp_get_wtime();
    if (OnGPUFlag) {
        gpuSetDataOnGPU();
        tb_computation = omp_get_wtime();
        gpuKmeans();
        tf_computation = omp_get_wtime();
        gpuGetResultOnCPU();
    } else {
        tb_computation = omp_get_wtime();
        cpuKmeans();
        tf_computation = omp_get_wtime();
    }
    tf_transfer_computation = omp_get_wtime();

    if (OnGPUFlag)
        gpuFinalize();   // Finalize GPU device usage

    tb_output = omp_get_wtime();
    OutputResult();
    tf_output = omp_get_wtime();

    tf_application = omp_get_wtime();

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
