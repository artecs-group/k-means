#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../main.hpp"
#include "init.hpp"


/*-----------------------------------------------------------------------------------------*/
/* Global variable declarations                                                            */
/*-----------------------------------------------------------------------------------------*/
int NbThreads;                           // Number of OpenMP threads on CPU
int OnGPUFlag;                           // Flag of computation on GPU     
int MaxNbIters;                          // Maximal number of iterations
int NbIters;                             // Count the number of iterations
T_real TOL;                              // Tolerance (convergence criterion)
T_real *data;                            // Array for the matrix of data instances
T_real *dataT;                           // Array for the transponsed matrix of data instances
T_real centroid[NbClusters][NbDims];     // Centroids 
T_real package[NbClusters][NbDims];      // Package
int *label;                              // Array for cluster labels of data points
unsigned long long int track;            // Number of points changing label between two iterations
int count[NbClusters];                   // Number of points in each cluster 

float Tms_init;                          // Time in ms of initializing centroids
float Tms_transpose;                     // Time in ms of transposing centroid matrix
float Tms_compute_assign;                // Time in ms of ComputeAssign routine
float Tms_update;                        // Time in ms of UpdateCentroids routine

double Ts_input;                         // Time in sec of loading input data
double Ts_transfer;                      // Time in sec of data transfers
double Ts_computation;                   // Time in sec of computation
double Ts_transfer_computation;          // Time in sec of data transfers and computation
double Ts_output;                        // Time in sec of writing clustering results
double Ts_application;                   // Time in sec of the application

double NumError = 0.0;                   // Average numerical error of final calculated centroids
FILE *fp;                                // File pointer


/*-----------------------------------------------------------------------------------------*/
/* Load input dataset                                                                      */
/*-----------------------------------------------------------------------------------------*/
void InputDataset(void)
{
    // Dynamic memory allocation for CPU arrays
    data = (T_real *) malloc(sizeof(T_real)*NbPoints*NbDims);
    dataT = (T_real *) malloc(sizeof(T_real)*NbDims*NbPoints);
    label = (int *) malloc(sizeof(int)*NbPoints);

    // Load input data from a text file
    fp = fopen(INPUT_DATA, "r");
    if (fp == NULL) {
        printf("    Fail to open data file!\n");
        exit(0);
    }
    int count1 = 0;
    int count2 = 0;
    T_real value;
    for (int i = 0; i < NbPoints; i++) {
        for (int j = 0; j < NbDims; j++) {
            count1 += fscanf(fp, T_REAL_PRINT, &value); 
            data[i*NbDims + j] = value;
            dataT[j*NbPoints + i] = value;
        }
        count2 += fscanf(fp, "\n");
    }
    if (count1 == NbPoints*NbDims)
        printf("    The input data have been successfully loaded!\n");
    else 
        printf("    Failed to load input data!\n");
    fclose(fp);


    // Load initial centroids from a text file
    if (INPUT_INITIAL_CENTROIDS != "") {
        fp = fopen(INPUT_INITIAL_CENTROIDS, "r");
        if (fp == NULL) {
            printf("    Fail to open inital centroids file!\n");
            exit(0);
        }
        count1 = 0;
        count2 = 0;
        for (int i = 0; i < NbClusters; i++) {
            for (int j = 0; j < NbDims; j++) {
                count1 += fscanf(fp, T_REAL_PRINT, &centroid[i][j]);
            }
            count2 += fscanf(fp, "\n");
        }
        if (count1 == NbClusters*NbDims)
            printf("    The initial centroids have been successfully loaded!\n");
        else 
            printf("    Failed to load initial centroids!\n");
        fclose(fp);
    }
}


/*-----------------------------------------------------------------------------------------*/
/* Output result                                                                           */
/*-----------------------------------------------------------------------------------------*/
void OutputResult(void)
{
    int CountPerCluster[NbClusters] = {0};

    // Write labels into a text file & Count the number of points in each cluster
    fp = fopen(OUTPUT_LABELS, "w");
    if (fp == NULL) {
        printf("Fail to open file!\n");
        exit(0);
    }
    int lb;
    for (int i = 0; i < NbPoints; i++) {
        lb = label[i];
        fprintf(fp, "%d\n", lb);
        CountPerCluster[lb]++;
    }
    fclose(fp);


    // Write the number of points of each cluster into a text file
    fp = fopen(OUTPUT_COUNT_PER_CLUSTER, "w");
    if (fp == NULL) {
        printf("Fail to open file!\n");
        exit(0);
    }
    for (int i = 0; i < NbClusters; i++) {
        fprintf(fp, "%d\n", CountPerCluster[i]);
    }
    fclose(fp);


    // Write the final centroids into a text file
    fp = fopen(OUTPUT_FINAL_CENTROIDS, "w");
    if (fp == NULL) {
        printf("Fail to open file!\n");
        exit(0);
    }
    for (int i = 0; i < NbClusters; i++) {
        for (int j = 0; j < NbDims; j++) {
            fprintf(fp, "%f ", centroid[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);


    // To uncomment when using the synthetic dataset specified in our paper
    /*
    NumError = 0.0;
    for (int i = 0; i < NbClusters; i++) {
        for (int j = 0; j < NbDims; j++) {
            T_real value = centroid[i][j];
            if (value > 50.0)
                NumError = NumError + fabs(value - 60.0);
            else
                NumError = NumError + fabs(value - 40.0);
        }
    }
    NumError = NumError / (NbDims*NbClusters);
    */


    // Free dynamic allocations
    free(data);
    free(dataT);
    free(label);
}


/*-----------------------------------------------------------------------------------------*/
/* Command Line parsing                                                                    */
/*-----------------------------------------------------------------------------------------*/
void usage(int ExitCode, FILE *std)
{
    fprintf(std,"KmeansClustering usage: \n");
    fprintf(std,"\t [-h]: print this help\n");
    fprintf(std,"\t [-t <GPU(default)|CPU>]: run computations on target GPU or on target CPU\n");
    fprintf(std,"\t [-cpu-nt <number of OpenMP threads> (default %d)]\n", DEFAULT_NB_THREADS);
    fprintf(std,"\t [-max-iters <max number of iterations> (default %d)]\n", DEFAULT_MAX_NB_ITERS);
    fprintf(std,"\t [-tol <tolerance> (default %f)]\n", DEFAULT_TOL);

    exit(ExitCode);
}

void CommandLineParsing(int argc, char *argv[])
{
    // Default init
    NbThreads = DEFAULT_NB_THREADS;
    OnGPUFlag = DEFAULT_ONGPUFLAG;     
    MaxNbIters = DEFAULT_MAX_NB_ITERS;
    TOL = DEFAULT_TOL;

    // Init from the command line
    argc--; argv++;
    while (argc > 0) {
        if (strcmp(argv[0],"-t") == 0) {
            argc--; argv++;
            if (argc > 0) {
                if (strcmp(argv[0],"GPU") == 0) {
                    OnGPUFlag = 1;
                    argc--; argv++;
                } else if (strcmp(argv[0],"CPU") == 0) {
                    OnGPUFlag = 0;
                    argc--; argv++;
                } else {
                    fprintf(stderr,"Error: unknown computation target '%s'!\n",argv[0]);
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-cpu-nt") == 0) {
            argc--; argv++;
            if (argc > 0) {
                NbThreads = atoi(argv[0]);
                argc--; argv++;
                if (NbThreads <= 0) {
                    fprintf(stderr,"Error: number of thread has to be >= 1!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-max-iters") == 0) {
            argc--; argv++;
            if (argc > 0) {
                MaxNbIters = atoi(argv[0]);
                argc--; argv++;
                if (MaxNbIters <= 0) {
                    fprintf(stderr,"Error: number of iterations has to be >= 1!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-tol") == 0) {
            argc--; argv++;
            if (argc > 0) {
				TOL = atof(argv[0]);
                argc--; argv++;
                if (TOL < 0.0f) {
                    fprintf(stderr,"Error: tolerance has to be >= 0!\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                usage(EXIT_FAILURE, stderr);
            }

        } else if (strcmp(argv[0],"-h") == 0) {
            usage(EXIT_SUCCESS, stdout);
        } else {
            usage(EXIT_FAILURE, stderr);
        }
    }

    if (OnGPUFlag) {
        if (NbClusters > 1024) {
            fprintf(stderr,"Error: number of clusters has to be <= 1024 on GPU!\n");
            exit(EXIT_FAILURE);
        }
        if (BSXP*BSYD < NbClusters || BSXP*BSYD > 1024) {
            fprintf(stderr,"Error: BSXP*BSYD has to be in [NbClusters, 1024]!\n");
            exit(EXIT_FAILURE);
        }
    }

    if (NbPackages > NbPoints) {
        fprintf(stderr,"Error: number of packages has to be <= number of points!\n");
        exit(EXIT_FAILURE);
    }
}


/*-----------------------------------------------------------------------------------------*/
/* Print configurations                                                                    */
/*-----------------------------------------------------------------------------------------*/
void PrintConfig(void)
{
    fprintf(stdout,"K-means clustering %s ", (OnGPUFlag ? "on GPU" : "on CPU"));
    if (OnGPUFlag == 0)
        fprintf(stdout,"with %d OpenMP thread%s ", NbThreads, (NbThreads > 1 ? "s" : ""));
    fprintf(stdout,"in %s precision:\n", T_REAL_TEXT);

    fprintf(stdout, "- Dataset:\n");
    fprintf(stdout,"    Nb of data instances: %d\n", NbPoints);
    fprintf(stdout,"    Nb of dimensions:     %d\n", NbDims);
    fprintf(stdout,"    Nb of clusters:       %d\n", NbClusters);
    fprintf(stdout,"    Nb of packages:       %d\n", NbPackages);

    fprintf(stdout, "- Stopping criterion:\n");
    fprintf(stdout,"    Tolerance:       %f\n", TOL);
    fprintf(stdout,"    Max nb of iters: %d\n", MaxNbIters);

    if (OnGPUFlag) {
        fprintf(stdout, "- Block size & Nb of streams:\n");
        fprintf(stdout,"    BSXN: %d\n", BSXN);
        fprintf(stdout,"    BSXP: %d\n", BSXP);
        fprintf(stdout,"    BSXK: %d\n", BSXK);
        fprintf(stdout,"    BSYD: %d\n", BSYD);
        fprintf(stdout,"    Nb of streams for Update_step1: %d\n", nStreams1);
        fprintf(stdout,"    Nb of streams for Update_step2: %d\n", nStreams2);
    }

    fflush(stdout);
}


/*-----------------------------------------------------------------------------------------*/
/* Print results and performances of the parallel computation                              */
/*-----------------------------------------------------------------------------------------*/
void PrintResultsAndPerf(void)
{
    fprintf(stdout,"- Results & Performance:\n");

    //fprintf(stdout,"    Time of loading input data:               %f s\n", (float) Ts_input);
    if (OnGPUFlag)
        fprintf(stdout,"    Time of data transfers:                   %f ms\n", (float) Ts_transfer*1E3);
    if (INPUT_INITIAL_CENTROIDS == "")
        fprintf(stdout,"    Time of initializing centroids:           %f ms\n", Tms_init);
    if (OnGPUFlag)
        fprintf(stdout,"    Time of transposing centroid matrix:      %f ms\n", Tms_transpose);

    fprintf(stdout,"    Nb of iterations:                         %d\n", NbIters);
    fprintf(stdout,"    Time of ComputeAssign per iteration:      %f ms\n", Tms_compute_assign/NbIters);
    fprintf(stdout,"    Time of UpdateCentroids per iteration:    %f ms\n", Tms_update/NbIters);
    fprintf(stdout,"    Time of one iteration:                    %f ms\n", (Tms_compute_assign + Tms_update)/NbIters);
    if (OnGPUFlag) {
        fprintf(stdout,"    Time of computation on GPU:               %f s\n", (float) Ts_computation);
        fprintf(stdout,"    Time of transfers and computation on GPU: %f s\n", (float) Ts_transfer_computation);
    } else {
        fprintf(stdout,"    Time of computation on CPU:               %f s\n", (float) Ts_computation);
    }
    // fprintf(stdout,"    Time of writing clustering results:       %f s\n", (float) Ts_output);
    // fprintf(stdout,"    Total elapsed time of the app.:           %f s\n", (float) Ts_application);

    // To uncomment the following code when using the synthetic dataset specified in our paper
    // fprintf(stdout,"    Average numerical error of final calculated centroids: %lf\n", NumError);

    fflush(stdout);
}
