#include <stdio.h>
#include <stdlib.h>
#include <cuda.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#include "main.h"
#include "gpu.h"
#include "init.h"


/*-----------------------------------------------------------------------------------------*/
/* Define dynamic arrays and variables of GPU                                              */
/*-----------------------------------------------------------------------------------------*/
T_real *GPU_dataT;       // Array for the transposed matrix of data instances
T_real *GPU_centroid;    // Array for the matrix of centroids
T_real *GPU_centroidT;   // Array for the transposed matrix of centroids
T_real *GPU_package;     // Array for the packages used in UpdateCentroids
int *GPU_label;          // Array for cluster labels of data points
int *GPU_count;          // Count of data instances in each cluster
__device__ unsigned long long int GPU_track_sum;  // Sum of label changes in two consecutive iterations
unsigned long long int *AdrGPU_track_sum = NULL;  // Address of GPU_track_sum
curandState *devStates;                           // States for using cuRAND library

cublasHandle_t cublasHandle;                      // Handle for cuBLAS library

cudaEvent_t start;                                // CUDA event used for time measurement
cudaEvent_t stop;


/*-----------------------------------------------------------------------------------------*/
/* Init and finalize the GPU device                                                        */
/*-----------------------------------------------------------------------------------------*/
void gpuInit(void)
{
    cuInit(0);

    // Allocate memory space for the dynamic arrays
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_dataT, sizeof(T_real)*NbDims*NbPoints), "Dynamic allocation for GPU_dataT");
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroid, sizeof(T_real)*NbClusters*NbDims), "Dynamic allocation for GPU_centroid");
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_centroidT, sizeof(T_real)*NbDims*NbClusters), "Dynamic allocation for GPU_centroidT");
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_package, sizeof(T_real)*NbDims*NbClusters*NbPackages), "Dynamic allocation for GPU_package");
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_label, sizeof(int)*NbPoints), "Dynamic allocation for GPU_label");
    CHECK_CUDA_SUCCESS(cudaMalloc((void**) &GPU_count, sizeof(int)*NbClusters), "Dynamic allocation for GPU_count");
	CHECK_CUDA_SUCCESS(cudaMalloc((void**) &devStates, sizeof(curandState)*NbClusters), "Dynamic allocation for devStates");
    CHECK_CUDA_SUCCESS(cudaGetSymbolAddress((void **) &AdrGPU_track_sum, GPU_track_sum), "Get the address of GPU_track_sum");

    // Turn CPU arrays dataT, centroid and label into "pinned" memory areas
    CHECK_CUDA_SUCCESS(cudaHostRegister(dataT, sizeof(T_real)*NbDims*NbPoints, cudaHostRegisterPortable), "Turn dataT into pinned memory");
    CHECK_CUDA_SUCCESS(cudaHostRegister(label, sizeof(int)*NbPoints, cudaHostRegisterPortable), "Turn label into pinned memory");
    CHECK_CUDA_SUCCESS(cudaHostRegister(centroid, sizeof(T_real)*NbClusters*NbDims, cudaHostRegisterPortable), "Turn centroid into pinned memory");
    CHECK_CUDA_SUCCESS(cudaHostRegister(&track, sizeof(unsigned long long int), cudaHostRegisterPortable), "Turn track into pinned memory");

    // Initialize CUBLAS lib usage
    CHECK_CUBLAS_SUCCESS(cublasCreate(&cublasHandle), "Init of the CUBLAS lib handle"); 

    // Create events
    CHECK_CUDA_SUCCESS(cudaEventCreateWithFlags(&start, cudaEventBlockingSync), "Create the event start");
    CHECK_CUDA_SUCCESS(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync), "Create the event stop");
}


void gpuFinalize(void)
{
    // Free dynamic allocations on GPU
    CHECK_CUDA_SUCCESS(cudaFree(GPU_dataT), "Free the dynamic allocation for GPU_dataT");
    CHECK_CUDA_SUCCESS(cudaFree(GPU_centroid), "Free the dynamic allocation for GPU_centroid");
    CHECK_CUDA_SUCCESS(cudaFree(GPU_centroidT), "Free the dynamic allocation for GPU_centroidT");
    CHECK_CUDA_SUCCESS(cudaFree(GPU_package), "Free the dynamic allocation for GPU_package");
    CHECK_CUDA_SUCCESS(cudaFree(GPU_label), "Free the dynamic allocation for GPU_label");
    CHECK_CUDA_SUCCESS(cudaFree(GPU_count), "Free the dynamic allocation for GPU_count");
    CHECK_CUDA_SUCCESS(cudaFree(devStates), "Free the dynamic allocation for devStates");

    // Turn "pinned" CPU arrays into std array
    CHECK_CUDA_SUCCESS(cudaHostUnregister(dataT), "Turn pinned dataT into standard array");
    CHECK_CUDA_SUCCESS(cudaHostUnregister(label), "Turn pinned label into standard array");
    CHECK_CUDA_SUCCESS(cudaHostUnregister(centroid), "Turn pinned centroid into standard array");
    CHECK_CUDA_SUCCESS(cudaHostUnregister(&track), "Turn pinned track into standard array");

    // Destroy events
    CHECK_CUDA_SUCCESS(cudaEventDestroy(start), "Destroy the event start");
    CHECK_CUDA_SUCCESS(cudaEventDestroy(stop), "Destroy the event stop");

    // Free CUBLAS lib usage
    CHECK_CUBLAS_SUCCESS(cublasDestroy(cublasHandle), "Free the CUBLAS lib");
}


/*-----------------------------------------------------------------------------------------*/
/* Transfer of CPU input data into GPU symbols                                             */
/*-----------------------------------------------------------------------------------------*/
void gpuSetDataOnGPU(void)
{
    CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_dataT, dataT, sizeof(T_real)*NbDims*NbPoints, cudaMemcpyHostToDevice),
                       "Transfer dataT --> GPU_dataT");
    if (INPUT_INITIAL_CENTROIDS != "") {
        CHECK_CUDA_SUCCESS(cudaMemcpy(GPU_centroid, centroid, sizeof(T_real)*NbClusters*NbDims, cudaMemcpyHostToDevice),
                       "Transfer centroid --> GPU_centroid");
    }
}


/*-----------------------------------------------------------------------------------------*/
/* Transfer of GPU results into CPU array                                                  */
/*-----------------------------------------------------------------------------------------*/
void gpuGetResultOnCPU(void)
{
    CHECK_CUDA_SUCCESS(cudaMemcpy(label, GPU_label, sizeof(int)*NbPoints, cudaMemcpyDeviceToHost),
                       "Transfer GPU_label-->label");
    CHECK_CUDA_SUCCESS(cudaMemcpy(centroid, GPU_centroid, sizeof(T_real)*NbClusters*NbDims, cudaMemcpyDeviceToHost),
                       "Transfer GPU_centroid-->centroid");
}


/*-----------------------------------------------------------------------------------------*/
/* Select initial centroids                                                                */
/*-----------------------------------------------------------------------------------------*/
__global__ void SetupcuRand(curandState *state)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col < NbClusters) 
        curand_init(0, col, 0, &state[col]);
}


__global__ void InitializeCentroids(curandState *state, T_real *GPU_centroidT, T_real *GPU_dataT)
{
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col < NbClusters) {
        curandState localState = state[col];
        int idx = (ceil(NbPoints * curand_uniform(&localState))) - 1;  // Control idx in [0, NbPoints - 1]
        for (int j = 0; j < NbDims; j++)
            GPU_centroidT[j*NbClusters + col] = GPU_dataT[j*NbPoints + idx];
    }
}


/*-----------------------------------------------------------------------------------------*/
/* Compute point-centroid distances and assign each point to a cluter                      */
/*-----------------------------------------------------------------------------------------*/
__global__ void ComputeAssign(T_real *GPU_dataT, T_real *GPU_centroid, int *GPU_label, unsigned long long int *AdrGPU_track_sum)
{
    int idx = blockIdx.x * BSXN + threadIdx.x;
    __shared__ unsigned long long int shTrack[BSXN];
    shTrack[threadIdx.x] = 0;

    if (idx < NbPoints) {
        int min = 0;
        T_real diff, dist_sq, minDist_sq;
        for (int k = 0; k < NbClusters; k++) {
            dist_sq = 0.0f;
            // Calculate the square of distance between instance idx and centroid k
            for(int j = 0; j < NbDims; j++) {
                diff = (GPU_dataT[j*NbPoints + idx] - GPU_centroid[k*NbDims + j]);
                dist_sq += (diff*diff);
            }
            // Find and record the nearest centroid to instance idx
            if (dist_sq < minDist_sq || k == 0) {
                minDist_sq = dist_sq;
                min = k;
            }
        }
        // Change the label if necessary
        if (GPU_label[idx] != min) {
            shTrack[threadIdx.x] = 1;
            GPU_label[idx] = min;
        }
    }

    // Count the changes of label into "track": two-part reduction
    // 1 - Parallel reduction of 1D block shared array shTrack[*] into shTrack[0],
    //     kill useless threads step by step, only thread 0 survives at the end.
    #if BSXN > 512
        __syncthreads();
        if (threadIdx.x < 512)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 512];
        else
            return;
    #endif

    #if BSXN > 256
        __syncthreads();
        if (threadIdx.x < 256)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 256];
        else
            return;
    #endif

    #if BSXN > 128
        __syncthreads();
        if (threadIdx.x < 128)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 128];
        else
            return;
    #endif

    #if BSXN > 64
        __syncthreads();
        if (threadIdx.x < 64)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 64];
        else
            return;
    #endif

    #if BSXN > 32
        __syncthreads();
        if (threadIdx.x < 32)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 32];
        else
            return;
    #endif

    #if BSXN > 16
        __syncwarp();          // avoid races between threads within the same warp
        if (threadIdx.x < 16)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 16];
        else
            return;
    #endif

    #if BSXN > 8
        __syncwarp();          // avoid races between threads within the same warp
        if (threadIdx.x < 8)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 8];
        else
            return;
    #endif

    #if BSXN > 4
        __syncwarp();          // avoid races between threads within the same warp
        if (threadIdx.x < 4)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 4];
        else
            return;
    #endif

    #if BSXN > 2
        __syncwarp();          // avoid races between threads within the same warp
        if (threadIdx.x < 2)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 2];
        else
            return;
    #endif

    #if BSXN > 1
        __syncwarp();          // avoid races between threads within the same warp
        if (threadIdx.x < 1)
            shTrack[threadIdx.x] += shTrack[threadIdx.x + 1];
        else
            return;
    #endif

    // 2 - Final reduction into a global array
    if (shTrack[0] > 0)
        atomicAdd(AdrGPU_track_sum, shTrack[0]);
}


/*-----------------------------------------------------------------------------------------*/
/* Update centroids												                           */
/*-----------------------------------------------------------------------------------------*/
__global__ void UpdateCentroids_Step1_Child(int pid, int offset, int length, int *GPU_label, T_real *GPU_package, T_real *GPU_dataT, int *GPU_count)
{
    __shared__ T_real shTabV[BSYD][BSXP];            // Tab of instance values
    __shared__ int shTabL[BSXP];                     // Tab of labels (cluster Id)
    // Index initialization
    int baseRow = blockIdx.y * BSYD;                 // Base row of the block
    int row = baseRow + threadIdx.y;                 // Row of child thread
    int baseCol = blockIdx.x * BSXP + offset;        // Base column of the block
    int col = baseCol + threadIdx.x;                 // Column of child thread
    int cltIdx = threadIdx.y * BSXP + threadIdx.x;   // 1D cluster index

    // Load the values and cluster labels of instances into sh mem tables
    if (col < (offset + length) && row < NbDims) {
        shTabV[threadIdx.y][threadIdx.x] = GPU_dataT[row*NbPoints + col];
        if (threadIdx.y == 0)
            shTabL[threadIdx.x] = GPU_label[col];
    }

    __syncthreads();                 // Wait for all data loaded into the sh mem

    // Compute partial evolution of centroid related to cluster number 'cltIdx'
    if (cltIdx < NbClusters) {             // Required condition: NbClusters <= BSXP*BSYD <= 1024
        #define BlND (NbDims < BSYD ? NbDims : BSYD) // BlND: nb of dims stored by block
        T_real Sv[BlND];             // Sum of values in BlND dimensions
        for (int j = 0; j < BlND; j++)
            Sv[j] = 0.0f;            // Init the tab Sv to zeros
        int count = 0;               // Init the counter of instances

        // - Accumulate contributions to cluster number 'cltIdx'
        for (int x = 0; x < BSXP && (baseCol + x) < (offset + length); x++) {
            if (shTabL[x] == cltIdx) { 
                count++;
                for (int y = 0; y < BSYD && (baseRow + y) < NbDims; y++)
                    Sv[y] += shTabV[y][x];
            }
        }

        // - Save the contribution of block into global contribution of the package
        if (count != 0) {
            if (blockIdx.y == 0)
                atomicAdd(&GPU_count[cltIdx], count);
            int dmax = (blockIdx.y == NbDims/BSYD ? NbDims%BSYD : BSYD);
            for (int j = 0; j < dmax; j++)  // BlND_max: nb of dims managed by blk
                atomicAdd(&GPU_package[(baseRow + j)*NbClusters*NbPackages + NbClusters*pid + cltIdx], Sv[j]);
        }
    } 
}


__global__ void UpdateCentroids_Step1_Parent(int *GPU_label, T_real *GPU_package, T_real *GPU_dataT, int *GPU_count)
{
    int tid = threadIdx.x;              // Thread id

    if (tid < NbPackages) {
        int offset, length, quotient, remainder;
        int np = NbPackages/nStreams1 + (NbPackages%nStreams1 > 0 ? 1 : 0);  // Nb of packages for each stream
        int pid;                        // Id of package
        cudaStream_t stream;
        dim3 Dg, Db;

        cudaStreamCreateWithFlags(&stream, cudaStreamDefault); 

	    quotient = NbPoints/NbPackages;
        remainder = NbPoints%NbPackages;

        Db.x = BSXP;
        Db.y = BSYD;
        Db.z = 1;

        Dg.y = NbDims/Db.y + (NbDims%Db.y > 0 ? 1 : 0);
        Dg.z = 1;

        for (int i = 0; i < np; i++) {
            pid = i*nStreams1 + tid;     // Calculate the id of package
            if (pid < NbPackages) {
                offset = (pid < remainder ? ((quotient + 1) * pid) : (quotient * pid + remainder));
                length = (pid < remainder ? (quotient + 1) : quotient);
                Dg.x = length/Db.x + (length%Db.x > 0 ? 1 : 0);
                // Launch a child kernel on a stream to process a package
                UpdateCentroids_Step1_Child<<<Dg,Db,0,stream>>>(pid, offset, length, GPU_label, GPU_package, GPU_dataT, GPU_count);
            }
        }
        cudaStreamDestroy(stream); 
    }
}


__global__ void UpdateCentroids_Step2_Child(int pid, T_real *GPU_package, T_real *GPU_centroidT, int *GPU_count)
{
    int rowC = blockIdx.y;                           // Row of child thread
    int colC = blockIdx.x * BSXK + threadIdx.x;      // Col of child thread

    if (colC < NbClusters && rowC < NbDims)
        if (GPU_count[colC] != 0)
            atomicAdd(&GPU_centroidT[rowC*NbClusters + colC], GPU_package[rowC*NbClusters*NbPackages + NbClusters*pid + colC] / GPU_count[colC]);
}


__global__ void UpdateCentroids_Step2_Parent(T_real *GPU_package, T_real *GPU_centroidT, int *GPU_count)
{
    int tid = threadIdx.x;

    if (tid < NbPackages) {
        int np = NbPackages/nStreams2 + (NbPackages%nStreams2 > 0 ? 1 : 0); // Nb of packages for each stream
        int pid;   // Id of package
        cudaStream_t stream;
        dim3 Dg, Db;

        cudaStreamCreateWithFlags(&stream, cudaStreamDefault); 

        Db.x = BSXK;
        Db.y = 1;
        Db.z = 1;
        Dg.x = NbClusters/BSXK + (NbClusters%BSXK > 0 ? 1 : 0);
        Dg.y = NbDims;
        Dg.z = 1;

        for (int i = 0; i < np; i++) {
            pid = i*nStreams2 + tid;   // Calculate the id of package
            if (pid < NbPackages) 
                UpdateCentroids_Step2_Child<<<Dg,Db,0,stream>>>(pid, GPU_package, GPU_centroidT, GPU_count);
        }
        cudaStreamDestroy(stream); 
    }
}


/*-----------------------------------------------------------------------------------------*/
/* K-means clustering on the GPU                                                           */
/*-----------------------------------------------------------------------------------------*/
void gpuKmeans(void)
{
    dim3 Dg, Db;
    double tolerance = 0.0;
    float elapsed;
    T_real alpha, beta;       // Parameters for CUBLAS_GEAM

    // Reset global variables to zeros
    NbIters = 0;              
    Tms_init = 0.0f;
    Tms_transpose = 0.0f;
    Tms_compute_assign = 0.0f;
    Tms_update = 0.0f;

    if (INPUT_INITIAL_CENTROIDS != "") {
        alpha = 1.0f;
        beta = 0.0f;
        // Get GPU_centroidT by transposing GPU_centroid
        CHECK_CUDA_SUCCESS(cudaEventRecord(start, 0), "Record the beginning time of transposing GPU_centroid");
        CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(cublasHandle, 
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             NbClusters, NbDims,
                             &alpha,
                             GPU_centroid, NbDims,
                             &beta,
                             NULL, NbClusters,
                             GPU_centroidT, NbClusters), "Use CUBLAS_GEAM to transpose GPU_centroid"); 
        CHECK_CUDA_SUCCESS(cudaEventRecord(stop, 0), "Record the ending time of transposing GPU_centroid");
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(stop), "Wait for completion of transposing GPU_centroid");
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, stop), "Elapsed time of transposing GPU_centroid");
        Tms_transpose += elapsed;
    } else {
        // Select initial centroids randomly
        Db.x = BSXK;
        Db.y = 1;
        Db.z = 1;
        Dg.x = NbClusters/Db.x + (NbClusters%Db.x > 0 ? 1 : 0);
        Dg.y = 1;
        Dg.z = 1;
        CHECK_CUDA_SUCCESS(cudaEventRecord(start, 0), "Record the beginning time of initialization");
        SetupcuRand<<<Dg,Db>>>(devStates);
        InitializeCentroids<<<Dg,Db>>>(devStates, GPU_centroidT, GPU_dataT);
        CHECK_CUDA_SUCCESS(cudaEventRecord(stop, 0), "Record the ending time of initialization");
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(stop), "Wait for completion of initialization");
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, stop), "Elapsed time of initialization");
        Tms_init += elapsed;

        // Get GPU_centroid by transposing GPU_centroidT
        alpha = 1.0f;
        beta = 0.0f;
        CHECK_CUDA_SUCCESS(cudaEventRecord(start, 0), "Record the beginning time of transposing GPU_centroidT");
        CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(cublasHandle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             NbDims, NbClusters,
                             &alpha,
                             GPU_centroidT, NbClusters,
                             &beta,
                             NULL, NbDims,
                             GPU_centroid, NbDims), "Use CUBLAS_GEAM to transpose GPU_centroidT");
        CHECK_CUDA_SUCCESS(cudaEventRecord(stop, 0), "Record the ending time of transposing GPU_centroidT");
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(stop), "Wait for completion of transposing GPU_centroidT");
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, stop), "Elapsed time of transposing GPU_centroidT");
        Tms_transpose += elapsed;
    }

    // CHECK_CUDA_SUCCESS(cudaMemset(GPU_label, 0, sizeof(int)*NbPoints), "Reset GPU_label to zeros");

    do {
        // Compute point-centroid distances & Assign each point to its nearest centroid
        Db.x = BSXN;
        Db.y = 1;
        Db.z = 1;
        Dg.x = NbPoints/Db.x + (NbPoints%Db.x > 0 ? 1 : 0);
        Dg.y = 1;
        Dg.z = 1;
        CHECK_CUDA_SUCCESS(cudaEventRecord(start, 0), "Record the beginning time of ComputeAssign");
        CHECK_CUDA_SUCCESS(cudaMemset(AdrGPU_track_sum, 0, sizeof(unsigned long long int)*1), "Reset GPU_track_sum to zero");
        ComputeAssign<<<Dg,Db>>>(GPU_dataT, GPU_centroid, GPU_label, AdrGPU_track_sum);
        CHECK_CUDA_SUCCESS(cudaMemcpy(&track, AdrGPU_track_sum, sizeof(unsigned long long int)*1, cudaMemcpyDeviceToHost),
                           "Transfer GPU_track_sum-->track");
        CHECK_CUDA_SUCCESS(cudaEventRecord(stop, 0), "Record the ending time of ComputeAssign");
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(stop), "Wait for completion of ComputeAssign");
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, stop), "Elapsed time of ComputeAssign");
        Tms_compute_assign += elapsed;

        // Update centroids - Step1
        CHECK_CUDA_SUCCESS(cudaEventRecord(start, 0), "Record the beginning time of Update Step1");
        CHECK_CUDA_SUCCESS(cudaMemset(GPU_count, 0, sizeof(int)*NbClusters), "Reset GPU_count to zeros");
        CHECK_CUDA_SUCCESS(cudaMemset(GPU_package, 0, sizeof(T_real)*NbDims*NbClusters*NbPackages), "Reset GPU_package to zeros");
        CHECK_CUDA_SUCCESS(cudaMemset(GPU_centroidT, 0, sizeof(T_real)*NbDims*NbClusters), "Reset GPU_centroidT to zeros");
        UpdateCentroids_Step1_Parent<<<1,nStreams1>>>(GPU_label, GPU_package, GPU_dataT, GPU_count);
        CHECK_CUDA_SUCCESS(cudaEventRecord(stop, 0), "Record the ending time of Update Step1");
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(stop), "Wait for completion of Update Step1");
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, stop), "Elapsed time of Update Step1");
        Tms_update += elapsed;
		
        // Update centroids - Step2
        alpha = 1.0f;
        beta = 0.0f;
        CHECK_CUDA_SUCCESS(cudaEventRecord(start, 0), "Record the beginning time of Update Step2");
        UpdateCentroids_Step2_Parent<<<1,nStreams2>>>(GPU_package, GPU_centroidT, GPU_count);
        CHECK_CUBLAS_SUCCESS(CUBLAS_GEAM(cublasHandle,
                             CUBLAS_OP_T,
                             CUBLAS_OP_N,
                             NbDims, NbClusters,
                             &alpha,
                             GPU_centroidT, NbClusters,
                             &beta,
                             NULL, NbDims,
                             GPU_centroid, NbDims), "Use CUBLAS_GEAM to transpose GPU_centroidT");
        CHECK_CUDA_SUCCESS(cudaEventRecord(stop, 0), "Record the ending time of Update Step2");
        CHECK_CUDA_SUCCESS(cudaEventSynchronize(stop), "Wait for completion of Update Step2");
        CHECK_CUDA_SUCCESS(cudaEventElapsedTime(&elapsed, start, stop), "Elapsed time of Update Step2");
        Tms_update += elapsed;

        // Calculate the variables for checking stopping criteria
        NbIters++;   // Count the number of iterations
        tolerance = (double)track / NbPoints;     
        //printf("Track = %llu  Tolerance = %lf\n", track, tolerance); 
    } while (NbIters < MaxNbIters);
}
