#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>
#include <oneapi/mkl/rng/device.hpp>

#include "../main.hpp"
#include "device.hpp"
#include "../init/init.hpp"
#include <chrono>

/*-----------------------------------------------------------------------------------------*/
/* Define dynamic arrays and variables of GPU                                              */
/*-----------------------------------------------------------------------------------------*/
T_real *GPU_dataT;       // Array for the transposed matrix of data instances
T_real *GPU_centroid;    // Array for the matrix of centroids
T_real *GPU_centroidT;   // Array for the transposed matrix of centroids
T_real *GPU_package;     // Array for the packages used in UpdateCentroids
int *GPU_label;          // Array for cluster labels of data points
int *GPU_count;          // Count of data instances in each cluster
unsigned long long int* GPU_track_sum; // Sum of label changes in two consecutive iterations
/*
DPCT1032:0: A different random number generator is used. You may need to adjust
the code.
*/
oneapi::mkl::rng::device::philox4x32x10<1> *devStates; // States for using cuRAND library
sycl::queue queue;
std::chrono::time_point<std::chrono::steady_clock> start, stop;


sycl::queue get_queue() {
#if defined(INTEL_GPU_DEVICE)
	IntelGpuSelector selector{};
#elif defined(NVIDIA_DEVICE)
	CudaGpuSelector selector{};
#elif defined(CPU_DEVICE)	
	cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue queue{selector};
	std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;
    return queue;
}


void device_sync(){
    queue.wait();
}


/*-----------------------------------------------------------------------------------------*/
/* Init and finalize the GPU device                                                        */
/*-----------------------------------------------------------------------------------------*/
void device_init(void) try {
    queue = get_queue();

    // Allocate memory space for the dynamic arrays
    GPU_dataT     = (T_real *)sycl::malloc_device(sizeof(T_real) * NbDims * NbPoints, queue);
    GPU_centroid  = (T_real *)sycl::malloc_device(sizeof(T_real) * NbClusters * NbDims, queue);
    GPU_centroidT = (T_real *)sycl::malloc_device(sizeof(T_real) * NbDims * NbClusters, queue);
    GPU_package   = (T_real *)sycl::malloc_device(sizeof(T_real) * NbDims * NbClusters * NbPackages, queue);
    GPU_label     = sycl::malloc_device<int>(NbPoints, queue);
    GPU_count     = sycl::malloc_device<int>(NbClusters, queue);
    devStates     = sycl::malloc_device<oneapi::mkl::rng::device::philox4x32x10<1>>(NbClusters, queue);
    GPU_track_sum = sycl::malloc_device<unsigned long long int>(1, queue);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


void device_finish(void) try {
    // Free dynamic allocations
    sycl::free(GPU_dataT, queue);
    sycl::free(GPU_centroid, queue);
    sycl::free(GPU_centroidT, queue);
    sycl::free(GPU_package, queue);
    sycl::free(GPU_label, queue);
    sycl::free(GPU_count, queue);
    sycl::free(devStates, queue);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


/*-----------------------------------------------------------------------------------------*/
/* Transfer of CPU input data into GPU symbols                                             */
/*-----------------------------------------------------------------------------------------*/
void set_data_device(void) try {
    queue.memcpy(GPU_dataT, dataT, sizeof(T_real) * NbDims * NbPoints);
    if (INPUT_INITIAL_CENTROIDS != "")
        queue.memcpy(GPU_centroid, centroid, sizeof(T_real) * NbClusters * NbDims);
    
    device_sync();
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


/*-----------------------------------------------------------------------------------------*/
/* Transfer of GPU results into CPU array                                                  */
/*-----------------------------------------------------------------------------------------*/
void get_result_host(void) try {
    queue.memcpy(label, GPU_label, sizeof(int) * NbPoints);
    queue.memcpy(centroid, GPU_centroid, sizeof(T_real) * NbClusters * NbDims);
    device_sync();
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


/*-----------------------------------------------------------------------------------------*/
/* Select initial centroids                                                                */
/*-----------------------------------------------------------------------------------------*/
void SetupcuRand(oneapi::mkl::rng::device::philox4x32x10<1> *state,
                 sycl::nd_item<3> item)
{
    int col = item.get_global_id(2);

    if (col < NbClusters)
        state[col] = oneapi::mkl::rng::device::philox4x32x10<1>(
            0, {0, static_cast<std::uint64_t>(col * 8)});
}


void InitializeCentroids(oneapi::mkl::rng::device::philox4x32x10<1> *state,
                         T_real *GPU_centroidT, T_real *GPU_dataT,
                         sycl::nd_item<3> item)
{
    int col = item.get_global_id(2);

    if (col < NbClusters) {
        oneapi::mkl::rng::device::uniform<float> distr_ct1;
        oneapi::mkl::rng::device::philox4x32x10<1> localState = state[col];
        int idx = (sycl::ceil(NbPoints * curand_uniform(&localState))) - 1; // Control idx in [0, NbPoints - 1]
        for (int j = 0; j < NbDims; j++)
            GPU_centroidT[j*NbClusters + col] = GPU_dataT[j*NbPoints + idx];
    }
}


/*-----------------------------------------------------------------------------------------*/
/* Compute point-centroid distances and assign each point to a cluter                      */
/*-----------------------------------------------------------------------------------------*/
void ComputeAssign(T_real *GPU_dataT, T_real *GPU_centroid, int *GPU_label, unsigned long long int *GPU_track_sum,
    sycl::nd_item<3> item, 
    sycl::accessor<unsigned long long int, 1, sycl::access_mode::read_write, sycl::access::target::local> shTrack)
{
    int local_idx = item.get_local_id(2);
    int idx = item.get_group(0) * BSXN + item.get_local_id(2);

    shTrack[local_idx] = 0;

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
            shTrack[local_idx] = 1;
            GPU_label[idx] = min;
        }
    }

    // Count the changes of label into "track": two-part reduction
    // 1 - Parallel reduction of 1D block shared array shTrack[*] into shTrack[0],
    //     kill useless threads step by step, only thread 0 survives at the end.
    #if BSXN > 512
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 512)
            shTrack[local_idx] += shTrack[local_idx + 512];
        else
            return;
    #endif

    #if BSXN > 256
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 256)
            shTrack[local_idx] += shTrack[local_idx + 256];
        else
            return;
    #endif

    #if BSXN > 128
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 128)
            shTrack[local_idx] += shTrack[local_idx + 128];
        else
            return;
    #endif

    #if BSXN > 64
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 64)
            shTrack[local_idx] += shTrack[local_idx + 64];
        else
            return;
    #endif

    #if BSXN > 32
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 32)
            shTrack[local_idx] += shTrack[local_idx + 32];
        else
            return;
    #endif

    #if BSXN > 16
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 16)
            shTrack[local_idx] += shTrack[local_idx + 16];
        else
            return;
    #endif

    #if BSXN > 8
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 8)
            shTrack[local_idx] += shTrack[local_idx + 8];
        else
            return;
    #endif

    #if BSXN > 4
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 4)
            shTrack[local_idx] += shTrack[local_idx + 4];
        else
            return;
    #endif

    #if BSXN > 2
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 2)
            shTrack[local_idx] += shTrack[local_idx + 2];
        else
            return;
    #endif

    #if BSXN > 1
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 1)
            shTrack[local_idx] += shTrack[local_idx + 1];
        else
            return;
    #endif

    // 2 - Final reduction into a global array
    if (shTrack[0] > 0)
        sycl::atomic<unsigned long long>(sycl::global_ptr<unsigned long long>(&GPU_track_sum[0])).fetch_add(shTrack[0]);
}


/*-----------------------------------------------------------------------------------------*/
/* Update centroids												                           */
/*-----------------------------------------------------------------------------------------*/
void UpdateCentroids_Step1_Child(int pid, int offset, int length, int *GPU_label, T_real *GPU_package, 
    T_real *GPU_dataT, int *GPU_count, sycl::nd_item<3> item,
    sycl::accessor<T_real, 2, sycl::access_mode::read_write, sycl::access::target::local> shTabV,
    int *shTabL)
{
    // Index initialization
    int baseRow = item.get_group(1) * BSYD;   // Base row of the block
    int row     = baseRow + item.get_local_id(1); // Row of child thread
    int baseCol = item.get_group(0) * BSXP + offset;    // Base column of the block
    int col     = baseCol + item.get_local_id(2); // Column of child thread
    int cltIdx  = item.get_local_id(1) * BSXP + item.get_local_id(2); // 1D cluster index

    // Load the values and cluster labels of instances into sh mem tables
    if (col < (offset + length) && row < NbDims) {
        shTabV[item.get_local_id(1)][item.get_local_id(2)] = GPU_dataT[row * NbPoints + col];
        if (item.get_local_id(1) == 0)
            shTabL[item.get_local_id(2)] = GPU_label[col];
    }

    item.barrier(sycl::access::fence_space::local_space); // Wait for all data loaded into the sh mem

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
            if (item.get_group(1) == 0)
                sycl::atomic<int>(sycl::global_ptr<int>(&GPU_count[cltIdx])).fetch_add(count);

            int dmax = (item.get_group(1) == NbDims / BSYD ? NbDims % BSYD : BSYD);
            for (int j = 0; j < dmax; j++)  // BlND_max: nb of dims managed by blk
                dpct::atomic_fetch_add(&GPU_package[(baseRow + j) * NbClusters * NbPackages + NbClusters * pid + cltIdx], Sv[j]);
        }
    } 
}


void UpdateCentroids_Step1_Parent(int *GPU_label, T_real *GPU_package, T_real *GPU_dataT, int *GPU_count,
    sycl::nd_item<3> item,
    sycl::accessor<T_real, 2, sycl::access_mode::read_write, sycl::access::target::local> shTabV,
    int *shTabL)
{
    dpct::device_ext &dev = queue.get_device();
    int tid = item.get_local_id(2); // Thread id

    if (tid < NbPackages) {
        int offset, length, quotient, remainder;
        int np = NbPackages/nStreams1 + (NbPackages%nStreams1 > 0 ? 1 : 0);  // Nb of packages for each stream
        int pid;                        // Id of package
        sycl::queue *stream;
        sycl::range<3> Dg(1, 1, 1), Db(1, 1, 1);

        stream = dev.create_queue();

        quotient  = NbPoints / NbPackages;
        remainder = NbPoints % NbPackages;

        Db[2] = BSXP;
        Db[1] = BSYD;
        Db[0] = 1;

        Dg[1] = NbDims / Db[1] + (NbDims % Db[1] > 0 ? 1 : 0);
        Dg[0] = 1;

        for (int i = 0; i < np; i++) {
            pid = i*nStreams1 + tid;     // Calculate the id of package
            if (pid < NbPackages) {
                offset = (pid < remainder ? ((quotient + 1) * pid) : (quotient * pid + remainder));
                length = (pid < remainder ? (quotient + 1) : quotient);
                Dg[2] = length / Db[2] + (length % Db[2] > 0 ? 1 : 0);
                // Launch a child kernel on a stream to process a package
                stream->submit([&](sycl::handler &cgh) {
                    sycl::accessor<T_real, 2, sycl::access_mode::read_write, sycl::access::target::local>
                        shTabV_acc(sycl::range<2>(BSYD, BSXP), cgh);

                    sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::access::target::local>
                        shTabL_acc(sycl::range<1>(BSXP), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item) {
                        UpdateCentroids_Step1_Child(
                            pid, offset, length, GPU_label,
                            GPU_package, GPU_dataT, GPU_count,
                            item, shTabV_acc,
                            shTabL_acc.get_pointer());
                    });
                });
            }
        }
        dev.destroy_queue(stream);
    }
}


void UpdateCentroids_Step2_Child(int pid, T_real *GPU_package, T_real *GPU_centroidT, int *GPU_count,
    sycl::nd_item<3> item)
{
    int rowC = item.get_group(1); // Row of child thread
    int colC = item.get_group(2) * BSXK + item.get_local_id(2); // Col of child thread

    if (colC < NbClusters && rowC < NbDims)
        if (GPU_count[colC] != 0)
            dpct::atomic_fetch_add(&GPU_centroidT[rowC * NbClusters + colC], 
                GPU_package[rowC * NbClusters * NbPackages + NbClusters * pid + colC] / GPU_count[colC]);
}


void UpdateCentroids_Step2_Parent(T_real *GPU_package, T_real *GPU_centroidT, int *GPU_count, 
    sycl::nd_item<3> item)
{
    dpct::device_ext &dev = queue.get_device();
    int tid = item.get_local_id(2);

    if (tid < NbPackages) {
        int np = NbPackages/nStreams2 + (NbPackages % nStreams2 > 0 ? 1 : 0); // Nb of packages for each stream
        int pid;   // Id of package
        sycl::queue *stream;
        sycl::range<3> Dg(1, 1, 1), Db(1, 1, 1);
        stream = dev.create_queue();

        Db[2] = BSXK;
        Db[1] = 1;
        Db[0] = 1;
        Dg[2] = NbClusters / BSXK + (NbClusters % BSXK > 0 ? 1 : 0);
        Dg[1] = NbDims;
        Dg[0] = 1;

        for (int i = 0; i < np; i++) {
            pid = i*nStreams2 + tid;   // Calculate the id of package
            if (pid < NbPackages)
                stream->parallel_for(sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item) {
                    UpdateCentroids_Step2_Child(pid, GPU_package, GPU_centroidT, GPU_count, item);
                });
        }
        dev.destroy_queue(stream);
    }
}


/*-----------------------------------------------------------------------------------------*/
/* K-means clustering on the GPU                                                           */
/*-----------------------------------------------------------------------------------------*/
void run_k_means(void) try {
    sycl::range<3> Dg(1, 1, 1), Db(1, 1, 1);
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
        /*
        DPCT1012:50: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:51: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS(
            0, "Record the beginning time of transposing GPU_centroid");
        /*
        DPCT1007:52: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        CHECK_CUBLAS_SUCCESS(
            CUBLAS_GEAM(queue, oneapi::mkl::transpose::trans,
                        oneapi::mkl::transpose::nontrans, NbClusters, NbDims,
                        &alpha, GPU_centroid, NbDims, &beta, NULL, NbClusters,
                        GPU_centroidT, NbClusters),
            "Use CUBLAS_GEAM to transpose GPU_centroid");
        /*
        DPCT1012:53: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:54: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS(
            0, "Record the ending time of transposing GPU_centroid");
        CHECK_CUDA_SUCCESS(0,
                           "Wait for completion of transposing GPU_centroid");
        /*
        DPCT1003:55: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((elapsed = std::chrono::duration<float, std::milli>(
                                          stop_ct1 - start_ct1)
                                          .count(),
                            0),
                           "Elapsed time of transposing GPU_centroid");
        Tms_transpose += elapsed;
    } else {
        // Select initial centroids randomly
        Db[2] = BSXK;
        Db[1] = 1;
        Db[0] = 1;
        Dg[2] = NbClusters / Db[2] + (NbClusters % Db[2] > 0 ? 1 : 0);
        Dg[1] = 1;
        Dg[0] = 1;
        /*
        DPCT1012:58: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:59: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS(0, "Record the beginning time of initialization");
        /*
        DPCT1049:56: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        stop = queue.submit([&](sycl::handler &cgh) {
            auto devStates_ct0 = devStates;

            cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db),
                             [=](sycl::nd_item<3> item) {
                                 SetupcuRand(devStates_ct0, item);
                             });
        });
        /*
        DPCT1049:57: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        stop = queue.submit([&](sycl::handler &cgh) {
            auto devStates_ct0 = devStates;
            auto GPU_centroidT_ct1 = GPU_centroidT;
            auto GPU_dataT_ct2 = GPU_dataT;

            cgh.parallel_for(
                sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item) {
                    InitializeCentroids(devStates_ct0, GPU_centroidT_ct1,
                                        GPU_dataT_ct2, item);
                });
        });
        /*
        DPCT1012:60: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:61: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop.wait();
        stop_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS(0, "Record the ending time of initialization");
        CHECK_CUDA_SUCCESS(0, "Wait for completion of initialization");
        /*
        DPCT1003:62: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((elapsed = std::chrono::duration<float, std::milli>(
                                          stop_ct1 - start_ct1)
                                          .count(),
                            0),
                           "Elapsed time of initialization");
        Tms_init += elapsed;

        // Get GPU_centroid by transposing GPU_centroidT
        alpha = 1.0f;
        beta = 0.0f;
        /*
        DPCT1012:63: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:64: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS(
            0, "Record the beginning time of transposing GPU_centroidT");
        /*
        DPCT1007:65: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        CHECK_CUBLAS_SUCCESS(
            CUBLAS_GEAM(queue, oneapi::mkl::transpose::trans,
                        oneapi::mkl::transpose::nontrans, NbDims, NbClusters,
                        &alpha, GPU_centroidT, NbClusters, &beta, NULL, NbDims,
                        GPU_centroid, NbDims),
            "Use CUBLAS_GEAM to transpose GPU_centroidT");
        /*
        DPCT1012:66: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:67: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        stop_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS(
            0, "Record the ending time of transposing GPU_centroidT");
        CHECK_CUDA_SUCCESS(0,
                           "Wait for completion of transposing GPU_centroidT");
        /*
        DPCT1003:68: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((elapsed = std::chrono::duration<float, std::milli>(
                                          stop_ct1 - start_ct1)
                                          .count(),
                            0),
                           "Elapsed time of transposing GPU_centroidT");
        Tms_transpose += elapsed;
    }

    // CHECK_CUDA_SUCCESS(cudaMemset(GPU_label, 0, sizeof(int)*NbPoints), "Reset GPU_label to zeros");

    do {
        // Compute point-centroid distances & Assign each point to its nearest centroid
        Db[2] = BSXN;
        Db[1] = 1;
        Db[0] = 1;
        Dg[2] = NbPoints / Db[2] + (NbPoints % Db[2] > 0 ? 1 : 0);
        Dg[1] = 1;
        Dg[0] = 1;
        /*
        DPCT1012:70: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:71: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS((start = queue.submit_barrier(), 0),
                           "Record the beginning time of ComputeAssign");
        /*
        DPCT1003:72: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((queue
                                .memset(GPU_track_sum, 0,
                                        sizeof(unsigned long long int) * 1)
                                .wait(),
                            0),
                           "Reset GPU_track_sum to zero");
        /*
        DPCT1049:69: The workgroup size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the workgroup size if needed.
        */
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<unsigned long long int, 1, sycl::access_mode::read_write, sycl::access::target::local> shTrack(sycl::range<1>(BSXN), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item) {
                    ComputeAssign(GPU_dataT GPU_centroid, GPU_label,
                        GPU_track_sum, item, shTrack.get_pointer());
                });
        });
        /*
        DPCT1003:73: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((queue
                                .memcpy(&track, GPU_track_sum,
                                        sizeof(unsigned long long int) * 1)
                                .wait(),
                            0),
                           "Transfer GPU_track_sum-->track");
        /*
        DPCT1012:74: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:75: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        dpct::get_current_device().queues_wait_and_throw();
        stop_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS((stop = queue.submit_barrier(), 0),
                           "Record the ending time of ComputeAssign");
        CHECK_CUDA_SUCCESS(0, "Wait for completion of ComputeAssign");
        /*
        DPCT1003:76: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((elapsed = std::chrono::duration<float, std::milli>(
                                          stop_ct1 - start_ct1)
                                          .count(),
                            0),
                           "Elapsed time of ComputeAssign");
        Tms_compute_assign += elapsed;

        // Update centroids - Step1
        /*
        DPCT1012:77: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:78: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS((start = queue.submit_barrier(), 0),
                           "Record the beginning time of Update Step1");
        /*
        DPCT1003:79: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS(
            (queue.memset(GPU_count, 0, sizeof(int) * NbClusters).wait(), 0),
            "Reset GPU_count to zeros");
        /*
        DPCT1003:80: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS(
            (queue
                 .memset(GPU_package, 0,
                         sizeof(T_real) * NbDims * NbClusters * NbPackages)
                 .wait(),
             0),
            "Reset GPU_package to zeros");
        /*
        DPCT1003:81: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS(
            (queue
                 .memset(GPU_centroidT, 0, sizeof(T_real) * NbDims * NbClusters)
                 .wait(),
             0),
            "Reset GPU_centroidT to zeros");
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<T_real, 2, sycl::access_mode::read_write,
                           sycl::access::target::local>
                shTabV_acc(sycl::range<2>(4 /*BSYD*/, 32 /*BSXP*/), cgh);
            sycl::accessor<int, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                shTabL_acc(sycl::range<1>(32 /*BSXP*/), cgh);

            auto GPU_label_ct0 = GPU_label;
            auto GPU_package_ct1 = GPU_package;
            auto GPU_dataT_ct2 = GPU_dataT;
            auto GPU_count_ct3 = GPU_count;

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nStreams1),
                                               sycl::range<3>(1, 1, nStreams1)),
                             [=](sycl::nd_item<3> item) {
                                 UpdateCentroids_Step1_Parent(
                                     GPU_label_ct0, GPU_package_ct1,
                                     GPU_dataT_ct2, GPU_count_ct3, item,
                                     shTabV_acc,
                                     shTabL_acc.get_pointer());
                             });
        });
        /*
        DPCT1012:82: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:83: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        dpct::get_current_device().queues_wait_and_throw();
        stop_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS((stop = queue.submit_barrier(), 0),
                           "Record the ending time of Update Step1");
        CHECK_CUDA_SUCCESS(0, "Wait for completion of Update Step1");
        /*
        DPCT1003:84: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((elapsed = std::chrono::duration<float, std::milli>(
                                          stop_ct1 - start_ct1)
                                          .count(),
                            0),
                           "Elapsed time of Update Step1");
        Tms_update += elapsed;
		
        // Update centroids - Step2
        alpha = 1.0f;
        beta = 0.0f;
        /*
        DPCT1012:85: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:86: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        start_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS((start = queue.submit_barrier(), 0),
                           "Record the beginning time of Update Step2");
        queue.submit([&](sycl::handler &cgh) {
            auto GPU_package_ct0 = GPU_package;
            auto GPU_centroidT_ct1 = GPU_centroidT;
            auto GPU_count_ct2 = GPU_count;

            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nStreams2),
                                               sycl::range<3>(1, 1, nStreams2)),
                             [=](sycl::nd_item<3> item) {
                                 UpdateCentroids_Step2_Parent(
                                     GPU_package_ct0, GPU_centroidT_ct1,
                                     GPU_count_ct2, item);
                             });
        });
        /*
        DPCT1007:87: Migration of this CUDA API is not supported by the Intel(R)
        DPC++ Compatibility Tool.
        */
        CHECK_CUBLAS_SUCCESS(
            CUBLAS_GEAM(queue, oneapi::mkl::transpose::trans,
                        oneapi::mkl::transpose::nontrans, NbDims, NbClusters,
                        &alpha, GPU_centroidT, NbClusters, &beta, NULL, NbDims,
                        GPU_centroid, NbDims),
            "Use CUBLAS_GEAM to transpose GPU_centroidT");
        /*
        DPCT1012:88: Detected kernel execution time measurement pattern and
        generated an initial code for time measurements in SYCL. You can change
        the way time is measured depending on your goals.
        */
        /*
        DPCT1024:89: The original code returned the error code that was further
        consumed by the program logic. This original code was replaced with 0.
        You may need to rewrite the program logic consuming the error code.
        */
        dpct::get_current_device().queues_wait_and_throw();
        stop_ct1 = std::chrono::steady_clock::now();
        CHECK_CUDA_SUCCESS((stop = queue.submit_barrier(), 0),
                           "Record the ending time of Update Step2");
        CHECK_CUDA_SUCCESS(0, "Wait for completion of Update Step2");
        /*
        DPCT1003:90: Migrated API does not return error code. (*, 0) is
        inserted. You may need to rewrite this code.
        */
        CHECK_CUDA_SUCCESS((elapsed = std::chrono::duration<float, std::milli>(
                                          stop_ct1 - start_ct1)
                                          .count(),
                            0),
                           "Elapsed time of Update Step2");
        Tms_update += elapsed;

        // Calculate the variables for checking stopping criteria
        NbIters++;   // Count the number of iterations
        tolerance = (double)track / NbPoints;     
        //printf("Track = %llu  Tolerance = %lf\n", track, tolerance); 
    } while (tolerance > TOL && NbIters < MaxNbIters);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
