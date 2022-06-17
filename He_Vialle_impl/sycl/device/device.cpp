#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng.hpp>

#include "../main.hpp"
#include "device.hpp"
#include "../init/init.hpp"

/*-----------------------------------------------------------------------------------------*/
/* Define dynamic arrays and variables of GPU                                              */
/*-----------------------------------------------------------------------------------------*/
T_real *GPU_dataT;       // Array for the transposed matrix of data instances
T_real *GPU_centroid;    // Array for the matrix of centroids
T_real *GPU_centroidT;   // Array for the transposed matrix of centroids
T_real *GPU_package;     // Array for the packages used in UpdateCentroids
int *GPU_label;          // Array for cluster labels of data points
int *GPU_count;          // Count of data instances in each cluster
unsigned long long int* track;            // Number of points changing label between two iterations
unsigned long long int* GPU_track_sum; // Sum of label changes in two consecutive iterations

sycl::queue dqueue;
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

	sycl::queue dqueue{selector};
	std::cout << "Running on " << dqueue.get_device().get_info<sycl::info::device::name>() << std::endl;
    return dqueue;
}


void device_sync(){
    dqueue.wait();
}


/*-----------------------------------------------------------------------------------------*/
/* Init and finalize the GPU device                                                        */
/*-----------------------------------------------------------------------------------------*/
void device_init(void) try {
    dqueue = get_queue();

    // Allocate memory space for the dynamic arrays
    GPU_dataT     = (T_real *)sycl::malloc_device(sizeof(T_real) * NbDims * NbPoints, dqueue);
    GPU_centroid  = (T_real *)sycl::malloc_device(sizeof(T_real) * NbClusters * NbDims, dqueue);
    GPU_centroidT = (T_real *)sycl::malloc_device(sizeof(T_real) * NbDims * NbClusters, dqueue);
    GPU_package   = (T_real *)sycl::malloc_device(sizeof(T_real) * NbDims * NbClusters * NbPackages, dqueue);
    GPU_label     = sycl::malloc_device<int>(NbPoints, dqueue);
    GPU_count     = sycl::malloc_device<int>(NbClusters, dqueue);
    GPU_track_sum = sycl::malloc_device<unsigned long long int>(1, dqueue);
    track         = sycl::malloc_host<unsigned long long int>(1, dqueue);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


void device_finish(void) try {
    // Free dynamic allocations
    sycl::free(GPU_dataT, dqueue);
    sycl::free(GPU_centroid, dqueue);
    sycl::free(GPU_centroidT, dqueue);
    sycl::free(GPU_package, dqueue);
    sycl::free(GPU_label, dqueue);
    sycl::free(GPU_count, dqueue);
    sycl::free(GPU_track_sum, dqueue);
    sycl::free(track, dqueue);
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
    dqueue.memcpy(GPU_dataT, dataT, sizeof(T_real) * NbDims * NbPoints);
    if (strcmp(INPUT_INITIAL_CENTROIDS, "") != 0)
        dqueue.memcpy(GPU_centroid, centroid, sizeof(T_real) * NbClusters * NbDims);
    
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
    dqueue.memcpy(label, GPU_label, sizeof(int) * NbPoints);
    dqueue.memcpy(centroid, GPU_centroid, sizeof(T_real) * NbClusters * NbDims);
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

void InitializeCentroids(T_real* GPU_centroidT, T_real* GPU_dataT, sycl::range<3> Db, sycl::range<3> Dg)
{
    int* r = sycl::malloc_shared<int>(NbPoints, dqueue);

    std::random_device random_device;
    std::mt19937 random_engine(random_device());
    std::uniform_int_distribution<int> distribution(0, NbPoints-1);

    for(int i{0}; i < NbPoints; i++)
        r[i] = distribution(random_engine);

    dqueue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item) {
                int col = item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);

                if (col < NbClusters) {
                    int idx = r[col];
                    for (int j = 0; j < NbDims; j++)
                        GPU_centroidT[j*NbClusters + col] = GPU_dataT[j*NbPoints + idx];
                }                 
            });
    }).wait();
    sycl::free(r, dqueue);
}


/*-----------------------------------------------------------------------------------------*/
/* Compute point-centroid distances and assign each point to a cluter                      */
/*-----------------------------------------------------------------------------------------*/
void ComputeAssign(T_real *GPU_dataT, T_real *GPU_centroid, int *GPU_label, unsigned long long int *GPU_track_sum,
    sycl::nd_item<3> item, local_ptr<unsigned long long int> shTrack)
{
    int local_idx = item.get_local_id(2);
    int idx = item.get_group(2) * BSXN + item.get_local_id(2);

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
    #endif

    #if BSXN > 256
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 256)
            shTrack[local_idx] += shTrack[local_idx + 256];
    #endif

    #if BSXN > 128
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 128)
            shTrack[local_idx] += shTrack[local_idx + 128];
    #endif

    #if BSXN > 64
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 64)
            shTrack[local_idx] += shTrack[local_idx + 64];
    #endif

    #if BSXN > 32
        item.barrier(sycl::access::fence_space::local_space);
        if (local_idx < 32)
            shTrack[local_idx] += shTrack[local_idx + 32];
    #endif

    #if BSXN > 16
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 16)
            shTrack[local_idx] += shTrack[local_idx + 16];
    #endif

    #if BSXN > 8
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 8)
            shTrack[local_idx] += shTrack[local_idx + 8];
    #endif

    #if BSXN > 4
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 4)
            shTrack[local_idx] += shTrack[local_idx + 4];
    #endif

    #if BSXN > 2
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 2)
            shTrack[local_idx] += shTrack[local_idx + 2];
    #endif

    #if BSXN > 1
        item.barrier(sycl::access::fence_space::local_space); // avoid races between threads within the same warp
        if (local_idx < 1)
            shTrack[local_idx] += shTrack[local_idx + 1];
    #endif

    // 2 - Final reduction into a global array
    if(local_idx == 0)
        sycl::atomic<unsigned long long int>(sycl::global_ptr<unsigned long long int>(&GPU_track_sum[0])).fetch_add(shTrack[0]);
}


/*-----------------------------------------------------------------------------------------*/
/* Update centroids												                           */
/*-----------------------------------------------------------------------------------------*/
void UpdateCentroids_Step1_Child_Load(int offset, int length, int* GPU_label,
    T_real* GPU_dataT, T_real shTabV[BSYD][BSXP], int* shTabL, 
    sycl::group<3> grp, sycl::h_item<3> item, int block_id_x=-1)
{
    // Index initialization
    block_id_x = (block_id_x == -1) ? grp.get_id(1) : block_id_x;
    int baseRow = grp.get_id(2) * BSYD;   // Base row of the block
    int row     = baseRow + item.get_local_id(1); // Row of child thread
    int baseCol = block_id_x * BSXP + offset;    // Base column of the block
    int col     = baseCol + item.get_local_id(0); // Column of child thread

    // Load the values and cluster labels of instances into sh mem tables
    if (col < (offset + length) && row < NbDims) {
        shTabV[item.get_local_id(1)][item.get_local_id(0)] = GPU_dataT[row * NbPoints + col];
        if (item.get_local_id(1) == 0)
            shTabL[item.get_local_id(0)] = GPU_label[col];
    }
}


void UpdateCentroids_Step1_Child(int pid, int offset, int length, T_real* GPU_package, 
    int* GPU_count, T_real shTabV[BSYD][BSXP], int* shTabL, 
    sycl::group<3> grp, sycl::h_item<3> item, int block_id_x=-1)
{
    // Index initialization
    block_id_x = (block_id_x == -1) ? grp.get_id(1) : block_id_x;
    int baseRow = grp.get_id(2) * BSYD;   // Base row of the block
    int baseCol = block_id_x * BSXP + offset;    // Base column of the block
    int cltIdx  = item.get_local_id(1) * BSXP + item.get_local_id(0); // 1D cluster index

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
            if (grp.get_id(2) == 0)
                sycl::atomic<int>(sycl::global_ptr<int>(&GPU_count[cltIdx])).fetch_add(count);

            int dmax = (grp.get_id(2) == NbDims / BSYD ? NbDims % BSYD : BSYD);
            for (int j = 0; j < dmax; j++)  // BlND_max: nb of dims managed by blk
                dpct::atomic_fetch_add(&GPU_package[(baseRow + j) * NbClusters * NbPackages + NbClusters * pid + cltIdx], Sv[j]);
        }
    } 
}


void UpdateCentroids_Step1_Parent(int *GPU_label, T_real *GPU_package, T_real *GPU_dataT, int *GPU_count,
    sycl::queue dqueue)
{
    size_t dim2 = (NbPoints / NbPackages) / BSXP;
    size_t dim3 = NbDims / BSYD + (NbDims % BSYD > 0 ? 1 : 0);
    sycl::range<3> num_groups{nStreams1, dim2, dim3};

    dqueue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for_work_group(num_groups, [=](sycl::group<3> grp) {
            int tid = grp.get_id(0); // Thread id

            if (tid < NbPackages) {
                int offset, length, quotient, remainder;
                int np = NbPackages / nStreams1 + (NbPackages % nStreams1 > 0 ? 1 : 0);  // Nb of packages for each stream
                int pid;                        // Id of package

                quotient  = NbPoints / NbPackages;
                remainder = NbPoints % NbPackages;

                for (int i = 0; i < np; i++) {
                    pid = i*nStreams1 + tid;     // Calculate the id of package

                    if (pid < NbPackages) {
                        offset = (pid < remainder ? ((quotient + 1) * pid) : (quotient * pid + remainder));
                        length = (pid < remainder ? (quotient + 1) : quotient);
                        sycl::range<3> group_size{BSXP, BSYD, 1};
                        T_real shTabV[BSYD][BSXP];
                        int shTabL[BSXP];

                        grp.parallel_for_work_item(group_size, [&](sycl::h_item<3> it) {
                            UpdateCentroids_Step1_Child_Load(
                                offset, length, GPU_label, GPU_dataT, shTabV, shTabL, grp, it);
                        });
                        // Implicit barrier
                        grp.parallel_for_work_item(group_size, [&](sycl::h_item<3> it) {
                            UpdateCentroids_Step1_Child(
                                pid, offset, length, GPU_package, GPU_count, shTabV, shTabL, grp, it);
                        });

                        if (grp.get_id(1) == (dim2-1) && (length % BSXP) > 0) {
                            grp.parallel_for_work_item(group_size, [&](sycl::h_item<3> it) {
                                UpdateCentroids_Step1_Child_Load(
                                    offset, length, GPU_label, GPU_dataT, shTabV, shTabL, grp, it, dim2);
                            });
                            // Implicit barrier
                            grp.parallel_for_work_item(group_size, [&](sycl::h_item<3> it) {
                                UpdateCentroids_Step1_Child(
                                    pid, offset, length, GPU_package, GPU_count, shTabV, shTabL, grp, it, dim2);
                            });
                        }
                    }
                }
            }         
        });
    });
}


void UpdateCentroids_Step2_Child(int pid, T_real *GPU_package, T_real *GPU_centroidT, int *GPU_count,
    sycl::h_item<3> item, sycl::group<3> grp)
{
    int rowC = grp.get_id(2); // Row of child thread
    int colC = grp.get_id(1) * BSXK + item.get_local_id(0); // Col of child thread

    if (colC < NbClusters && rowC < NbDims && GPU_count[colC] != 0)
        dpct::atomic_fetch_add(&GPU_centroidT[rowC * NbClusters + colC], 
            GPU_package[rowC * NbClusters * NbPackages + NbClusters * pid + colC] / GPU_count[colC]);
}


void UpdateCentroids_Step2_Parent(T_real *GPU_package, T_real *GPU_centroidT, int *GPU_count, 
    sycl::queue dqueue)
{
    size_t dim2 = NbClusters / BSXK + (NbClusters % BSXK > 0 ? 1 : 0);
    sycl::range<3> num_groups{nStreams2, dim2, NbDims};

    dqueue.submit([&](sycl::handler &h) {
        h.parallel_for_work_group(num_groups, [=](sycl::group<3> grp) {
            int tid = grp.get_id(0);

            if (tid < NbPackages) {
                // Nb of packages for each stream
                int np = NbPackages/nStreams2 + (NbPackages % nStreams2 > 0 ? 1 : 0);
                int pid;   // package ID
                sycl::range<3> group_size{BSXK, 1, 1};

                for (int i = 0; i < np; i++) {
                    pid = i*nStreams2 + tid;   // Calculate the id of package
                    if (pid < NbPackages)
                        grp.parallel_for_work_item(group_size, [&](h_item<3> it) {
                            UpdateCentroids_Step2_Child(pid, GPU_package, GPU_centroidT, GPU_count, it, grp);
                        });
                }
            }     
        });
    });
}


/*-----------------------------------------------------------------------------------------*/
/* K-means clustering on the GPU                                                           */
/*-----------------------------------------------------------------------------------------*/
void run_k_means(void) try {
    sycl::range<3> Dg(1, 1, 1), Db(1, 1, 1);
    double tolerance = 0.0;
    float elapsed;

    // Reset global variables to zeros
    NbIters = 0;              
    Tms_init = 0.0f;
    Tms_transpose = 0.0f;
    Tms_compute_assign = 0.0f;
    Tms_update = 0.0f;

    if (strcmp(INPUT_INITIAL_CENTROIDS, "") != 0) {

        // Get GPU_centroidT by transposing GPU_centroid
        start = std::chrono::steady_clock::now();
        transpose<NbDims>(GPU_centroidT, GPU_centroid, NbClusters, NbDims);
        device_sync();
        stop = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop - start).count();
        Tms_transpose += elapsed;
    }
    else {
        // Select initial centroids randomly
        Db[2] = BSXK;
        Db[1] = 1;
        Db[0] = 1;
        Dg[2] = NbClusters / Db[2] + (NbClusters % Db[2] > 0 ? 1 : 0);
        Dg[1] = 1;
        Dg[0] = 1;

        start = std::chrono::steady_clock::now();
        InitializeCentroids(GPU_centroidT, GPU_dataT, Db, Dg);
        stop = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop - start).count();
        Tms_init += elapsed;

        start = std::chrono::steady_clock::now();
        transpose<NbClusters>(GPU_centroid, GPU_centroidT, NbDims, NbClusters);
        device_sync();
        stop = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop - start).count();
        Tms_transpose += elapsed;
    }

    do {
        // Compute point-centroid distances & Assign each point to its nearest centroid
        Db[2] = BSXN;
        Db[1] = 1;
        Db[0] = 1;
        Dg[2] = NbPoints / Db[2] + (NbPoints % Db[2] > 0 ? 1 : 0);
        Dg[1] = 1;
        Dg[0] = 1;

        start = std::chrono::steady_clock::now();
        dqueue.memset(GPU_track_sum, 0, sizeof(unsigned long long int) * 1);
        device_sync();

        dqueue.submit([&](sycl::handler &cgh) {
            auto GPU_dataT_d = GPU_dataT;
            auto GPU_centroid_d = GPU_centroid;
            auto GPU_label_d = GPU_label;
            auto GPU_track_sum_d = GPU_track_sum;
            sycl::accessor<unsigned long long int, 1, sycl::access_mode::read_write, sycl::access::target::local> shTrack(sycl::range<1>(BSXN), cgh);
            cgh.parallel_for(sycl::nd_range<3>(Dg * Db, Db), [=](sycl::nd_item<3> item) {
                ComputeAssign(GPU_dataT_d, GPU_centroid_d, GPU_label_d, GPU_track_sum_d, item, shTrack.get_pointer());
            });
        });
        device_sync();
        dqueue.memcpy(track, GPU_track_sum, sizeof(unsigned long long int));
        device_sync();

        stop = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop - start).count();
        Tms_compute_assign += elapsed;

        // Update centroids - Step1
        start = std::chrono::steady_clock::now();
        dqueue.memset(GPU_count, 0, sizeof(int) * NbClusters);
        dqueue.memset(GPU_package, 0, sizeof(T_real) * NbDims * NbClusters * NbPackages);
        dqueue.memset(GPU_centroidT, 0, sizeof(T_real) * NbDims * NbClusters);
        device_sync();

        UpdateCentroids_Step1_Parent(GPU_label, GPU_package, GPU_dataT, GPU_count, dqueue);
        device_sync();

        stop = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop - start).count();
        Tms_update += elapsed;

        start = std::chrono::steady_clock::now();
        UpdateCentroids_Step2_Parent(GPU_package, GPU_centroidT, GPU_count, dqueue);
        device_sync();
        transpose<NbClusters>(GPU_centroid, GPU_centroidT, NbDims, NbClusters);
        device_sync();

        stop = std::chrono::steady_clock::now();
        elapsed = std::chrono::duration<float, std::milli>(stop - start).count();
        Tms_update += elapsed;

        // Calculate the variables for checking stopping criteria
        NbIters++;   // Count the number of iterations
        tolerance = (double)track[0] / NbPoints;     
        //printf("Track = %llu  Tolerance = %lf\n", track, tolerance); 
    //} while (tolerance > TOL && NbIters < MaxNbIters);
    } while (NbIters < MaxNbIters);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

template <int unroll_factor>
void transpose(T_real *odata, const T_real *idata, size_t m , size_t n) {
    // size_t tile_dim = (T_TILE_DIM < m) ? T_TILE_DIM : m;
    // size_t block_rows = (T_BLOCK_ROWS < n) ? T_BLOCK_ROWS : n;
    // size_t dim_grid_y = (T_TILE_DIM < n) ? n/tile_dim : 1;
 
    // sycl::range<2> dimGrid(m/tile_dim, dim_grid_y), dimBlock(tile_dim, block_rows);

    // dqueue.submit([&](sycl::handler &cgh) {
    //     sycl::accessor<T_real, 2, sycl::access_mode::read_write, sycl::access::target::local> tile(sycl::range<2>(tile_dim, tile_dim), cgh);
    //     cgh.parallel_for(sycl::nd_range<2>(dimGrid * dimBlock, dimBlock), [=](sycl::nd_item<2> item) {

    //         int x = item.get_group(0) * tile_dim + item.get_local_id(0);
    //         int y = item.get_group(1) * tile_dim + item.get_local_id(1);
    //         int width = item.get_group_range(0) * tile_dim;

    //         for (int j = 0; j < tile_dim; j += block_rows)
    //             tile[item.get_local_id(1)+j][item.get_local_id(0)] = idata[(y+j)*width + x];

    //         item.barrier(sycl::access::fence_space::local_space);

    //         x = item.get_group(1) * tile_dim + item.get_local_id(0);  // transpose block offset
    //         y = item.get_group(0) * tile_dim + item.get_local_id(1);

    //         for (int j = 0; j < tile_dim; j += block_rows)
    //             odata[(y+j)*width + x] = tile[item.get_local_id(0)][item.get_local_id(1) + j];        
    //     });
    // });
    dqueue.submit([&](handler &h) {
        h.single_task<Trans<unroll_factor>>([=]()[[intel::kernel_args_restrict]] {
            const int size = m*n;
            //#pragma unroll unroll_factor1
            for(int i{0}; i < m; i++) {
                #pragma unroll unroll_factor
                for(int j{0}; j < n; j++)
                    odata[j*m + i] = idata[i*n + j];
            }
            // for(int i{0}; i < m*n; i++)
            //     odata[(m*i) - (size-1) * (int)(i/size)] = idata[i];
        });
  });
}