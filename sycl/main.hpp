/*-----------------------------------------------------------------------------------------*/
/* Define constants                                                                        */
/*-----------------------------------------------------------------------------------------*/
// Benchmark dataset
#define NbPoints    100000      // Number of data instances
#define NbDims      2             // Number of dimensions
#define NbClusters  4             // Number of clusters
#define NbPackages  100           // Number of packages used for UpdateCentroids
#define INPUT_DATA                "../data/SyntheticDataset.txt"
#define INPUT_INITIAL_CENTROIDS   ""   // Optional

// Adjustable block size
// - For ComputeAssign kernel
#define BSXN        128           // BLOCK_SIZE_X related to NbPoints (BSXN has to be in [1, max_work_group_size] & be a power of 2)
// - For UpdateCentroids kernel
#define BSXP        32            // BLOCK_SIZE_X related to NbPoints devided by NbPackages (BSXP has to be in [1, max_work_group_size] & be a power of 2)
#define BSYD        4             // BLOCK_SIZE_Y related to NbDims (BSXP*BSYD has to be in [NbClusters, max_work_group_size])
#define BSXK        32            // BLOCK_SIZE_X related to NbClusters (BSXK has to be in [1, max_work_group_size] & be a power of 2)

// Nb of streams
#define nStreams1   16            // Number of streams for Update_step1
#define nStreams2   32            // Number of streams for Update_step2

// Traspose values
#define T_TILE_DIM   32
#define T_BLOCK_ROWS 8

// Default values
#define DEFAULT_ONGPUFLAG     1         // Default flag of computation on GPU
#define DEFAULT_NB_THREADS    1         // Default number of OpenMP threads on CPU
#define DEFAULT_MAX_NB_ITERS  200       // Default maximal number of iterations
#define DEFAULT_TOL           1.0E-4    // Default tolerance (convergence criterion)

// Output files
#define OUTPUT_LABELS              "Labels.txt"
#define OUTPUT_INITIAL_CENTROIDS   "InitialCentroids.txt"
#define OUTPUT_FINAL_CENTROIDS     "FinalCentroids.txt"
#define OUTPUT_COUNT_PER_CLUSTER   "CountPerCluster.txt"


/*-----------------------------------------------------------------------------------------*/
/* Floating point datatype and op                                                          */
/*-----------------------------------------------------------------------------------------*/
#ifdef DP
typedef double T_real;
#define T_REAL_PRINT  "%lf"
#define T_REAL_TEXT   "double"
#define CUBLAS_GEAM   cublasDgeam
#else
typedef float T_real;
#define T_REAL_PRINT  "%f"
#define T_REAL_TEXT   "single"
#define CUBLAS_GEAM   cublasSgeam
#endif


/*-----------------------------------------------------------------------------------------*/
/* Global functions                                                                        */
/*-----------------------------------------------------------------------------------------*/
int main(int argc, char *argv[]);
