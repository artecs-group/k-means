## CPU-GPU-kmeans
Optimized parallel implementations of the k-means clustering algorithm:
1. **on multi-core CPU with vector units**: thread parallelization using OpenMP, auto-vectorization using AVX units
2. **on NVIDIA GPU**: using shared memory, dynamic parallelism, and multiple streams

In particular, for both implementations we use a two-step summation method with package processing to handle the effect of rounding errors that may occur during the phase of updating cluster centroids.

## Makefile Configuration
- By commenting the `-DDP` option or not, our code supports computations either in single or double precision, respectively.
- The choices for the `-march` and `--gpu-architecture` options should be updated according to your own CPU and GPU devices, respectively.
- If necessary, update the CUDA path according to your own situation.

## "main.h" Configuration
The configuration for benchmark dataset, block size, etc., are adjustable in the _main.h_ file.

Our k-means code does NOT generate any synthetic data, so your need to give the path and filename of your benchmark dataset in the `INPUT_DATA` constant, and also specifiy the `NbPoints`, `NbDims`, `NbClusters`. 

Optionally, if you want to impose initial centroids, you need to provide a text file and specifiy the corresponding path and filename in the `INPUT_INITIAL_CENTROIDS` constant. Otherwise, the initial centroids will be selected uniformly at random.

## Benchmark Datasets
We tested our code on one synthetic dataset created by our own and two real-world datasets downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). Each of them contains millions of instances, hence is too large to be loaded here. Instead we provide the _Synthetic_Data_Generator.py_, and describe the filtering operations on real-world datasets.
- **Synthetic dataset** (our dataset). It contains 50 million instances uniformly distributed in 4 convex clusters. Each instance has 4 dimensions. Since the _Synthetic_Data_Generator.py_ uses the `random` function, the dataset generated each time will have different values but will always keep the same distribution.
- [**Household power consumption dataset**](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption) (UCI Machine Learning Repository). It contains 2,075,259 measurements of electric power consumption in a household over a period
of nearly 4 years. Each measurement has 9 attributes. We remove the measurements containing missing values and also remove the first 2
attributes that record the date and time of measurements. The remaining set that we use for evaluation contains 2,049,280 measurements
with 7 numerical attributes.
- [**US census 1990 dataset**](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)) (UCI Machine Learning Repository). It contains 2,458,285 instances with 68 categorical attributes. It is a simplified and discretized version of the USCensus1990raw dataset which contains one percent sample drawn from the full 1990 US census data.

## Execution
Before execution, recompile the code by entering the `make` command if any change has been made to the code.

Then you can run the executable file _kmeans_ with several arguments:

- `-t <GPU|CPU>`: run computations on target GPU or on target CPU (default: GPU)
- `-cpu-nt <int>`: number of OpenMP threads (default: 1)
- `-max-iters <int>`: maximal number of iterations (default: 200)
- `-tol <float>`: tolerance, i.e. convergence criterion (default: 1.0E-4)

Examples:
- k-means on CPU: 
```
./kmeans -t CPU -cpu-nt 20
```
- k-means on GPU: 
```
./kmeans
```

## Corresponding papers
The approaches and experiments are documented in the following papers.

He, G., Vialle, S., & Baboulin, M. (2021). Parallelization of the k-means algorithm in a spectral clustering chain on CPU-GPU platforms. In B. B. et al. (Ed.), Euro-par 2020: Parallel processing workshops (Vol. 12480, LNCS, pp. 135–147). Warsaw, Poland: Springer. Available from: https://link.springer.com/chapter/10.1007/978-3-030-71593-9_11

He, G, Vialle, S, Baboulin, M. Parallel and accurate k-means algorithm on CPU-GPU architectures for spectral clustering. Concurrency Computat Pract Exper. 2021;e6621. Available from: http://doi.org/10.1002/cpe.6621 

If you find any part of this project useful for your scientific research, please cite the papers mentioned above.

