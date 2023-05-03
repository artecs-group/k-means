# K-Means

<img alt="license" src="https://img.shields.io/github/license/mashape/apistatus.svg"/>

This repository contains a k-means implementation for CUDA and SYCL.

## Requirements
You have to intall the following dependencies:

* [CUDA Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive)
* [oneAPI 2022.3](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
* [Intel Clang/LLVM](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md) -> In order to run SYCL over Nvidia GPUs.
* [hipSYCL](https://github.com/OpenSYCL/OpenSYCL) -> Optional.

## Project Structure
The repository is ordered in this folders:

* [He_Vialle_impl](/He_Vialle_impl/): Has the original implementation, you can found it [here](https://gitlab-research.centralesupelec.fr/Stephane.Vialle/cpu-gpu-kmeans).
* [custom_impl](/custom_impl/): Has the CUDA and SYCL custom implementations based on the original one.
* [etc](/etc/): Has the scripts to automatically get the application times.
* [data](/data/): You have there all the data requiered to reproduce the experiments.

## Publications
* Youssef Faqir-Rhazoui and Carlos García (2023). "Exploring the Performance and Portability of the k-means Algorithm on SYCL Across CPU and GPU Architectures". The Journal of Supercomputing.
    * [Free available here](https://doi.org/10.21203/rs.3.rs-2402689/v1).

## Acknowledgements
This work has been supported by the EU (FEDER), the Spanish MINECO and CM under grants S2018/TCS-4423, PID2021-126576NB-I00 funded by MCIN/AEI/10.13039/501100011033 and by "ERDF A way of making Europe".
