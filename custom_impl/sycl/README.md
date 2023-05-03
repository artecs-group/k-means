# Configuring the SYCL implementation
The code is built by a Makefile. The following table explains the main variables to build and run the code:

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| DEVICE   | Chooses the device where to run. | cpu, igpu(Intel GPU), ngpu(Nvidia GPU) | cpu |
| IMPL     | Implementation to use. | sycl-cpu, sycl-igpu, sycl-ngpu, sycl-por | sycl-cpu |
| SYCL     | SYCL implementation to use. | dpcpp, hipsycl | hipsycl |
| HIPSYCL_COMPILER | Compiler used for hipSYCL (only if SYCL=hipsycl was chosen) | clang, nvc++ | clang |

## Building and running
Here is an example to build and run the code:

```cpp
$: make DEVICE=ngpu IMPL=sycl-por SYCL=dpcpp
$: make run
```