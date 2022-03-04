#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_SUCCESS(exp, msg)                                           \
        { if ((exp) != 0) {                                                    \
            fprintf(stderr, "Error on CUDA operation (%s)\n", msg);            \
    exit(EXIT_FAILURE);}                                                       \
    }

#define CHECK_CUBLAS_SUCCESS(exp, msg)                                         \
        { int r = (exp); if (r != 0) {                                         \
            fprintf(stderr, "Error (%d) on CUBLAS operation (%s)\n", r, msg);  \
    exit(EXIT_FAILURE);}                                                       \
    }

void gpuInit(void);
void gpuFinalize(void);
void gpuSetDataOnGPU(void);
void gpuGetResultOnCPU(void);
void gpuKmeans(void);
