#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>

void init(void);
void finish(void);
void set_data_device(void);
void get_result_host(void);
void run_k_means(void);
