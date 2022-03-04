CPUCC = g++
GPUCC = /usr/local/cuda/bin/nvcc   

# For RTX 2080 Ti
# CUDA_TARGET_FLAGS = --gpu-architecture=sm_75
# For GTX 1080
# CUDA_TARGET_FLAGS = --gpu-architecture=sm_61

CXXFLAGS = -O3 #-DDP
CXXFLAGS += -I/usr/local/cuda/include/
CC_CXXFLAGS = -Ofast -fopenmp -funroll-loops -march=skylake-avx512 #-march=native
CUDA_CXXFLAGS = $(CUDA_TARGET_FLAGS)

CC_LDFLAGS =  -fopenmp 
CUDA_LDFLAGS = -L/usr/local/cuda/lib64/ 

CC_LIBS = -lopenblas
CUDA_LIBS = -lcuda -lcublas -lcudart -lcudadevrt -lcurand

CC_SOURCES =  main.cc init.cc  
CUDA_SOURCES = gpu.cu 
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

EXECNAME = kmeans


all:
	$(CPUCC) -c $(CXXFLAGS) $(CC_CXXFLAGS) $(CC_SOURCES)
	$(GPUCC) -dc $(CXXFLAGS) $(CUDA_CXXFLAGS) $(CUDA_SOURCES)
	$(GPUCC) -dlink $(CXXFLAGS) $(CUDA_CXXFLAGS) $(CUDA_OBJECTS) -o __gpu_rdc.o -lcudadevrt
	$(CPUCC) -o $(EXECNAME) $(CC_LDFLAGS) $(CUDA_LDFLAGS) $(CUDA_OBJECTS) __gpu_rdc.o $(CC_OBJECTS) $(CUDA_LIBS) $(CC_LIBS)

clean:
	rm -f *.o $(EXECNAME) *.linkinfo *~ *.bak .depend


#Regles automatiques pour les objets
#%.o:  %.cc
#	$(CPUCC)  -c  $(CXXFLAGS) $(CC_CXXFLAGS) $<
#
#%.o:  %.cu
#	$(GPUCC)  -c  $(CXXFLAGS) $(CUDA_CXXFLAGS) $<

