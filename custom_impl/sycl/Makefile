DEVICE = cpu # {cpu, ngpu, igpu}
DEBUG = no # {yes, no}
IMPL = sycl-cpu # {sycl-cpu, sycl-ngpu, sycl-igpu, sycl-com}
SYCL = dpcpp # {dpcpp, hipsycl}: hipsycl does not support igpu
HIPSYCL_COMPILER = clang # {clang, nvc++}

# device
CPU = CPU_DEVICE
IGPU = INTEL_GPU_DEVICE
NVIDIA = NVIDIA_DEVICE

PDEVICE = $(CPU)

ifeq ($(DEVICE),ngpu)
  PDEVICE = $(NVIDIA)
else ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
endif

# implementation
SCPU = SYCL_CPU
SNGPU = SYCL_NGPU
SIGPU = SYCL_IGPU
SCOM = SYCL_COMMON

PIMPL = $(SYCL_CPU)

ifeq ($(IMPL),sycl-ngpu)
  PIMPL = $(SNGPU)
else ifeq ($(IMPL), sycl-igpu)
  PIMPL = $(SIGPU)
else ifeq ($(IMPL), sycl-com)
  PIMPL = $(SCOM)
endif

ifeq ($(SYCL), dpcpp)
	CC = icpx
	CFLAGS = -O3 -D$(PDEVICE) -D$(PIMPL) -DDPCPP -fsycl
	CLINK = -fsycl-device-code-split=per_kernel

	ifeq ($(PDEVICE), $(NVIDIA))
		CC = clang++
		CFLAGS += -w -fsycl-targets=nvptx64-nvidia-cuda
	endif
else
	ifeq ($(PDEVICE), $(IGPU))
		$(error 'DEVICE=igpu' option is not allowed in conjunction to 'SYCL=hipsycl'.)
	endif
	CC = syclcc
	CFLAGS = -O3 -D$(PDEVICE) -D$(PIMPL) -DHIPSYCL
	ifeq ($(PDEVICE),$(NVIDIA))
		ifeq ($(HIPSYCL_COMPILER),nvc++)
			CFLAGS += --hipsycl-targets="cuda-nvcxx:cc61"
		else
			CFLAGS += --hipsycl-targets="cuda:sm_61" -I/usr/local/cuda/include
			CLINK = -L/usr/local/cuda/include
		endif
	else
		CFLAGS += --hipsycl-targets="omp" --hipsycl-use-accelerated-cpu -w
	endif  
endif

ifeq ($(DEBUG),yes)
	CFLAGS += -g
endif

k_means: ./device/device.o main.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./device/device.o -o k_means

main.o: ./device/device.o
	$(CC) $(CFLAGS) main.cpp -c -o main.o

./device/device.o: ./device/device.hpp
	$(CC) $(CFLAGS) ./device/device.cpp -c -o ./device/device.o

run: k_means
	./k_means ../../data/USCensus1990.data.txt

.PHONY: clean
clean:
	rm -f k_means \
	*.o \
	./device/*.o