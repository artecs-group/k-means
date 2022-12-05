DEVICE = cpu # {cpu, nvidia, igpu}
DEBUG = no # {yes, no}
SYCL = dpcpp # {dpcpp, hipsycl}: hipsycl does not support igpu
HIPSYCL_COMPILER = clang # {clang, nvc++}

CPU = CPU_DEVICE
IGPU = INTEL_GPU_DEVICE
NVIDIA = NVIDIA_DEVICE

PDEVICE = $(CPU)

ifeq ($(DEVICE),nvidia)
  PDEVICE = $(NVIDIA)
else ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
endif

ifeq ($(SYCL), dpcpp)
	CC = dpcpp
	CFLAGS = -O3 -D$(PDEVICE) -fsycl
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
	CFLAGS = -O3 -D$(PDEVICE)
	ifeq ($(PDEVICE),$(NVIDIA))
		ifeq ($(HIPSYCL_COMPILER),clang)
			CFLAGS += --hipsycl-targets="cuda:sm_61" -I/usr/local/cuda/include
			CLINK = -L/usr/local/cuda/include
		else
			CFLAGS += --hipsycl-targets="cuda-nvcxx:cc61"
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