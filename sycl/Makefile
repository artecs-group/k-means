DEVICE = cpu # {cpu, nvidia, igpu}
DEBUG = no #{yes, no}

CPU = CPU_DEVICE
IGPU = INTEL_GPU_DEVICE
NVIDIA = NVIDIA_DEVICE

PDEVICE = $(CPU)

ifeq ($(DEVICE),nvidia)
  PDEVICE = NVIDIA_DEVICE
else ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
endif

CC = dpcpp
CFLAGS = -D$(PDEVICE) -fsycl
CLINK = -fsycl-device-code-split=per_kernel

ifeq ($(PDEVICE),NVIDIA_DEVICE)
	CC = clang++
	CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda -I"${ONEAPI_ROOT}/dpl/latest/linux/include/" -I"${ONEAPI_ROOT}/tbb/latest/include/"
endif

ifeq ($(DEBUG),yes)
	CFLAGS += -g -O0
else
  CFLAGS += -O3
endif

main: ./device/device.o main.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./device/device.o -o k_means

main.o: ./device/device.o
	$(CC) $(CFLAGS) main.cpp -c -o main.o

./device/device.o: ./device/device.hpp
	$(CC) $(CFLAGS) ./device/device.cpp -c -o ./device/device.o

run: main
	./k_means ../data/100k.csv 4 100000 300

.PHONY: clean
clean:
	rm -f k_means *.o ./device/*.o