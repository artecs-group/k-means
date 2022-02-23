DEVICE = cpu # {cpu, nvidia, igpu}

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
CFLAGS = -D$(PDEVICE) -std=c++11 -O3 -fsycl
CLINK = -fsycl-device-code-split=per_kernel

ifeq ($(PDEVICE),NVIDIA_DEVICE)
	CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda
endif

main: main.o ./device/device.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./device/device.o -o k-means

main.o: ./main.cpp ./device/device.o
	$(CC) $(CFLAGS) main.cpp -c -o main.o

device.o: ./device/device.hpp ./device/device.cpp
	$(CC) $(CFLAGS) ./device/device.cpp -c -o ./device/device.o

run: k-means
	./k-means ../data/100k.csv 4 300

.PHONY: clean
clean:
	rm -f k-means \
	*.o \
	./device/*.o