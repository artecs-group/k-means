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

CC = clang++
CFLAGS = -D$(PDEVICE) -std=c++11 -O3 -fsycl
CLINK = -fsycl-device-code-split=per_kernel

ifeq ($(PDEVICE),NVIDIA_DEVICE)
	CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda
endif

k-means: k-means.o
	$(CC) $(CFLAGS) $(CLINK) k-means.o -o k-means

k-means.o: ./k-means.cpp
	$(CC) $(CFLAGS) k-means.cpp -c -o k-means.o

run: k-means
	./k-means ../data/100k.csv 4 300

.PHONY: clean
clean:
	rm -f k-means \
	*.o