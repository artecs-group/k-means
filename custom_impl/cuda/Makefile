COMPILER = nvcc #{nvcc, nvc++, clang}
DEBUG = no # {yes, no}

CC = nvcc
CFLAGS = -std=c++17 -O3

ifeq ($(COMPILER),nvc++)
	CC = nvc++
	CFLAGS += -gpu=cc61 -cuda
endif

ifeq ($(COMPILER),clang)
	CC = /usr/bin/clang++-15
	CFLAGS += -pthread --cuda-path=/usr/local/cuda
	CLINK = -L/usr/local/cuda/lib64 -lcudart -ldl -lrt
endif

ifeq ($(DEBUG),yes)
	CFLAGS += -g
endif

k_means: ./device/device.o main.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./device/device.o -o k_means

main.o: ./device/device.o
	$(CC) $(CFLAGS) main.cu -c -o main.o

./device/device.o: ./device/device.cuh
	$(CC) $(CFLAGS) ./device/device.cu -c -o ./device/device.o

run: k_means
	./k_means ../../data/USCensus1990.data.txt

.PHONY: clean
clean:
	rm -f k_means \
	*.o \
	./device/*.o