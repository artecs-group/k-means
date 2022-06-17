DEBUG = no # {yes, no}

CC = nvcc
CFLAGS = -std=c++17 -O3

ifeq ($(DEBUG),yes)
	CFLAGS += -g
endif

k_means: ./device/device.o main.o
	$(CC) $(CFLAGS) main.o ./device/device.o -o k_means

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