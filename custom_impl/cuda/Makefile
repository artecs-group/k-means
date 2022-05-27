DEBUG = no # {yes, no}

CC = nvcc
CFLAGS = -std=c++17

ifeq ($(DEBUG),yes)
	CFLAGS += -g -G -O0
else
	CFLAGS += -O3
endif

k_means: ./device/device.o main.o
	$(CC) $(CFLAGS) main.o ./device/device.o -o k_means

main.o: ./device/device.o
	$(CC) $(CFLAGS) main.cu -c -o main.o

./device/device.o: ./device/device.cuh
	$(CC) $(CFLAGS) ./device/device.cu -c -o ./device/device.o

run: k_means
	./k_means ../../data/household_power_consumption.txt 4 7 2049280 20

.PHONY: clean
clean:
	rm -f k_means \
	*.o \
	./device/*.o