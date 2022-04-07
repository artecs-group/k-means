CC = nvcc
CFLAGS = -std=c++17 -O3

main: ./device/device.o main.o
	$(CC) $(CFLAGS) main.o ./device/device.o -o k_means

main.o: ./device/device.o
	$(CC) $(CFLAGS) main.cu -c -o main.o

./device/device.o: ./device/device.cuh
	$(CC) $(CFLAGS) ./device/device.cu -c -o ./device/device.o

run: main
	./k_means ../data/100k.csv 4 2 100000 300

.PHONY: clean
clean:
	rm -f k_means \
	*.o \
	./device/*.o