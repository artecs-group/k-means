CC = nvcc
CFLAGS = -std=c++11 -O3

k_means: k_means.o
	$(CC) $(CFLAGS) $(CLINK) k_means.o -o k_means

k_means.o: ./k_means.cu
	$(CC) $(CFLAGS) k_means.cu -c -o k_means.o

run: k_means
	./k_means ../data/100k.csv 4 300 100000

.PHONY: clean
clean:
	rm -f k_means \
	*.o