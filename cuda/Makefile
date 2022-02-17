CC = nvcc
CFLAGS = -std=c++11 -O3

k-means: k-means.o
	$(CC) $(CFLAGS) $(CLINK) k-means.o -o k-means

k-means.o: ./k-means.cu
	$(CC) $(CFLAGS) k-means.cu -c -o k-means.o

run: k-means
	./k-means ../data/100k.csv 4 300

.PHONY: clean
clean:
	rm -f k-means \
	*.o