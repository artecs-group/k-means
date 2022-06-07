DEBUG = no # {yes, no}

CC = clang++
DPCT_INCLUDE_DIR = ${ONEAPI_ROOT}/dpcpp-ct/latest/include/
CFLAGS = -fsycl -w -fsycl-targets=nvptx64-nvidia-cuda -I$(DPCT_INCLUDE_DIR)
CLINK = -fsycl-device-code-split=per_kernel

ifeq ($(DEBUG),yes)
	CFLAGS += -g -O0
else
	CFLAGS += -O3
endif

k_means: ./device/device.o main.o
	$(CC) $(CFLAGS) $(CLINK) main.o ./device/device.o -o k_means

main.o: ./device/device.o
	$(CC) $(CFLAGS) main.dp.cpp -c -o main.o

./device/device.o: ./device/device.dp.hpp
	$(CC) $(CFLAGS) ./device/device.dp.cpp -c -o ./device/device.o

run: k_means
	./k_means ../../data/household_power_consumption.txt 4 7 2049280 20

.PHONY: clean
clean:
	rm -f k_means \
	*.o \
	./device/*.o