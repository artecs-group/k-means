DEVICE      = cpu # {cpu, nvidia, igpu}
REAL        = simple # {simple, double}

CPU = CPU_DEVICE
IGPU = INTEL_IGPU_DEVICE
NVIDIA = NVIDIA_DEVICE

# Program flags
PDEVICE = $(CPU)
PREAL = SP

ifeq ($(DEVICE),nvidia)
  PDEVICE = NVIDIA_DEVICE
else ifeq ($(DEVICE), igpu)
  PDEVICE = $(IGPU)
endif

ifeq ($(REAL),double)
  PREAL = DP
endif

DPCPP = dpcpp
CFLAGS = -O3 -D$(PDEVICE) -D$(PREAL) -qmkl -DMKL_ILP64 -I"${MKLROOT}/include"
CLINK = -fsycl-device-code-split=per_kernel  ${MKLROOT}/lib/intel64/libmkl_sycl.a -Wl,-export-dynamic -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb -lsycl -lOpenCL -lpthread -lm -ldl

ifeq ($(DEVICE),nvidia)
  DPCPP = clang++
  CFLAGS = -D$(PDEVICE) -D$(PREAL) -fsycl -fsycl-targets=nvptx64-nvidia-cuda -I${ONEMKL}/include
  CLINK = -fsycl-device-code-split=per_kernel -L${ONEMKL}/lib -lonemkl
endif

EXECNAME = kmeans

$(EXECNAME): main.o ./device/device.o ./init/init.o
	$(DPCPP) $(CFLAGS) $(CLINK) main.o ./device/device.o ./init/init.o -o $(EXECNAME)

main.o: ./device/device.o ./init/init.o main.cpp
	$(DPCPP) $(CFLAGS) main.cpp -c -o main.o

./init/init.o: ./init/init.cpp ./init/init.hpp
	$(DPCPP) $(CFLAGS) ./init/init.cpp -c -o ./init/init.o

./device/device.o: ./device/device.cpp ./device/device.hpp
	$(DPCPP) $(CFLAGS) ./device/device.cpp -c -o ./device/device.o

clean:
	rm -f *.o $(EXECNAME) *.linkinfo *~ *.bak .depend ./device/*.o ./init/*.o
