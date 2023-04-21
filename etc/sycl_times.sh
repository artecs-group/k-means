#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: Please specify oneapi location."
    echo "Usage: ./get_times.sh /opt/intel/oneapi <cpu, igpu, ngpu>"
    exit 1
fi

if [ -d ./data/program_times ]; then
    mv ./data/program_times ./data/program_times_$(date +%s)
fi

mkdir ./data/program_times

if ! test -e ./data/oneapi_config.txt; then
    echo "compiler=2022.2.0" >> ./data/oneapi_config.txt
fi

# sets oneapi compiler to 2022.2.0
source $1/setvars.sh --config=./data/oneapi_config.txt --force > /dev/null 2>&1 
echo $(dpcpp --version)
echo ""
echo "Be sure that \"Intel(R) oneAPI DPC++/C++ Compiler 2022.2.0\" was prompted."
echo ""

cd custom_impl/sycl

dev=$2
sycl="dpcpp"
attributes=2458285
dims=68
iterations=10
clusters=(4 16 64 256)
impls=("sycl-cpu" "sycl-ngpu" "sycl-igpu" "sycl-por")

for impl in "${impls[@]}"
do
    for k in "${clusters[@]}"
    do
        make clean 
        make DEVICE=${dev} SYCL=${sycl} IMPL=${impl} ATRIBUTES=${attributes} DIMS=${dims} CLUSTERS=${k} ITERATIONS=${iterations}
        touch "../../data/program_times/${dev}_${sycl}_${impl}_${attributes}_${dims}_${k}_${iterations}"
        for i in {1..10}
        do
            make run >> ../../data/program_times/${dev}_${sycl}_${impl}_${attributes}_${dims}_${k}_${iterations}
            echo ""  >> ../../data/program_times/${dev}_${sycl}_${impl}_${attributes}_${dims}_${k}_${iterations}
            echo ""  >> ../../data/program_times/${dev}_${sycl}_${impl}_${attributes}_${dims}_${k}_${iterations}
        done
    done
done



