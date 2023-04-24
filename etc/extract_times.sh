#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Error: Please specify input and output locations."
    echo "Usage: ./extract_times.sh <input.txt> <output.txt>"
    exit 1
fi

grep -Eo 'Total time = [0-9]+\.[0-9]+s' $1 | sed -E 's/Total time = ([0-9]+\.[0-9]+)s/\1/' > $2
