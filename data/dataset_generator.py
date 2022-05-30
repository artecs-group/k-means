import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Arguments must be: python dataset_generator.py n_rows n_columns")
        exit(0)

    n_rows: int = int(sys.argv[1])
    n_columns: int = int(sys.argv[2])
    dataset = np.random.rand(n_rows, n_columns) 
    np.savetxt(f'data/synthetic_{n_rows}_{n_columns}.txt', dataset, delimiter=' ')
