import sys
sys.path.append('./processing_scripts/')
import numpy as np
from simple_processing import process_text_file
from linear_error import compute_linear_error

# contants
fname = './train_data/stanford/bookstore_0.txt'     # location of the datafile
Nf = 3


if __name__ == '__main__':    
    trajectories_simple, trajectories_others = process_text_file(fname)
    print(trajectories_simple[100])
