from simple_processing import *

# contants
fname = '../train_data/stanford/bookstore_0.txt'     # location of the datafile


if __name__ == '__main__':    
    trajectories_simple, trajectories_others = process_text_file_csv(fname)
