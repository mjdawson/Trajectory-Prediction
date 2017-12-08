import numpy as np
import sys
sys.path.append('./processing_scripts/')
from simple_processing import load_others_array

path = './train/stanford/annotations/gates_6.txt'

T = load_others_array(path,N=10)
