import os
import sys
sys.path.append('./processing_scripts/')
from simple_processing import load_simple_array
from linear_error import compute_linear_error

if __name__ == '__main__':
  
  trajectories = []

  for dirname in os.listdir('dev'):
    if dirname == 'stanford':
      for filename in os.listdir('dev/' + dirname + '/annotations'):
        if filename.endswith('.txt'):
          trajectories += load_simple_array('dev/' + dirname + '/annotations/' + filename)

  error = compute_linear_error(trajectories, 10) 
  print error
