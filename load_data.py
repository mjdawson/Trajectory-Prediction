import os
import sys
sys.path.append('./processing_scripts/')
from simple_processing import load_simple_array

def load_data():

  train_trajectories = []
  for filename in os.listdir('train/stanford/annotations/'):
    train_trajectories += load_simple_array('train/stanford/annotations/' + filename)

  dev_trajectories = []
  for filename in os.listdir('dev/stanford/annotations/'):
    dev_trajectories += load_simple_array('dev/stanford/annotations/' + filename)

  test_trajectories = []
  for filename in os.listdir('test/stanford/annotations/'):
    test_trajectories += load_simple_array('test/stanford/annotations/' + filename)

  return train_trajectories, dev_trajectories, test_trajectories
    
