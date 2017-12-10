import torch
from torchvision import models
import numpy as np
import os
import sys
from simple_processing import load_full_augmented_data

train_videos = ['bookstore_2', 'bookstore_3', 'coupa_3', 'deathcircle_2', \
                'deathcircle_3', 'deathcircle_4', 'gates_3', 'gates_4', \
                'gates_5', 'gates_6', 'gates_7', 'gates_8', 'hyang_4', \
                'hyang_5', 'hyang_6', 'hyang_7', 'hyang_9', 'nexus_3', \
                'nexus_4', 'nexus_7', 'nexus_8', 'nexus_9']

dev_videos = ['bookstore_0', 'deathcircle_0', 'gates_0', 'nexus_0']

test_videos = ['bookstore_1', 'deathcircle_1', 'gates_1', 'nexus_1']


def load_data(include_me=True, include_others=True, include_seg=True, use_encoding=True, N=100):
  
  if use_encoding and not (include_others and include_seg and include_me):
    print('must include all 3 channels if encodings are to be used')
  
  alexnet = models.alexnet(pretrained=True)
  cnn_layer = torch.nn.Sequential(*(list(alexnet.features.children()))[:-1]).cuda()

  train_prefix = '/train_data/'
  dev_prefix = '/dev_data/'
  test_prefix = '/test_data/'

  os.makedirs('/output' + train_prefix)
  os.makedirs('/output' + dev_prefix)
  os.makedirs('/output' + test_prefix)
 
  train_trajectories = get_trajectories(train_prefix,
                                        train_videos,
                                        cnn_layer,
                                        include_me,
                                        include_others,
                                        include_seg,
                                        use_encoding,
                                        N)
  dev_trajectories = get_trajectories(dev_prefix,
                                      dev_videos,
                                      cnn_layer,
                                      include_me,
                                      include_others,
                                      include_seg,
                                      use_encoding,
                                      N)
  test_trajectories = get_trajectories(test_prefix,
                                       test_videos,
                                       cnn_layer,
                                       include_me,
                                       include_others,
                                       include_seg,
                                       use_encoding,
                                       N)

def get_trajectories(prefix,
                     videos,
                     cnn_layer,
                     include_me=True,
                     include_others=True,
                     include_seg=True,
                     use_encoding=True,
                     N=100):

  for video in videos:
    print(video)
    video_trajectories = []
    filename = prefix + video

    trajs = load_full_augmented_data(filename, N)
    
    for traj in trajs:
      new_traj = []
      for xy_array, me_array, others_array, seg_array in traj:
        features = xy_array

        if use_encoding:
          cnn_input = np.stack((me_array, others_array, seg_array)).reshape((1, 3, N, N))
          cnn_input_tensor = torch.autograd.Variable(torch.cuda.FloatTensor(cnn_input))
          cnn_output_tensor = cnn_layer(cnn_input_tensor)
          cnn_output = cnn_output_tensor.data.numpy()

          features = np.concatenate((features, cnn_output.flatten()))
        else:
          if include_me:
            features = np.concatenate((features, me_array.flatten()))
          if include_others:
            features = np.concatenate((features, others_array.flatten()))
          if include_seg:
            features = np.concatenate((features, seg_array.flatten()))

        new_traj.append(features)

      new_traj = np.stack(new_traj)
      video_trajectories.append(new_traj)
   
    video_trajectories = np.stack(video_trajectories)
    np.save('/output' + prefix + video, video_trajectories)

if __name__ == '__main__':
  load_data(True, True, True, True, 40)

