import torch
import torch.utils.data
import os
import numpy as np
import sys
import time
sys.path.append('./processing_scripts/')
sys.path.append('./')
from plot_trajectories import plot_trajectories
from simple_processing import load_simple_array


class TrajectoryPredictor:

  def __init__(self, input_dim, middle_dim, output_dim, batch_size):

    self.input_dim = input_dim
    self.middle_dim = middle_dim
    self.output_dim = output_dim
    self.batch_size = batch_size

    self.linear = torch.nn.Linear(input_dim, middle_dim)
    self.lstm = torch.nn.LSTM(middle_dim, output_dim)

  # tensors should be sequence_len x batch_size x input_dim
  def train(self, train_loader, num_epochs):

    loss_function = torch.nn.PairwiseDistance()
    optimizer = torch.optim.Adam([{'params':self.lstm.parameters()}, {'params':self.linear.parameters()}])

    for epoch in range(num_epochs):   
      total_loss = 0.0
      count = 0
      for data in train_loader:

        optimizer.zero_grad()
        traj_first_part, traj_second_part = data
        traj_first_part, traj_second_part = torch.autograd.Variable(traj_first_part), \
                                            torch.autograd.Variable(traj_second_part)

        first_part_len = int(traj_first_part.size()[1])
        second_part_len = int(traj_second_part.size()[1])

        actual_batch_size = min(int(traj_first_part.size()[0]), self.batch_size)
        count += actual_batch_size
        self.init_state(actual_batch_size)

        v_x = traj_first_part[:,1,0] - traj_first_part[:,0,0]
        v_y = traj_first_part[:,1,1] - traj_first_part[:,0,1]

        for i in range(first_part_len):
          if i > 0:
            v_x = traj_first_part[:,i,0] - traj_first_part[:,i-1,0]
            v_y = traj_first_part[:,i,1] - traj_first_part[:,i-1,1]
          inp = torch.cat((traj_first_part[:,i,:], v_x.contiguous().view(actual_batch_size, 1), \
                          v_y.contiguous().view(actual_batch_size, 1)), dim=1).contiguous().view(1, actual_batch_size, self.input_dim)
          tmp = self.linear(inp)
          out, self.state = self.lstm(tmp, self.state)
          
        second_part_preds = [out.contiguous().view(actual_batch_size, 1, self.output_dim)]

        v_x = traj_second_part[:,0,0] - traj_first_part[:, first_part_len - 1, 0]
        v_y = traj_second_part[:,0,1] - traj_first_part[:, first_part_len - 1, 1]
        
        for i in range(second_part_len-1):
          #if i > 0:
          #  v_x = traj_second_part[:,i,0] - traj_second_part[:,i-1,0]
          #  v_y = traj_second_part[:,i,1] - traj_second_part[:,i-1,1]
          inp = torch.cat((traj_second_part[:,i,:], v_x.contiguous().view(actual_batch_size, 1), \
                          v_y.contiguous().view(actual_batch_size, 1)), dim=1).contiguous().view(1, actual_batch_size, self.input_dim)

          tmp = self.linear(inp)
          pred, self.state = self.lstm(tmp, self.state)
          second_part_preds.append(pred.contiguous().view(actual_batch_size, 1, self.output_dim))

        second_part_preds = torch.cat(second_part_preds, dim=1)

        ground_truth_points = traj_second_part.contiguous().view(-1, self.output_dim)
        pred_points = second_part_preds.contiguous().view(-1, self.output_dim)

        loss = loss_function(ground_truth_points, pred_points).sum()
        loss.backward()

        total_loss += loss.data[0]

        optimizer.step()
      print(count)
      print("epoch: %s. mean loss: %s" % (epoch+1,total_loss / count))

  def test(self, test_loader):
    
    loss_function = torch.nn.PairwiseDistance()
    total_loss = 0.0
    count = 0

    ground_truth_trajectories = []
    predicted_trajectories = []

    for data in test_loader:

      traj_first_part, traj_second_part = data
      traj_first_part, traj_second_part = torch.autograd.Variable(traj_first_part), \
                                          torch.autograd.Variable(traj_second_part)

      first_part_len = int(traj_first_part.size()[1])
      second_part_len = int(traj_second_part.size()[1])

      actual_batch_size = min(int(traj_first_part.size()[0]), self.batch_size)
      count += actual_batch_size

      self.init_state(actual_batch_size)

      v_x = traj_first_part[:,1,0] - traj_first_part[:,0,0]
      v_y = traj_first_part[:,1,1] - traj_first_part[:,0,1]

      for i in range(first_part_len):
        if i > 0:
          v_x = traj_first_part[:,i,0] - traj_first_part[:,i-1,0]
          v_y = traj_first_part[:,i,1] - traj_first_part[:,i-1,1]
        inp = torch.cat((traj_first_part[:,i,:], v_x.contiguous().view(actual_batch_size, 1), \
                         v_y.contiguous().view(actual_batch_size, 1)), dim=1).contiguous().view(1, actual_batch_size, self.input_dim)
        tmp = self.linear(inp)                 
        out, self.state = self.lstm(tmp, self.state)
      
      second_part_preds = [out.contiguous().view(actual_batch_size, 1, self.output_dim)]

      v_x = out[0,:,0] - traj_first_part[:,first_part_len - 1,0]
      v_y = out[0,:,1] - traj_first_part[:,first_part_len - 1,1]

      prev_out = out

      for i in range(second_part_len-1):
        #if i > 0:
        #  v_x = out[0,:,0] - prev_out[0,:,0]
        #  v_y = out[0,:,1] - prev_out[0,:,1]
        prev_out = out
        inp = torch.cat((out, v_x.contiguous().view(1,actual_batch_size,1), v_y.contiguous().view(1,actual_batch_size,1)), dim=2)

        tmp = self.linear(inp)
        out, self.state = self.lstm(tmp, self.state)
        second_part_preds.append(out.contiguous().view(actual_batch_size, 1, self.output_dim))

      second_part_preds = torch.cat(second_part_preds, dim=1)

      ground_truth_points = traj_second_part.contiguous().view(-1, self.output_dim)
      pred_points = second_part_preds.contiguous().view(-1, self.output_dim)
      
      ground_truth_trajectories.append(ground_truth_points.data.numpy())
      predicted_trajectories.append(pred_points.data.numpy())

      loss = loss_function(ground_truth_points, pred_points).sum()
      total_loss += loss.data[0]

    return total_loss / count, ground_truth_trajectories, predicted_trajectories

  def init_state(self, batch_size):
    # first dimension is number of layers
    h_init = torch.autograd.Variable(torch.zeros(1, batch_size, self.output_dim))
    c_init = torch.autograd.Variable(torch.zeros(1, batch_size, self.output_dim))

    self.state = (h_init, c_init)

if __name__ == '__main__':
  train_trajectories1 = []
  train_trajectories2 = []
  for filename in os.listdir('train'):
    if filename.endswith('.txt'):
      trajectories = load_simple_array('train/' + filename, augment=True)
      train_trajectories1 += [traj[:10,:] for traj in trajectories]
      train_trajectories2 += [traj[10:,:] for traj in trajectories]

  dev_trajectories1 = []
  dev_trajectories2 = []
  for filename in os.listdir('dev'):
    if filename.endswith('.txt'):
      trajectories = load_simple_array('dev/' + filename, augment=True)
      dev_trajectories1 += [traj[:10,:] for traj in trajectories]
      dev_trajectories2 += [traj[10:,:] for traj in trajectories]
  
  train_trajectories1 = np.stack(train_trajectories1)
  train_trajectories2 = np.stack(train_trajectories2)
  train_data_tensor = torch.Tensor(train_trajectories1)
  train_target_tensor = torch.Tensor(train_trajectories2)

  train_dataset = torch.utils.data.TensorDataset(train_data_tensor, train_target_tensor)

  dev_trajectories1 = np.stack(dev_trajectories1)
  dev_trajectories2 = np.stack(dev_trajectories2)
  dev_data_tensor = torch.Tensor(dev_trajectories1)
  dev_target_tensor = torch.Tensor(dev_trajectories2)

  dev_dataset = torch.utils.data.TensorDataset(dev_data_tensor, dev_target_tensor)

  batch_size = 4
  num_epochs = 2

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
  test_loader = torch.utils.data.DataLoader(dev_dataset, batch_size)

  input_dim = 4
  middle_dim = 10
  output_dim = 2

  p = TrajectoryPredictor(input_dim, middle_dim, output_dim, batch_size)
  p.train(train_loader, num_epochs)

  loss, ground_truth_trajectories, predicted_trajectories = p.test(test_loader)

  print loss

  for i in range(len(ground_truth_trajectories)):
    plot_trajectories(ground_truth_trajectories[i], predicted_trajectories[i])

