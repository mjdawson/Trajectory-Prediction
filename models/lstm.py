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

  def __init__(self, input_dim, output_dim, batch_size, lstm_state_dict=None):

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size

    self.lstm = torch.nn.LSTM(input_dim, output_dim)

    if lstm_state_dict:
      self.lstm.load_state_dict(lstm_state_dict)

  # tensors should be sequence_len x batch_size x input_dim
  def train(self, train_loader, num_epochs, lr=0.001, wd=0):

    loss_function = torch.nn.PairwiseDistance()
    optimizer = torch.optim.Adam(self.lstm.parameters(), lr=lr, weight_decay=wd)

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

        for i in range(first_part_len):
          inp = traj_first_part[:,i,:].contiguous().view(1, actual_batch_size, self.input_dim)
          out, self.state = self.lstm(inp, self.state)
          
        second_part_preds = [out.contiguous().view(actual_batch_size, 1, self.output_dim)]
        
        for i in range(second_part_len-1):
          inp = traj_second_part[:,i,:].contiguous().view(1, actual_batch_size, self.input_dim)
          pred, self.state = self.lstm(inp, self.state)
          second_part_preds.append(pred.contiguous().view(actual_batch_size, 1, self.output_dim))

        second_part_preds = torch.cat(second_part_preds, dim=1)

        ground_truth_points = traj_second_part[:,:,0:2].contiguous().view(-1, self.output_dim)
        pred_points = second_part_preds.contiguous().view(-1, self.output_dim)

        loss = loss_function(ground_truth_points, pred_points).sum()
        loss.backward()

        total_loss += loss.data[0]

        optimizer.step()

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

      for i in range(first_part_len):
        inp = traj_first_part[:,i,:].contiguous().view(1, actual_batch_size, self.input_dim)
        out, self.state = self.lstm(inp, self.state)
      
      second_part_preds = [out.contiguous().view(actual_batch_size, 1, self.output_dim)]


      prev_out = out

      for i in range(second_part_len-1):
        inp = torch.cat((out, traj_second_part[:,i,2:].contiguous().view(1, actual_batch_size, -1)), dim=2)
        out, self.state = self.lstm(inp, self.state)
        second_part_preds.append(out.contiguous().view(actual_batch_size, 1, self.output_dim))

      second_part_preds = torch.cat(second_part_preds, dim=1)

      ground_truth_points = traj_second_part[:,:,0:2].contiguous().view(-1, self.output_dim)
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
  for dirname in os.listdir('train'):
    if dirname == 'stanford':
      for filename in os.listdir('train/' + dirname + '/annotations'):
        if filename.endswith('.txt'):
          trajectories = load_simple_array('train/' + dirname + '/annotations/' + filename)
          train_trajectories1 += [traj[:10,:] for traj in trajectories]
          train_trajectories2 += [traj[10:,:] for traj in trajectories]

  dev_trajectories1 = []
  dev_trajectories2 = []
  for dirname in os.listdir('dev'):
    if dirname == 'stanford':
      for filename in os.listdir('dev/' + dirname + '/annotations'):
        if filename.endswith('.txt'):
          trajectories = load_simple_array('dev/' + dirname + '/annotations/' + filename)
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
  num_epochs = 20

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size)
  test_loader = torch.utils.data.DataLoader(dev_dataset, batch_size)

  input_dim = 2
  output_dim = 2

  p = TrajectoryPredictor(input_dim, output_dim, batch_size)
  p.train(train_loader, num_epochs)

  loss, ground_truth_trajectories, predicted_trajectories = p.test(test_loader)

  print loss

  for i in range(len(ground_truth_trajectories)):
    plot_trajectories(ground_truth_trajectories[i], predicted_trajectories[i])

