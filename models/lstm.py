import torch

class TrajectoryPredictor:

  def __init__(self, input_dim, output_dim, batch_size):

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size

    self.lstm = torch.nn.LSTM(input_dim, output_dim)

    self.init_state()

  # tensors should be sequence_len x batch_size x input_dim
  def train(self, train_loader, num_epochs):

    loss_function = torch.nn.PairwiseDistance()
    optimizer = torch.optim.SGD(self.lstm.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
      
      for data in train_loader:

        optimizer.zero_grads()
        
        self.init_state()

        traj_first_part, traj_second_part = data
        traj_first_part, traj_second_part = torch.autograd.Variable(traj_first_part), \
                                            torch.autograd.Variable(traj_second_part)

        first_part_len = int(traj_first_part.size()[0])
        second_part_len = int(traj_second_part.size()[1])

        for i in range(first_part_len):
          inp = traj_first_part[i,:,:].view(1, self.batch_size, self.input_dim)
          out, self.state = self.lstm(inp, self.state)

        second_part_preds = []

        for i in range(second_part_len):
          if i == 0:
            inp = out
          else:
            inp = traj_second_part[i,:,:].view(1, self.batch_size, self.input_dim)

          pred, self.state = self.lstm(inp, self.state)
          second_part_preds.append(pred)

        second_part_preds = torch.cat(second_part_preds)

        ground_truth_points = traj_second_part.view(-1, self.output_dim)
        pred_points = second_part_preds.view(-1, self.output_dim)

        loss = loss_function(ground_truth_points, pred_points).sum()
        loss.backward()
        optimizer.step()

  def test(self, test_loader):
    
    loss_function = torch.nn.PairwiseDistance()
    total_loss = 0.0

    for data in test_loader:

      self.init_state()

      traj_first_part, traj_second_part = data
      traj_first_part, traj_second_part = torch.autograd.Variable(traj_first_part), \
                                          torch.autograd.Variable(traj_second_part)

      first_part_len = int(traj_first_part.size()[0])
      second_part_len = int(traj_second_part.size()[0])

      for i in range(first_part_len):
        inp = traj_first_part[i,:,:].view(1, self.batch_size, self.input_dim)
        out, self.state = self.lstm(inp, self.state)

      second_part_preds = [out]

      for i in range(second_part_len - 1):
        out, self.state = self.lstm(out, self.state)
        second_part_preds.append(out)

      second_part_preds = torch.cat(second_part_preds)

      ground_truth_points = traj_second_part.view(-1, self.output_dim)
      pred_points = second_part_preds.view(-1, self.output_dim)

      loss = loss_function(ground_truth_points, pred_points).sum()
      
      total_loss += loss.data[0]

    return total_loss

  def init_state(self):
    # first dimension is number of layers; not sure what adding layers does
    h_init = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.output_dim))
    c_init = torch.autograd.Variable(torch.zeros(1, self.batch_size, self.output_dim))

    self.state = (h_init, c_init)


