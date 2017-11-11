import torch

class TrajectoryPredictor:

  def __init__(self, input_dim, output_dim, batch_size):

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.batch_size = batch_size

    self.lstm = torch.nn.LSTM(input_dim, output_dim)

    # first dimension is number of layers; not sure what adding layers does
    h_init = torch.autograd.Variable(torch.randn(1, batch_size, output_dim))
    c_init = torch.autograd.Variable(torch.randn(1, batch_size, output_dim))

    self.state = (h_init, c_init)

  # tensors should be sequence_len x batch_size x input_dim
  def train(self, train_loader, num_epochs):

    loss_function = torch.nn.PairwiseDistance()
    optimizer = torch.optim.SGD(self.lstm.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):
      
      for data in train_loader:

        optimizer.zero_grads()

        traj_first_half, traj_second_half = data
        traj_first_half, traj_second_half = torch.autograd.Variable(traj_first_half), \
                                            torch.autograd.Variable(traj_second_half)

        first_half_len = int(traj_first_half.size()[0])
        second_half_len = int(traj_second_half.size()[1])

        for i in range(first_half_len):
          inp = traj_first_half[i,:,:].view(1, self.batch_size, self.input_dim)
          out, self.state = self.lstm(inp, self.state)

        second_half_preds = []

        for i in range(second_half_len):
          if i == 0:
            inp = out
          else:
            inp = traj_second_half[i,:,:].view(1, self.batch_size, self.input_dim)

          pred, self.state = self.lstm(inp, self.state)
          second_half_preds.append(pred)

        second_half_preds = torch.cat(second_half_preds)

        ground_truth_points = traj_second_half.view(-1, output_dim)
        pred_points = second_half_preds.view(-1, output_dim)

        loss = loss_function(ground_truth_points, pred_points).sum()
        loss.backward()
        optimizer.step()



    
