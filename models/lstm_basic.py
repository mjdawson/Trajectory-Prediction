import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import sys

def train_example():
  lstm = nn.LSTM(2, 2)

  input_len = 20
  points = [autograd.Variable(torch.randn((1, 2))) for _ in range(input_len)]

  hidden = (autograd.Variable(torch.randn(1, 1, 2)),
            autograd.Variable(torch.randn(1, 1, 2)))

  first_half_points = points[:input_len / 2]
  last_half_points = points[input_len / 2:]

  for p in first_half_points:
    out, hidden = lstm(p.view(1, 1, -1), hidden)

  loss_function = nn.L1Loss()

  last_half_inputs = [out] + last_half_points[1:]
  last_half_predictions = []

  for i in last_half_inputs:
    pred, hidden = lstm(i.view(1, 1, -1), hidden)
    last_half_predictions.append(pred)

  ground_truth_tensor = torch.stack(last_half_points)
  predictions_tensor = torch.stack(last_half_predictions)

  loss = loss_function(predictions_tensor, ground_truth_tensor)
  loss.backward()

def test_example():
  lstm = nn.LSTM(2, 2)

  input_len = 20
  points = [autograd.Variable(torch.randn((1, 2))) for _ in range(input_len)]

  hidden = (autograd.Variable(torch.randn(1, 1, 2)),
            autograd.Variable(torch.randn(1, 1, 2)))

  first_half_points = points[:input_len / 2]
  last_half_points = points[input_len / 2:]

  for p in first_half_points:
    out, hidden = lstm(p.view(1, 1, -1), hidden)

  loss_function = nn.L1Loss()

  last_half_predictions = []

  pred = out
  for _ in last_half_points:
    pred, hidden = lstm(pred, hidden)
    last_half_predictions.append(pred)

  ground_truth_tensor = torch.stack(last_half_points)
  predictions_tensor = torch.stack(last_half_predictions)

  loss = loss_function(predictions_tensor, ground_truth_tensor)
    
if __name__ == '__main__':
  train_example()
  test_example()
