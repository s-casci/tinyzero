import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNetwork(nn.Module):
  def __init__(self, input_shape, action_space, first_layer_size=512, second_layer_size=256):
    super().__init__()
    self.first_layer = nn.Linear(input_shape[0], first_layer_size)
    self.second_layer = nn.Linear(first_layer_size, second_layer_size)
    self.value_head = nn.Linear(second_layer_size, 1)
    self.policy_head = nn.Linear(second_layer_size, action_space)

  def __call__(self, observations):
    x = F.relu(self.first_layer(observations))
    x = F.relu(self.second_layer(x))
    value = F.tanh(self.value_head(x))
    log_policy = F.log_softmax(self.policy_head(x), dim=-1)
    return value, log_policy

  @torch.no_grad
  def value_forward(self, observation):
    x = F.relu(self.first_layer(observation))
    x = F.relu(self.second_layer(x))
    value = F.tanh(self.value_head(x))
    return value

  @torch.no_grad
  def policy_forward(self, observation):
    x = F.relu(self.first_layer(observation))
    x = F.relu(self.second_layer(x))
    log_policy = F.softmax(self.policy_head(x), dim=-1)
    return log_policy


class ConvolutionalNetwork(nn.Module):
  def __init__(self, input_shape, action_space, first_linear_size=512, second_linear_size=256):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
    self.conv2 = nn.Conv2d(32, 32, kernel_size=1)
    self.conv3 = nn.Conv2d(32, 64, kernel_size=1)
    self.dropout = nn.Dropout2d(p=0.3)
    self.fc1 = nn.Linear(3 * 3 * 64, first_linear_size)
    self.fc2 = nn.Linear(first_linear_size, second_linear_size)
    self.value_head = nn.Linear(second_linear_size, 1)
    self.policy_head = nn.Linear(second_linear_size, action_space)

  def __call__(self, observations):
    x = F.relu(self.conv1(observations))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.dropout(x)
    x = x.view(-1, 3 * 3 * 64)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    value = F.tanh(self.value_head(x))
    log_policy = F.log_softmax(self.policy_head(x), dim=-1)
    return value, log_policy

  @torch.no_grad
  def value_forward(self, observation):
    x = F.relu(self.conv1(observation))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3 * 3 * 64)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    value = F.tanh(self.value_head(x))
    return value[0]

  @torch.no_grad
  def policy_forward(self, observation):
    x = F.relu(self.conv1(observation))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3 * 3 * 64)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    log_policy = F.softmax(self.policy_head(x), dim=-1)
    return log_policy[0]
