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
