from tinygrad import nn


class LinearNetwork:
  def __init__(self, input_shape, action_space, first_layer_size=512, second_layer_size=256):
    self.first_layer = nn.Linear(input_shape[0], first_layer_size)
    self.second_layer = nn.Linear(first_layer_size, second_layer_size)
    self.value_head = nn.Linear(second_layer_size, 1)
    self.policy_head = nn.Linear(second_layer_size, action_space)

  def __call__(self, observations):
    x = self.first_layer(observations).relu()
    x = self.second_layer(x).relu()
    value = self.value_head(x).tanh()
    log_policy = self.policy_head(x).log_softmax(axis=-1)
    return value, log_policy
