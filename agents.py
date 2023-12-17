from tinygrad import Tensor, TinyJit
import numpy as np
from replay_buffer import ReplayBuffer
from training import train_step
from mcts import search


class AlphaZeroAgent:
  def __init__(self, model, optimizer=None, replay_buffer_max_size=None):
    self.model = model
    # optimizer and training buffer might be None if the agent is used for evaluation only
    self.optimizer = optimizer
    self.training_buffer = ReplayBuffer(max_size=replay_buffer_max_size)

  @staticmethod
  @TinyJit
  def _value_fn_jitted(model, observation):
    Tensor.no_grad = True
    x = model.first_layer(observation).relu()
    x = model.second_layer(x).relu()
    value = model.value_head(x).tanh().realize()
    Tensor.no_grad = False
    return value

  @staticmethod
  @TinyJit
  def _policy_fn_jitted(model, observation):
    Tensor.no_grad = True
    x = model.first_layer(observation).relu()
    x = model.second_layer(x).relu()
    policy = model.policy_head(x).softmax(axis=-1).realize()
    Tensor.no_grad = False
    return policy

  def reset(self):
    self._value_fn_jitted.reset()
    self._policy_fn_jitted.reset()

  def value_fn(self, game):
    observation = game.to_observation()
    return AlphaZeroAgent._value_fn_jitted(self.model, Tensor(observation)).item()

  def policy_fn(self, game):
    observation = game.to_observation()
    return AlphaZeroAgent._policy_fn_jitted(self.model, Tensor(observation)).numpy()

  def selfplay(self, game, search_iterations, c_puct=1.0, dirichlet_alpha=None):
    buffer = []
    while (first_person_result := game.get_first_person_result()) is None:
      root_node = search(
        game, self.value_fn, self.policy_fn, search_iterations, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
      )
      visits_dist = root_node.children_visits / root_node.children_visits.sum()

      action = root_node.children_actions[np.random.choice(len(root_node.children), p=visits_dist)]

      actions_dist = np.zeros(game.action_space, dtype=np.float32)
      actions_dist[root_node.children_actions] = visits_dist
      buffer.append((game.to_observation(), actions_dist))

      game.step(action)

    return first_person_result, buffer

  def train_step(self, game, search_iterations, batch_size, epochs, c_puct=1.0, dirichlet_alpha=None):
    first_person_result, game_buffer = self.selfplay(
      game, search_iterations, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
    )

    result = game.swap_result(first_person_result)
    while len(game_buffer) > 0:
      observation, action_dist = game_buffer.pop()
      self.training_buffer.add_sample(observation, action_dist, result)
      result = game.swap_result(result)

    values_losses, policies_losses = [], []
    if len(self.training_buffer) >= batch_size:
      for _ in range(epochs):
        observations, actions_dist, results = self.training_buffer.sample(batch_size)
        values_loss, policies_loss = train_step(self.model, self.optimizer, observations, actions_dist, results)
        values_losses.append(values_loss)
        policies_losses.append(policies_loss)
      self.reset()

    return values_losses, policies_losses
