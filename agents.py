import torch
import torch.nn.functional as F
import numpy as np
from replay_buffer import ReplayBuffer
import copy
from mcts import search


class ClassicMCTSAgent:
  @staticmethod
  def value_fn(game):
    game = copy.deepcopy(game)
    while first_person_result := game.get_first_person_result() is None:
      game.step(np.random.choice(game.get_legal_actions()))
    return first_person_result

  @staticmethod
  def policy_fn(game):
    return np.ones(game.action_space) / game.action_space


class AlphaZeroAgent:
  def __init__(self, model, optimizer=None, replay_buffer_max_size=None):
    self.model = model
    # optimizer and training buffer might be None if the agent is used for evaluation only
    self.optimizer = optimizer
    self.replay_buffer = ReplayBuffer(max_size=replay_buffer_max_size)

  @torch.no_grad
  def value_fn(self, game):
    observation = torch.tensor(game.to_observation(), device=self.model.device)
    self.model.eval()
    value = self.model.value_forward(observation)
    return value.item()

  @torch.no_grad
  def policy_fn(self, game):
    observation = torch.tensor(game.to_observation(), device=self.model.device)
    self.model.eval()
    policy = self.model.policy_forward(observation)
    return policy.numpy()

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

  def save_training_state(self, model_out_path, optimizer_out_path):
    torch.save(self.model.state_dict(), model_out_path)
    torch.save(self.optimizer.state_dict(), optimizer_out_path)

  def load_training_state(self, model_out_path, optimizer_out_path):
    self.model.load_state_dict(torch.load(model_out_path))
    self.optimizer.load_state_dict(torch.load(optimizer_out_path))

  def _model_train_step(self, observations, actions_dist, results):
    observations = torch.tensor(observations, device=self.model.device)
    actions_dist = torch.tensor(actions_dist, device=self.model.device)
    results = torch.tensor(results, device=self.model.device)
    self.model.train()
    self.optimizer.zero_grad()
    values, log_policies = self.model(observations)
    # mean squared error
    values_loss = F.mse_loss(values.squeeze(1), results)
    # Kullback–Leibler divergence
    policies_loss = F.kl_div(log_policies, actions_dist, reduction="batchmean")
    (values_loss + policies_loss).backward()
    self.optimizer.step()
    return values_loss.item(), policies_loss.item()

  def train_step(self, game, search_iterations, batch_size, epochs, c_puct=1.0, dirichlet_alpha=None):
    first_person_result, game_buffer = self.selfplay(
      game, search_iterations, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
    )

    result = game.swap_result(first_person_result)
    while len(game_buffer) > 0:
      observation, action_dist = game_buffer.pop()
      self.replay_buffer.add_sample(observation, action_dist, result)
      result = game.swap_result(result)

    values_losses, policies_losses = [], []
    if len(self.replay_buffer) >= batch_size:
      for _ in range(epochs):
        observations, actions_dist, results = self.replay_buffer.sample(batch_size)
        values_loss, policies_loss = self._model_train_step(observations, actions_dist, results)
        values_losses.append(values_loss)
        policies_losses.append(policies_loss)

    return values_losses, policies_losses
