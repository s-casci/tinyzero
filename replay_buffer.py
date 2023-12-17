from collections import deque
import numpy as np


class ReplayBuffer:
  def __init__(self, max_size):
    self.observations = deque(maxlen=max_size)
    self.actions_dist = deque(maxlen=max_size)
    self.results = deque(maxlen=max_size)

  def __len__(self):
    return len(self.observations)

  def add_sample(self, observation, actions_dist, result):
    self.observations.append(observation)
    self.actions_dist.append(actions_dist)
    self.results.append(result)

  def sample(self, batch_size):
    indices = np.random.choice(len(self), batch_size, replace=False)
    observations = np.array([self.observations[i] for i in indices], dtype=np.float32)
    actions_dist = np.array([self.actions_dist[i] for i in indices], dtype=np.float32)
    # add a small value to avoid log(0)
    actions_dist += 1e-8
    results = np.array([self.results[i] for i in indices], dtype=np.float32)
    return observations, actions_dist, results
