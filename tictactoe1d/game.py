import numpy as np


class TicTacToe1D:
  def __init__(self):
    self.reset()

    self.observation_shape = self.to_observation().shape
    self.action_space = 9

  def reset(self):
    self.state = [0] * 9
    self.actions = []
    self.turn = 1

  def __str__(self):
    return "\n".join(["  ".join([str(x) for x in self.state[i : i + 3]]) for i in range(0, 9, 3)])

  def get_legal_actions(self):
    return [i for i, x in enumerate(self.state) if x == 0]

  def step(self, action):
    assert self.state[action] == 0
    self.state[action] = self.turn
    self.actions.append(action)
    self.turn *= -1

  def undo_last_action(self):
    self.state[self.actions.pop()] = 0
    self.turn *= -1

  def get_result(self):
    if len(self.actions) < 5:
      return
    for x, y, z in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
      if self.state[x] == self.state[y] == self.state[z] != 0:
        return self.state[x]
    if len(self.get_legal_actions()) == 0:
      return 0

  def get_first_person_result(self):
    result = self.get_result()
    if result is not None:
      return result * self.turn

  @staticmethod
  def swap_result(result):
    return -result

  def to_observation(self):
    obs = np.zeros(9, dtype=np.float32)
    for i, x in enumerate(self.state):
      if x == self.turn:
        obs[i] = 1
      elif x == -self.turn:
        obs[i] = -1
    return obs
