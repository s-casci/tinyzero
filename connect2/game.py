import numpy as np


class Connect2:
  STATE_LEN = 4

  def __init__(self):
    self.reset()

    self.observation_shape = self.to_observation().shape
    self.action_space = self.STATE_LEN

  def reset(self):
    self.state = [0] * self.STATE_LEN
    self.actions_stack = []
    self.turn = 1

  def __str__(self):
    return str(self.state)

  def to_observation(self):
    obs = np.zeros(self.STATE_LEN, dtype=np.float32)
    for i, x in enumerate(self.state):
      if x == self.turn:
        obs[i] = 1
      elif x == -self.turn:
        obs[i] = -1
    return obs

  def get_legal_actions(self):
    return [i for i, x in enumerate(self.state) if x == 0]

  def step(self, action):
    if self.state[action] != 0:
      raise ValueError(f"Action {action} is illegal")
    self.state[action] = self.turn
    self.actions_stack.append(action)
    self.turn *= -1

  def undo_last_action(self):
    self.state[self.actions_stack.pop()] = 0
    self.turn *= -1

  def get_result(self):
    for x, y in zip(self.state[:-1], self.state[1:]):
      if x == y != 0:
        return x
    if len(self.get_legal_actions()) == 0:
      return 0

  # get result from the point of view of the current player
  def get_first_person_result(self):
    result = self.get_result()
    if result is not None:
      return result * self.turn

  @staticmethod
  def swap_result(result):
    return -result
