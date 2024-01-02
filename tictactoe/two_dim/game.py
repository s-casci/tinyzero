import numpy as np
import os
import sys

sys.path.append(os.getcwd())
from tictactoe.one_dim.game import TicTacToe as TicTacToe1D  # noqa: E402


class TicTacToe(TicTacToe1D):
  def __init__(self):
    super().__init__()

  def to_observation(self):
    obs = np.zeros((3, 3), dtype=np.float32)
    for i, x in enumerate(self.state):
      if x == self.turn:
        obs[i // 3, i % 3] = 1
      elif x == -self.turn:
        obs[i // 3, i % 3] = -1
    return np.array([obs])
