import numpy as np
from mcts import search


def play(game, agent, search_iterations, c_puct=1.0, dirichlet_alpha=None):
  root = search(
    game, agent.value_fn, agent.policy_fn, search_iterations, c_puct=c_puct, dirichlet_alpha=dirichlet_alpha
  )
  return root.children_actions[np.argmax(root.children_visits)]


def pit(game, agent1, agent2, agent1_play_kwargs, agent2_play_kwargs):
  agent = [agent1, agent2]
  i = 0
  while (result := game.get_result()) is None:
    action = play(game, agent[i], **(agent1_play_kwargs if i == 0 else agent2_play_kwargs))
    game.step(action)
    i = 1 - i
  return result
