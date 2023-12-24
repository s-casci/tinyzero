from game import Connect2
import torch
from train import OUT_DIR, SEARCH_ITERATIONS
from tqdm import tqdm

import os
import sys

sys.path.append(os.getcwd())
from models import LinearNetwork  # noqa: E402
from agents import AlphaZeroAgent  # noqa: E402
from mcts import pit # noqa: E402

EVAL_GAMES = 100

if __name__ == "__main__":
  game = Connect2()

  model = LinearNetwork(game.observation_shape, game.action_space)
  model.load_state_dict(torch.load(f"{OUT_DIR}/model.pth"))

  agent = AlphaZeroAgent(model)
  agent_play_kwargs = {"search_iterations": SEARCH_ITERATIONS, "c_puct": 1.5, "dirichlet_alpha": 0.3}

  print(f"Playing {EVAL_GAMES} games against itself")

  results = {0: 0, 1: 0, -1: 0}
  for _ in tqdm(range(EVAL_GAMES)):
    result = pit(
      game,
      agent,
      agent,
      agent_play_kwargs,
      agent_play_kwargs,
    )
    results[result] += 1

  print("Results:")
  print(f"First player wins: {results[1]}")
  print(f"Second player wins: {results[-1]}")
  print(f"Draws: {results[0]}")
