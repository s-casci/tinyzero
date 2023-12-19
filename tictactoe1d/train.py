from game import TicTacToe1D
from datetime import datetime
import torch
import wandb
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from models import LinearNetwork  # noqa: E402
from agents import AlphaZeroAgent  # noqa: E402

OUT_DIR = "tictactoe1d/out"
INIT_FROM_CHECKPOINT = False
SELFPLAY_GAMES = 5000
SELFPLAY_GAMES_PER_SAVE = SELFPLAY_GAMES // 4
BATCH_SIZE = 128
SEARCH_ITERATIONS = 32
MAX_REPLAY_BUFFER_SIZE = BATCH_SIZE * 4
TRAINING_EPOCHS = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-1
C_PUCT = 1.9
DIRICHLET_ALPHA = 0.3  # set to None to disable
WANDB_LOG = True
WANDB_PROJECT_NAME = "tinyalphazero-tictactoe1d"
WANDB_RUN_NAME = "run" + datetime.now().strftime("%Y%m%d-%H%M%S")

if __name__ == "__main__":
  game = TicTacToe1D()

  model = LinearNetwork(game.observation_shape, game.action_space)
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

  agent = AlphaZeroAgent(model, optimizer, MAX_REPLAY_BUFFER_SIZE)

  if INIT_FROM_CHECKPOINT:
    agent.load_training_state(f"{OUT_DIR}/model.pth", f"{OUT_DIR}/optimizer.pth")

  if WANDB_LOG:
    wandb_run = wandb.init(project=WANDB_PROJECT_NAME, name=WANDB_RUN_NAME)

  os.makedirs(OUT_DIR, exist_ok=True)
  print("Starting training")

  for i in tqdm(range(SELFPLAY_GAMES)):
    game.reset()

    values_losses, policies_losses = agent.train_step(
      game, SEARCH_ITERATIONS, BATCH_SIZE, TRAINING_EPOCHS, c_puct=C_PUCT, dirichlet_alpha=DIRICHLET_ALPHA
    )

    if WANDB_LOG:
      for values_loss, policies_loss in zip(values_losses, policies_losses):
        wandb.log({"values_loss": values_loss, "policies_loss": policies_loss})

    if i > 0 and i % SELFPLAY_GAMES_PER_SAVE == 0:
      print("Saving training state")
      agent.save_training_state(f"{OUT_DIR}/model.pth", f"{OUT_DIR}/optimizer.pth")

  if WANDB_LOG:
    wandb_run.finish()

  print("Training complete")

  print("Saving final training state")
  agent.save_training_state(f"{OUT_DIR}/model.pth", f"{OUT_DIR}/optimizer.pth")
