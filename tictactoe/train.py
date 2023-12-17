from game import TicTacToe
from tinygrad import nn
from datetime import datetime
import wandb
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from models import LinearNetwork  # noqa: E402
from training import save_state, load_state  # noqa: E402
from agents import AlphaZeroAgent  # noqa: E402

OUT_DIR = "tictactoe/out"
INIT_FROM_CHECKPOINT = False
SELFPLAY_GAMES = 1024
SELFPLAY_GAMES_PER_SAVE = SELFPLAY_GAMES // 4
BATCH_SIZE = 128
SEARCH_ITERATIONS = 32
MAX_REPLAY_BUFFER_SIZE = BATCH_SIZE * 4
TRAINING_EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
C_PUCT = 1.5
DIRICHLET_ALPHA = 0.3  # set to None to disable
WANDB_LOG = True
WANDB_PROJECT_NAME = "tinyalphazero-tictactoe"
WANDB_RUN_NAME = "run" + datetime.now().strftime("%Y%m%d-%H%M%S")

if __name__ == "__main__":
  game = TicTacToe()

  model = LinearNetwork(game.observation_shape, game.action_space)
  optimizer = nn.optim.AdamW(nn.state.get_parameters(model), lr=LEARNING_RATE, wd=WEIGHT_DECAY)

  if INIT_FROM_CHECKPOINT:
    load_state(model, optimizer, f"{OUT_DIR}/model.safetensors", f"{OUT_DIR}/optimizer.safetensors")

  agent = AlphaZeroAgent(model, optimizer, MAX_REPLAY_BUFFER_SIZE)

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
      print("Saving state")
      save_state(model, optimizer, f"{OUT_DIR}/model.safetensors", f"{OUT_DIR}/optimizer.safetensors")

  if WANDB_LOG:
    wandb_run.finish()

  print("Training complete")

  print("Saving final state")
  save_state(model, optimizer, f"{OUT_DIR}/model.safetensors", f"{OUT_DIR}/optimizer.safetensors")
