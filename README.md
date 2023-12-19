# tinyzero

<img src="https://github.com/s-casci/tinyzero/blob/main/tinyzero.png" width="480">

Train AlphaZero-like agents on any environment you want!

## Usage
Run `pip install requirements.txt` to make sure you have all the dependencies installed.

Then, to train an agent on one of the existing environments, run:
```bash
python3 tictactoe/train.py
```
where `tictactoe` is the name of the environment you want to train on.

Inside the train script, you can change some parameters, such as the number of episodes, the number of simulations and enable [wandb](https://wandb.ai/site) logging.

## Add an environment

To add a new environment, you can follow the `game.py` files in every existing examples.

The environment you add should implement the following methods:
- `reset()`: reset the environment to its initial state
- `step(action)`: take an action and modifies the state of the environment accordingly
- `get_legal_actions()`: return a list of legal actions
- `undo_last_action()`: undo the last action taken
- `to_observation()`: return the current state of the environment as an observation (a numpy array) to be used as input to the model
- `get_result()`: return the result of the game (for example, it might be 1 if the first player won, -1 if the second player won, 0 if it's a draw, and None if the game is not over yet)
- `get_first_person_result()`: return the result of the game from the perspective of the current player (for example, it might be 1 if the current player won, -1 if the opponent won, 0 if it's a draw, and None if the game is not over yet)
- `swap_result(result)`: swap the result of the game (for example, if the result is 1, it should become -1, and vice versa). It's needed to cover all of the possible game types (zero-sum, non-zero-sum, cooperative, etc.)

## Add a model

To add a new model, you can follow the existing example in `models.py`. The model you add should implement the `__call__` method, which takes as input an observation and return a value and a policy.

The AlphaZero agent computes the policy loss as the Kulback-Leibler divergence between the distribution produced by the model and the one given by the MCTS. Therefore, the policy returned by the model should be logaritmic.

## Add a new agent

Thanks to the way the value and policy functions are interpreted by the search tree, it's possible to use or train any agent that implements them.
