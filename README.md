# tinyzero

<img src="https://github.com/s-casci/tinyzero/blob/main/tinyzero.png" width="480">

Easily train AlphaZero-like agents on any environment you want!

## Usage
Make sure you have Python >= 3.8 intalled. After that, run `pip install -r requirements.txt` to install the necessary dependencies.

Then, to train an agent on one of the existing environments, run:
```bash
python3 tictactoe2d/train.py
```
where `tictactoe2d` is the name of the environment you want to train on.

Inside the train script, you can change some parameters, such as the number of episodes, the number of simulations and enable [wandb](https://wandb.ai/site) logging.

Similarly, to evaluate the trained agent run:
```bash
python3 tictactoe2d/eval.py
```

## Add an environment

To add a new environment, you can follow the `game.py` files in every existing examples.

The environment you add should implement the following methods:
- `reset()`: resets the environment to its initial state
- `step(action)`: takes an action and modifies the state of the environment accordingly
- `get_legal_actions()`: returns a list of legal actions
- `undo_last_action()`: cancels the last action taken
- `to_observation()`: returns the current state of the environment as an observation (a numpy array) to be used as input to the model
- `get_result()`: returns the result of the game (for example, it might be 1 if the first player won, -1 if the second player won, 0 if it's a draw, and None if the game is not over yet)
- `get_first_person_result()`: returns the result of the game from the perspective of the current player (for example, it might be 1 if the current player won, -1 if the opponent won, 0 if it's a draw, and None if the game is not over yet)
- `swap_result(result)`: swaps the result of the game (for example, if the result is 1, it should become -1, and vice versa). It's needed to cover all of the possible game types (single player, two players, zero-sum, non-zero-sum, etc.)

## Add a model

To add a new model, you can follow the existing examples in `models.py`.

The model you add should implement the following methods:
- `__call__`: takes as input an observation and returns a value and a policy
- `value_forward(observation)`: takes as input an observation and returns a value
- `policy_forward(observation)`: takes as input an observation and returns a distribution over the actions (the policy)

The latter two methods are used to speed up the MCTS.

The AlphaZero agent computes the policy loss as the Kulback-Leibler divergence between the distribution produced by the model and the one given by the MCTS. Therefore, the policy returned by the `__call__` method should be logaritmic. On the other hand, the policy returned by the `policy_forward` method should represent a probability distribution.

## Add a new agent

Thanks to the way the value and policy functions are interpreted by the search tree, it's possible to use or train any agent that implements them. To add a new agent, you can follow the existing example in `agents.py`.

The agent you add should implement the following methods:
- `value_fn(game)`: takes as input a game and returns a value (float)
- `policy_fn(game)`: takes as input a game and returns a policy (Numpy array)

Any other method is not directly used by the MCTS, so it's optional and depends on the agent you want to implement. For example, the `AlphaZeroAgent` implements a `train_step` method to train the model after each episode.
