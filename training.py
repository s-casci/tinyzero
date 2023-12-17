from tinygrad import Tensor, nn, TinyJit


@TinyJit
def train_step_jitted(model, optimizer, observations, actions_dist, results):
  with Tensor.train():
    optimizer.zero_grad()
    values, log_policies = model(observations)
    # mean squared error
    values_loss = values.squeeze(1).sub(results).pow(2).mean()
    batch_size = observations.shape[0]
    # Kullback–Leibler divergence
    policies_loss = actions_dist.mul(actions_dist.log().sub(log_policies)).sum(axis=-1).sum() / batch_size
    (values_loss + policies_loss).backward()
    optimizer.step()
    return values_loss.realize(), policies_loss.realize()


def train_step(model, optimizer, observations, actions_dist, results):
  values_loss, policies_loss = train_step_jitted(
    model, optimizer, Tensor(observations), Tensor(actions_dist), Tensor(results)
  )
  return values_loss.item(), policies_loss.item()


def save_state(model, optimizer, model_out_path, optimizer_out_path):
  model_state_dict = nn.state.get_state_dict(model)
  nn.state.safe_save(model_state_dict, model_out_path)
  optimizer_state_dict = nn.state.get_state_dict(optimizer)
  nn.state.safe_save(optimizer_state_dict, optimizer_out_path)


def load_state(model, optimizer, model_out_path, optimizer_out_path):
  model_state_dict = nn.state.safe_load(model_out_path)
  nn.state.load_state_dict(model, model_state_dict)
  optimizer_state_dict = nn.state.safe_load(optimizer_out_path)
  nn.state.load_state_dict(optimizer, optimizer_state_dict)
