# Copyright 2019 The Dreamer Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import functools
import os
import re

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

from dreamer import control
from dreamer import tools
from dreamer.training import trainer as trainer_


Objective = collections.namedtuple(
    'Objective', 'name, value, goal, include, exclude')


def LOG(x, m, *a):
  xf = tf.cast(x, tf.float32)
  print_op = tf.print(
      m, '(',
      tf.shape(xf),
      tf.reduce_min(xf),
      tf.reduce_mean(xf),
      tf.reduce_max(xf),
      ')', *a)
  with tf.control_dependencies([print_op]):
    return tf.identity(x)


def save_config(config, logdir=None):
  if logdir:
    with config.unlocked:
      config.logdir = logdir
    message = 'Start a new run and write summaries and checkpoints to {}.'
    print(message.format(config.logdir))
    tf.gfile.MakeDirs(config.logdir)
    config_path = os.path.join(config.logdir, 'config.yaml')
    with tf.gfile.GFile(config_path, 'w') as file_:
      yaml.dump(
          config, file_, yaml.Dumper,
          allow_unicode=True,
          default_flow_style=False)
  else:
    message = (
        'Start a new run without storing summaries and checkpoints since no '
        'logging directory was specified.')
    print(message)
  return config


def load_config(logdir):
  config_path = logdir and os.path.join(logdir, 'config.yaml')
  if not config_path or not tf.gfile.Exists(config_path):
    message = (
        'Cannot resume an existing run since the logging directory does not '
        'contain a configuration file.')
    raise IOError(message)
  with tf.gfile.GFile(config_path, 'r') as file_:
    config = yaml.load(file_, yaml.Loader)
    message = 'Resume run and write summaries and checkpoints to {}.'
    print(message.format(config.logdir))
  return config


def train(model_fn, datasets, logdir, config):
  if not config:
    raise KeyError('You must specify a configuration.')
  logdir = logdir and os.path.expanduser(logdir)
  try:
    config = load_config(logdir)
  except RuntimeError:
    print('Failed to load existing config.')
  except IOError:
    config = save_config(config, logdir)
  with datasets.unlocked:
    datasets.train = datasets.train.make_one_shot_iterator()
    datasets.test = datasets.test.make_one_shot_iterator()
  trainer = trainer_.Trainer(logdir, config=config)
  cleanups = []
  try:
    with tf.variable_scope('graph', use_resource=True):
      data = tf.cond(
          tf.equal(trainer.phase, 'train'),
          datasets.train.get_next, datasets.test.get_next)
      score, summary, cleanups = model_fn(data, trainer, config)
      message = 'Graph contains {} trainable variables.'
      print(message.format(tools.count_weights()))
      if config.train_steps:
        trainer.add_phase(
            'train', config.train_steps, score, summary,
            batch_size=config.batch_shape[0],
            report_every=None,
            log_every=config.train_log_every,
            checkpoint_every=config.train_checkpoint_every)
      if config.test_steps:
        trainer.add_phase(
            'test', config.test_steps, score, summary,
            batch_size=config.batch_shape[0],
            report_every=config.test_steps,
            log_every=config.test_steps,
            checkpoint_every=config.test_checkpoint_every)
    for saver in config.savers:
      trainer.add_saver(**saver)
    for score in trainer.iterate(config.max_steps):
      yield score
  finally:
    for cleanup in cleanups:
      cleanup()


def compute_objectives(posterior, prior, target, graph, config):
  raw_features = graph.cell.features_from_state(posterior)
  heads = graph.heads
  if config.imagination_horizon:
    imagination_start = posterior
    if config.imagination_skip_last:
      imagination_start = tools.nested.map(
          lambda x: x[:, :-config.imagination_skip_last], imagination_start)
    raw_states = imagine_forward(
        imagination_start, config.imagination_horizon, graph, config,
        graph.heads.action, stop_grad_post_action=False,
        stop_grad_pre_action=config.stop_grad_pre_action)
  else:
    raw_states = None
  objectives = []
  for name, scale in sorted(config.loss_scales.items(), key=lambda x: x[0]):
    if config.loss_scales[name] == 0.0:
      continue
    if name in config.heads and name not in config.gradient_heads:
      features = tf.stop_gradient(raw_features)
      include = r'.*/head_{}/.*'.format(name)
      exclude = None
    else:
      features = raw_features
      include = None
      exclude = None

    if name == 'divergence':
      loss = graph.cell.divergence_from_states(posterior, prior)
      if config.free_nats is not None:
        loss = tf.maximum(0.0, loss - float(config.free_nats))
      objectives.append(Objective('divergence', loss, min, include, exclude))

    elif name == 'cpc':
      pred = heads.cpc(graph.embedded)
      objective = compute_cpc_loss(pred, features, config)
      objectives.append(Objective('cpc', objective, max, include, exclude))

    elif name == 'overshooting':
      shape = tools.shape(graph.data['action'])
      length = tf.tile(tf.constant(shape[1])[None], [shape[0]])
      _, priors, posteriors, mask = tools.overshooting(
          graph.cell, {}, graph.embedded, graph.data['action'], length,
          config.overshooting_distance, posterior)
      posteriors, priors, mask = tools.nested.map(
          lambda x: x[:, :, 1:-1], (posteriors, priors, mask))
      if config.os_stop_posterior_grad:
        posteriors = tools.nested.map(tf.stop_gradient, posteriors)
      loss = graph.cell.divergence_from_states(posteriors, priors)
      if config.free_nats is not None:
        loss = tf.maximum(0.0, loss - float(config.free_nats))
      objectives.append(Objective('overshooting', loss, min, include, exclude))

    elif name == 'value':
      if config.value_source == 'dataset':
        loss = compute_value_loss(
            config, graph, priors, features, target['reward'])
      elif config.value_source == 'model':
        if 'action_target' in graph.heads or not config.imagination_horizon:
          if 'action_target' in graph.heads:
            policy = graph.heads.action_target
          else:
            policy = graph.heads.action
          states = imagine_forward(
              posterior, config.value_model_horizon, graph, config, policy)
        else:
          states = raw_states
        feat = graph.cell.features_from_state(states)
        loss = compute_value_loss(config, graph, states, feat, None)
      else:
        raise NotImplementedError(config.value_source)
      objectives.append(Objective('value', loss, min, include, exclude))

    elif name == 'action':
      if config.action_source == 'model':
        if not config.imagination_horizon:
          states = imagine_forward(
              posterior, config.action_model_horizon, graph, config,
              policy=graph.heads.action, stop_grad_post_action=False)
        else:
          states = raw_states
        feat = graph.cell.features_from_state(states)
        objective = compute_action_values(config, graph, states, feat)
        objectives.append(Objective(
            'action', objective, max, include, exclude))
      elif config.action_source == 'dataset':
        objective = heads.action(features).log_prob(target[name])
        objective -= compute_action_divergence(features, graph, config)
        objectives.append(Objective(
            'action', objective, max, include, exclude))
      else:
        raise NotImplementedError(config.action_source)

    elif name == 'reward':
      reward_mask = tf.squeeze(target['reward_mask'], [-1])
      logprob = heads.reward(features).log_prob(target[name]) * reward_mask
      objectives.append(Objective('reward', logprob, max, include, exclude))

    elif name == 'pcont' and config.pcont_label_weight:
      terminal = tf.cast(tf.less(target[name], 0.5), tf.float32)
      logprob = heads[name](features).log_prob(target[name])
      logprob *= 1 + terminal * (config.pcont_label_weight - 1)
      objectives.append(Objective(name, logprob, max, include, exclude))

    else:
      logprob = heads[name](features).log_prob(target[name])
      objectives.append(Objective(name, logprob, max, include, exclude))

  objectives = [o._replace(value=tf.reduce_mean(o.value)) for o in objectives]
  return objectives


def imagine_forward(
    initial_state, distance, graph, config, policy,
    stop_grad_post_action=True, stop_grad_pre_action=True):
  extended_batch = np.prod(tools.shape(
      tools.nested.flatten(initial_state)[0])[:2])
  obs = tf.zeros([extended_batch] + list(graph.embedded.shape[2:]))
  use_obs = tf.zeros([extended_batch, 1], tf.bool)
  new_shape = lambda t: [
      tf.reduce_prod(tools.shape(t)[:2])] + tools.shape(t)[2:]
  initial_state = tools.nested.map(
      lambda tensor: tf.reshape(tensor, new_shape(tensor)),
      initial_state)
  def step_fn(prev, index):
    feature = graph.cell.features_from_state(prev)
    if stop_grad_pre_action:
      feature = tf.stop_gradient(feature)
    action = policy(feature).sample()
    if stop_grad_post_action:
      action = tf.stop_gradient(action)
    (_, state), _ = graph.cell((obs, action, use_obs), prev)
    return state
  states = tf.scan(step_fn, tf.range(distance), initial_state, back_prop=True)
  states = tools.nested.map(lambda x: tf.transpose(x, [1, 0, 2]), states)
  return states


def compute_value_loss(config, graph, states, features, reward):
  if reward is None:
    reward = graph.heads.reward(features).mode()
  if config.value_maxent:
    reward -= compute_action_divergence(features, graph, config)
    reward -= compute_state_divergence(states, graph, config)
  if config.value_loss_pcont:
    pcont = tf.stop_gradient(graph.heads.pcont(features).mean())
  else:
    pcont = tf.ones_like(reward)
  pcont *= config.value_discount
  pred = graph.heads.value(features)
  if 'value_target' in graph.heads:
    value = graph.heads.value_target(features).mode()
  else:
    value = pred.mode()
  bootstrap = None
  if config.value_bootstrap:
    reward = reward[:, :-1]
    value = value[:, :-1]
    pcont = pcont[:, :-1]
    bootstrap = value[:, -1]
  return_ = control.lambda_return(
      reward, value, bootstrap, pcont,
      config.value_lambda, axis=1, stop_gradient=True)
  return_ = tf.concat([return_, tf.zeros_like(return_[:, -1:])], 1)
  loss = -pred.log_prob(return_)[:, :-1]
  if config.value_pcont_weight:
    loss *= tf.stop_gradient(tf.cumprod(tf.concat([
        tf.ones_like(pcont[:, :1]), pcont[:, :-1]], 1), 1))
  return loss


def compute_action_values(config, graph, states, features):
  reward = graph.heads.reward(features).mode()
  reward -= compute_action_divergence(features, graph, config)
  reward -= compute_state_divergence(states, graph, config)
  if config.action_loss_pcont:
    pcont = graph.heads.pcont(features).mean()
    if config.action_pcont_stop_grad:
      pcont = tf.stop_gradient(pcont)
  else:
    pcont = tf.ones_like(reward)
  pcont *= config.action_discount
  if 'value' not in graph.heads:
    return control.discounted_return(
        reward, pcont, bootstrap=None, axis=1, stop_gradient=False)
  value = graph.heads.value(features).mode()
  bootstrap = None
  if config.action_bootstrap:
    reward = reward[:, :-1]
    value = value[:, :-1]
    pcont = pcont[:, :-1]
    bootstrap = value[:, -1]
  return_ = control.lambda_return(
      reward, value, bootstrap, pcont,
      config.action_lambda,
      axis=1, stop_gradient=False)
  if config.action_pcont_weight:
    return_ *= tf.stop_gradient(tf.cumprod(tf.concat([
        tf.ones_like(pcont[:, :1]), pcont[:, :-1]], 1), 1))
  return return_


def compute_action_divergence(features, graph, config):
  features = tf.stop_gradient(features)
  pred = graph.heads.action(features)
  if not config.action_beta:
    return 0.0
  try:
    amount = -pred.entropy()
  except NotImplementedError:
    samples = pred.sample(100)
    amount = tf.reduce_mean(pred.log_prob(samples), 0)
  amount *= config.action_beta
  value = config.action_beta_dims_value
  if value and value < 0:
    amount /= value * float(pred.event_shape[-1].value)
  if value and value > 0:
    amount *= value * float(pred.event_shape[-1].value)
  return amount


def compute_state_divergence(states, graph, config):
  pred = graph.cell.dist_from_state(states)
  if not config.state_beta:
    return 0.0
  try:
    amount = -pred.entropy()
  except NotImplementedError:
    samples = pred.sample(100)
    amount = tf.reduce_mean(pred.log_prob(samples), 0)
  amount *= config.state_beta
  return amount


def compute_cpc_loss(pred, features, config):
  if config.cpc_contrast == 'batch':
    ta = tf.TensorArray(tf.float32, 0, True, element_shape=[None, None])
    _, _, ta = tf.while_loop(
        lambda i, f, ta: tf.less(i, tf.shape(f)[0]),
        lambda i, f, ta: (
            i + 1, f, ta.write(ta.size(), pred.log_prob(tf.roll(f, i, 0)))),
        (0, features, ta), back_prop=True, swap_memory=True)
    positive = pred.log_prob(features)
    negative = tf.reduce_logsumexp(ta.stack(), 0)
    return positive - negative
  elif config.cpc_contrast == 'time':
    ta = tf.TensorArray(tf.float32, 0, True, element_shape=[None, None])
    _, _, ta = tf.while_loop(
        lambda i, f, ta: tf.less(i, tf.shape(f)[1]),
        lambda i, f, ta: (
            i + 1, f, ta.write(ta.size(), pred.log_prob(tf.roll(f, i, 1)))),
        (0, features, ta), back_prop=True, swap_memory=True)
    positive = pred.log_prob(features)
    negative = tf.reduce_logsumexp(ta.stack(), 0)
    return positive - negative
  elif config.cpc_contrast == 'window':
    assert config.cpc_batch_amount <= config.batch_shape[0]
    assert config.cpc_time_amount <= config.batch_shape[1]
    total_amount = config.cpc_batch_amount * config.cpc_time_amount
    ta = tf.TensorArray(tf.float32, 0, True, element_shape=[None, None])
    def compute_negatives(index, ta):
      batch_shift = tf.math.floordiv(index, config.cpc_time_amount)
      time_shift = tf.mod(index, config.cpc_time_amount)
      batch_shift -= config.cpc_batch_amount // 2
      time_shift -= config.cpc_time_amount // 2
      rolled = tf.roll(tf.roll(features, batch_shift, 0), time_shift, 1)
      return ta.write(ta.size(), pred.log_prob(rolled))
    _, ta = tf.while_loop(
        lambda index, ta: tf.less(index, total_amount),
        lambda index, ta: (index + 1, compute_negatives(index, ta)),
        (0, ta), back_prop=True, swap_memory=True)
    positive = pred.log_prob(features)
    negative = tf.reduce_logsumexp(ta.stack(), 0)
    return positive - negative
  else:
    raise NotImplementedError(config.cpc_contrast)


def apply_optimizers(objectives, trainer, config):
  # Make sure all losses are computed and apply loss scales.
  processed = []
  values = [ob.value for ob in objectives]
  for ob in objectives:
    loss = {min: ob.value, max: -ob.value}[ob.goal]
    loss *= config.loss_scales[ob.name]
    with tf.control_dependencies(values):
      loss = tf.identity(loss)
    processed.append(ob._replace(value=loss, goal=min))
  # Merge objectives that operate on the whole model to compute only one
  # backward pass and to share optimizer statistics.
  objectives = []
  losses = []
  for ob in processed:
    if ob.include is None and ob.exclude is None:
      assert ob.goal == min
      losses.append(ob.value)
    else:
      objectives.append(ob)
  loss = tf.reduce_sum(losses)
  # modules = ['encoder', 'rnn']
  # modules += ['head_{}'.format(name) for name in config.gradient_heads]
  # include = r'graph/({})/.*'.format('|'.join(modules))
  include = r'.*'
  objectives.append(Objective('model', loss, min, include, None))
  # Apply optimizers and collect loss summaries.
  summaries = []
  grad_norms = {}
  for ob in objectives:
    assert ob.name in list(config.loss_scales.keys()) + ['model'], ob
    assert ob.goal == min, ob
    if ob.name not in config.optimizers:
      with config.optimizers.unlocked:
        config.optimizers[ob.name] = config.optimizers.default.copy()
    optimizer = config.optimizers[ob.name](
        include=ob.include,
        exclude=ob.exclude,
        step=trainer.step,
        log=trainer.log,
        debug=config.debug,
        name=ob.name)
    condition = tf.equal(trainer.phase, 'train')
    summary, grad_norm = optimizer.maybe_minimize(condition, ob.value)
    summaries.append(summary)
    grad_norms[ob.name] = grad_norm
  return summaries, grad_norms


def simulate(metrics, config, params, graph, cleanups, gif_summary, name):
  def env_ctor():
    env = params.task.env_ctor()
    if params.save_episode_dir:
      env = control.wrappers.CollectDataset(env, params.save_episode_dir)
    return env
  bind_or_none = lambda x, **kw: x and functools.partial(x, **kw)
  cell = graph.cell
  agent_config = tools.AttrDict(
      cell=cell,
      encoder=graph.encoder,
      planner=functools.partial(params.planner, graph=graph),
      objective=bind_or_none(params.objective, graph=graph),
      exploration=params.exploration,
      preprocess_fn=config.preprocess_fn,
      postprocess_fn=config.postprocess_fn)
  params = params.copy()
  with params.unlocked:
    params.update(agent_config)
  with agent_config.unlocked:
    agent_config.update(params)
  with tf.variable_scope(name):
    summaries = []
    env = control.create_batch_env(
        env_ctor, params.num_envs, config.isolate_envs)
    agent = control.MPCAgent(env, graph.step, False, False, agent_config)
    cleanup = lambda: env.close()
    scores, lengths, data = control.simulate(
        agent, env, params.num_episodes, params.num_steps)
    summaries.append(tf.summary.scalar('return', scores[0]))
    summaries.append(tf.summary.scalar('length', lengths[0]))
    if gif_summary:
      summaries.append(tools.gif_summary(
          'gif', data['image'], max_outputs=1, fps=20))
    write_metrics = [
        metrics.add_scalars(name + '/return', scores),
        metrics.add_scalars(name + '/length', lengths),
        # metrics.add_tensor(name + '/frames', data['image']),
    ]
    with tf.control_dependencies(write_metrics):
      summary = tf.summary.merge(summaries)
  cleanups.append(cleanup)  # Work around tf.cond() tensor return type.
  return summary, tf.reduce_mean(scores)


def print_metrics(metrics, step, every, decimals=2, name='metrics'):
  factor = 10 ** decimals
  means, updates = [], []
  for key, value in metrics.items():
    key = 'metrics_{}_{}'.format(name, key)
    mean = tools.StreamingMean((), tf.float32, key)
    means.append(mean)
    updates.append(mean.submit(value))
  with tf.control_dependencies(updates):
    message = '{}: step/{} ='.format(name, '/'.join(metrics.keys()))
    print_metrics = tf.cond(
        tf.equal(step % every, 0),
        lambda: tf.print(message, [step] + [
            tf.round(mean.clear() * factor) / factor for mean in means]),
        tf.no_op)
  return print_metrics


def collect_initial_episodes(metrics, config):
  items = config.random_collects.items()
  items = sorted(items, key=lambda x: x[0])
  for name, params in items:
    metrics.set_tags(
        global_step=0,
        step=0,
        phase=name.split('-')[0],
        time=datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S'))
    tf.gfile.MakeDirs(params.save_episode_dir)
    # Compute number of missing episodes and steps.
    filenames = tf.gfile.Glob(os.path.join(params.save_episode_dir, '*.npz'))
    num_episodes = len(filenames)
    # num_steps = sum([
    #     int(f.rsplit('.', 1)[0].split('-')[2]) for f in filenames])
    num_steps = sum([
        int(re.search(r'-([0-9]+)\.npz', f).group(1)) for f in filenames])
    remaining_episodes = params.num_episodes - num_episodes
    remaining_steps = params.num_steps - num_steps
    if remaining_episodes <= 0 and remaining_steps <= 0:
      continue
    # User message.
    if params.give_rewards:
      env_ctor = params.task.env_ctor
      word = 'with'
    else:
      env_ctor = functools.partial(
          lambda ctor: control.wrappers.NoRewardHint(ctor()),
          params.task.env_ctor)
      word = 'without'
    message = 'Collecting initial {} episodes or {} steps ({}) {} rewards.'
    print(message.format(remaining_episodes, remaining_steps, name, word))
    episodes = control.random_episodes(
        env_ctor, remaining_episodes, remaining_steps,
        params.save_episode_dir, config.isolate_envs)
    scores = [episode['reward'].sum() for episode in episodes]
    lengths = [len(episode['reward']) for episode in episodes]
    metrics.add_scalars(name + '/return', scores)
    metrics.add_scalars(name + '/length', lengths)
  metrics.flush()
