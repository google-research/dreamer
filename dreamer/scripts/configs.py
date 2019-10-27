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

import os

import tensorflow as tf

from dreamer import control
from dreamer import models
from dreamer import networks
from dreamer import tools
from dreamer.scripts import tasks as tasks_lib
from dreamer.scripts import objectives as objectives_lib


ACTIVATIONS = {
    'relu': tf.nn.relu,
    'elu': tf.nn.elu,
    'tanh': tf.tanh,
    'swish': tf.nn.swish,
    'softplus': tf.nn.softplus,
    'none': None,
}


DEFAULTS = dict(
    dreamer=dict(
        train_planner='policy_sample',
        test_planner='policy_mode',
        planner_objective='reward_value',
        action_head=True,
        value_head=True,
        train_action_noise=0.3,
    ),
    actor=dict(
        train_planner='policy_sample',
        test_planner='policy_mode',
        planner_objective='reward',
        action_head=True,
        value_head=False,
        train_action_noise=0.3,
    ),
    planet=dict(
        train_planner='cem',
        test_planner='cem',
        planner_objective='reward',
        action_head=False,
        value_head=False,
        imagination_horizon=0,
        train_action_noise=0.3,
    ),
    debug=dict(
        debug=True,
        action_repeat=50,
        num_seed_episodes=1,
        num_seed_steps=1,
        collect_steps=1,
        train_steps=10,
        test_steps=10,
        max_steps=500,
        train_collects=[dict(steps_after=10, steps_every=10)],
        test_collects=[dict(steps_after=10, steps_every=10)],
        model_size=10,
        state_size=5,
        num_layers=1,
        num_units=10,
        batch_shape=[5, 10],
        loader_every=5,
        loader_window=2,
    ),
    pcont=dict(
        gradient_heads=['image', 'reward', 'pcont'],
        imagination_skip_last=1,
        value_loss_pcont=True,
        action_loss_pcont=True,
        action_discount=1.0,
        value_discount=1.0,
    ),
    discrete=dict(
        action_head_dist='onehot_score',
        imagination_horizon=10,
        divergence_scale=0.1,
        clip_rewards='tanh',
        action_noise_type='epsilon_greedy',
        train_action_noise=0.4,
        train_action_noise_ramp=-10000000,
        train_action_noise_min=0.25,
        test_action_noise=0.001,
    ),
    atari=dict(
        max_length=27000,
        atari_lifes=True,
    ),
)


def make_config(params):
  config = tools.AttrDict()
  config.debug = params.get('debug', False)
  with params.unlocked:
    for name in params.get('defaults', ['dreamer']):
      for key, value in DEFAULTS[name].items():
        if key not in params:
          params[key] = value
  config.loss_scales = tools.AttrDict()
  config = _data_processing(config, params)
  config = _model_components(config, params)
  config = _tasks(config, params)
  config = _loss_functions(config, params)
  config = _training_schedule(config, params)
  # Mark params as used which are only accessed at run-time.
  run_time_keys = [
      'planner_discount', 'planner_lambda', 'objective_entropy_scale',
      'normalize_actions', 'max_length', 'render_size', 'atari_lifes',
      'atari_noops', 'atari_sticky', 'atari_train_max_length',
      'atari_grayscale']
  for key in run_time_keys:
    params.get(key, None)
  if params.untouched:
    message = 'Found unused config overrides: {}'
    raise KeyError(message.format(', '.join(params.untouched)))
  return config


def _data_processing(config, params):
  config.batch_shape = params.get('batch_shape', (50, 50))
  config.num_chunks = params.get('num_chunks', 1)
  image_bits = params.get('image_bits', 8)
  config.preprocess_fn = tools.bind(
      tools.preprocess.preprocess, bits=image_bits)
  config.postprocess_fn = tools.bind(
      tools.preprocess.postprocess, bits=image_bits)
  config.open_loop_context = 5
  config.data_reader = tools.bind(
      tools.numpy_episodes.episode_reader,
      clip_rewards=params.get('clip_rewards', False),
      pcont_scale=params.get('pcont_scale', 0.99))
  config.data_loader = {
      'cache': tools.bind(
          tools.numpy_episodes.cache_loader,
          every=params.get('loader_every', 1000)),
      'recent': tools.bind(
          tools.numpy_episodes.recent_loader,
          every=params.get('loader_every', 1000)),
      'window': tools.bind(
          tools.numpy_episodes.window_loader,
          window=params.get('loader_window', 400),
          every=params.get('loader_every', 1000)),
      'reload': tools.numpy_episodes.reload_loader,
      'dummy': tools.numpy_episodes.dummy_loader,
  }[params.get('loader', 'cache')]
  config.gpu_prefetch = params.get('gpu_prefetch', False)
  return config


def _model_components(config, params):
  config.gradient_heads = params.get('gradient_heads', ['image', 'reward'])
  config.activation = ACTIVATIONS[params.get('activation', 'elu')]
  config.num_layers = params.get('num_layers', 3)
  config.num_units = params.get('num_units', 400)
  encoder = params.get('encoder', 'conv')
  if encoder == 'conv':
    config.encoder = networks.conv.encoder
  elif encoder == 'proprio':
    config.encoder = tools.bind(
        networks.proprio.encoder,
        keys=params.get('proprio_encoder_keys'),
        num_layers=params.get('proprio_encoder_num_layers', 3),
        units=params.get('proprio_encoder_units', 300))
  else:
    raise NotImplementedError(encoder)
  config.head_network = tools.bind(
      networks.feed_forward,
      num_layers=config.num_layers,
      units=config.num_units,
      activation=config.activation)
  config.heads = tools.AttrDict()
  if params.get('value_head', True):
    config.heads.value = tools.bind(
        config.head_network,
        num_layers=params.get('value_layers', 3),
        data_shape=[],
        dist=params.get('value_dist', 'normal'))
  if params.get('value_target_head', False):
    config.heads.value_target = tools.bind(
        config.head_network,
        num_layers=params.get('value_layers', 3),
        data_shape=[],
        stop_gradient=True,
        dist=params.get('value_dist', 'normal'))
  if params.get('return_head', False):
    config.heads['return'] = tools.bind(
        config.head_network,
        activation=config.activation)
  if params.get('action_head', True):
    config.heads.action = tools.bind(
        config.head_network,
        num_layers=params.get('action_layers', 4),
        mean_activation=ACTIVATIONS[
            params.get('action_mean_activation', 'none')],
        dist=params.get('action_head_dist', 'tanh_normal_tanh'),
        std=params.get('action_head_std', 'learned'),
        min_std=params.get('action_head_min_std', 1e-4),
        init_std=params.get('action_head_init_std', 5.0))
  if params.get('action_target_head', False):
    config.heads.action_target = tools.bind(
        config.head_network,
        num_layers=params.get('action_layers', 4),
        stop_gradient=True,
        mean_activation=ACTIVATIONS[
            params.get('action_mean_activation', 'none')],
        dist=params.get('action_head_dist', 'tanh_normal_tanh'),
        std=params.get('action_head_std', 'learned'),
        min_std=params.get('action_head_min_std', 1e-4),
        init_std=params.get('action_head_init_std', 5.0))
  if params.get('cpc_head', False):
    config.heads.cpc = config.head_network.copy(
        dist=params.get('cpc_head_dist', 'normal'),
        std=params.get('cpc_head_std', 'learned'),
        num_layers=params.get('cpc_head_layers', 3))
  image_head = params.get('image_head', 'conv')
  if image_head == 'conv':
    config.heads.image = tools.bind(
        networks.conv.decoder,
        std=params.get('image_head_std', 1.0))
  else:
    raise NotImplementedError(image_head)
  hidden_size = params.get('model_size', 200)
  state_size = params.get('state_size', 30)
  model = params.get('model', 'rssm')
  if model == 'rssm':
    config.cell = tools.bind(
        models.RSSM, state_size, hidden_size, hidden_size,
        params.get('future_rnn', True),
        params.get('mean_only', False),
        params.get('min_stddev', 1e-1),
        config.activation,
        params.get('model_layers', 1))
  else:
    raise NotImplementedError(model)
  return config


def _tasks(config, params):
  config.isolate_envs = params.get('isolate_envs', 'thread')
  train_tasks, test_tasks = [], []
  for name in params.get('tasks', ['cheetah_run']):
    try:
      train_tasks.append(getattr(tasks_lib, name)(config, params, 'train'))
      test_tasks.append(getattr(tasks_lib, name)(config, params, 'test'))
    except TypeError:
      train_tasks.append(getattr(tasks_lib, name)(config, params))
      test_tasks.append(getattr(tasks_lib, name)(config, params))
  def common_spaces_ctor(task, action_spaces):
    env = task.env_ctor()
    env = control.wrappers.SelectObservations(env, ['image'])
    env = control.wrappers.PadActions(env, action_spaces)
    return env
  if len(train_tasks) > 1:
    action_spaces = [task.env_ctor().action_space for task in train_tasks]
    for index, task in enumerate(train_tasks):
      env_ctor = tools.bind(common_spaces_ctor, task, action_spaces)
      train_tasks[index] = tasks_lib.Task(task.name, env_ctor, [])
  if len(test_tasks) > 1:
    action_spaces = [task.env_ctor().action_space for task in test_tasks]
    for index, task in enumerate(test_tasks):
      env_ctor = tools.bind(common_spaces_ctor, task, action_spaces)
      test_tasks[index] = tasks_lib.Task(task.name, env_ctor, [])
  if config.gradient_heads == 'all_but_image':
    config.gradient_heads = train_tasks[0].state_components
  diags = params.get('state_diagnostics', True)
  for name in train_tasks[0].state_components + ['reward', 'pcont']:
    if name not in config.gradient_heads + ['reward', 'pcont'] and not diags:
      continue
    kwargs = {}
    kwargs['stop_gradient'] = name not in config.gradient_heads
    if name == 'pcont':
      kwargs['dist'] = 'binary'
    default = dict(reward=2).get(name, config.num_layers)
    kwargs['num_layers'] = params.get(name + '_layers', default)
    config.heads[name] = tools.bind(config.head_network, **kwargs)
    config.loss_scales[name] = 1.0
  config.train_tasks = train_tasks
  config.test_tasks = test_tasks
  return config


def _loss_functions(config, params):
  for head in config.gradient_heads:
    assert head in config.heads, head
  config.imagination_horizon = params.get('imagination_horizon', 15)
  config.imagination_skip_last = params.get('imagination_skip_last', None)
  config.imagination_include_initial = params.get(
      'imagination_include_initial', True)

  config.action_source = params.get('action_source', 'model')
  config.action_model_horizon = params.get('action_model_horizon', None)
  config.action_bootstrap = params.get('action_bootstrap', True)
  config.action_discount = params.get('action_discount', 0.99)
  config.action_lambda = params.get('action_lambda', 0.95)
  config.action_target_update = params.get('action_target_update', 1)
  config.action_target_period = params.get('action_target_period', 50000)
  config.action_loss_pcont = params.get('action_loss_pcont', False)
  config.action_pcont_stop_grad = params.get('action_pcont_stop_grad', False)
  config.action_pcont_weight = params.get('action_pcont_weight', True)

  config.value_source = params.get('value_source', 'model')
  config.value_model_horizon = params.get('value_model_horizon', None)
  config.value_discount = params.get('value_discount', 0.99)
  config.value_lambda = params.get('value_lambda', 0.95)
  config.value_bootstrap = params.get('value_bootstrap', True)
  config.value_target_update = params.get('value_target_update', 1)
  config.value_target_period = params.get('value_target_period', 50000)
  config.value_loss_pcont = params.get('value_loss_pcont', False)
  config.value_pcont_weight = params.get('value_pcont_weight', True)
  config.value_maxent = params.get('value_maxent', False)

  config.action_beta = params.get('action_beta', 0.0)
  config.action_beta_dims_value = params.get('action_beta_dims_value', None)
  config.state_beta = params.get('state_beta', 0.0)
  config.stop_grad_pre_action = params.get('stop_grad_pre_action', True)
  config.pcont_label_weight = params.get('pcont_label_weight', None)

  config.loss_scales.divergence = params.get('divergence_scale', 1.0)
  config.loss_scales.global_divergence = params.get('global_div_scale', 0.0)
  config.loss_scales.overshooting = params.get('overshooting_scale', 0.0)
  for head in config.heads:
    if head in ('value_target', 'action_target'):  # Untrained.
      continue
    config.loss_scales[head] = params.get(head + '_loss_scale', 1.0)

  config.free_nats = params.get('free_nats', 3.0)
  config.overshooting_distance = params.get('overshooting_distance', 0)
  config.os_stop_posterior_grad = params.get('os_stop_posterior_grad', True)
  config.cpc_contrast = params.get('cpc_contrast', 'window')
  config.cpc_batch_amount = params.get('cpc_batch_amount', 10)
  config.cpc_time_amount = params.get('cpc_time_amount', 30)

  optimizer_cls = tools.bind(
      tf.train.AdamOptimizer,
      epsilon=params.get('optimizer_epsilon', 1e-4))
  config.optimizers = tools.AttrDict()
  config.optimizers.default = tools.bind(
      tools.CustomOptimizer,
      optimizer_cls=optimizer_cls,
      # schedule=tools.bind(tools.schedule.linear, ramp=0),
      learning_rate=params.get('default_lr', 1e-3),
      clipping=params.get('default_gradient_clipping', 1000.0))
  config.optimizers.model = config.optimizers.default.copy(
      learning_rate=params.get('model_lr', 6e-4),
      clipping=params.get('model_gradient_clipping', 100.0))
  config.optimizers.value = config.optimizers.default.copy(
      learning_rate=params.get('value_lr', 8e-5),
      clipping=params.get('value_gradient_clipping', 100.0))
  config.optimizers.action = config.optimizers.default.copy(
      learning_rate=params.get('action_lr', 8e-5),
      clipping=params.get('action_gradient_clipping', 100.0))
  return config


def _training_schedule(config, params):
  config.train_steps = int(params.get('train_steps', 50000))
  config.test_steps = int(params.get('test_steps', config.batch_shape[0]))
  config.max_steps = int(params.get('max_steps', 5e7))
  config.train_log_every = params.get('train_log_every', config.train_steps)
  config.train_checkpoint_every = None
  config.test_checkpoint_every = int(
      params.get('checkpoint_every', 10 * config.test_steps))
  config.checkpoint_to_load = None
  config.savers = [tools.AttrDict(exclude=(r'.*_temporary.*',))]
  config.print_metrics_every = config.train_steps // 10
  config.train_dir = os.path.join(params.logdir, 'train_episodes')
  config.test_dir = os.path.join(params.logdir, 'test_episodes')
  config.random_collects = _initial_collection(config, params)

  defaults = tools.AttrDict()
  defaults.name = 'main'
  defaults.give_rewards = True
  defaults.horizon = params.get('planner_horizon', 12)
  defaults.objective = params.get('planner_objective', 'reward_value')
  defaults.num_envs = params.get('num_envs', 1)
  defaults.num_episodes = params.get('collect_episodes', defaults.num_envs)
  defaults.num_steps = params.get('collect_steps', 500)
  defaults.steps_after = params.get('collect_every', 5000)
  defaults.steps_every = params.get('collect_every', 5000)
  defaults.steps_until = -1
  defaults.action_noise_type = params.get(
      'action_noise_type', 'additive_normal')

  train_defaults = defaults.copy(_unlocked=True)
  train_defaults.prefix = 'train'
  train_defaults.mode = 'train'
  train_defaults.save_episode_dir = config.train_dir
  train_defaults.planner = params.get('train_planner', 'policy_sample')
  train_defaults.objective = params.get(
      'train_planner_objective', defaults.objective)
  train_defaults.action_noise_scale = params.get('train_action_noise', 0.3)
  train_defaults.action_noise_ramp = params.get('train_action_noise_ramp', 0)
  train_defaults.action_noise_min = params.get('train_action_noise_min', 0.0)
  train_defaults.action_noise_factors = params.get(
      'train_action_noise_factors', [])
  config.train_collects = _active_collection(
      config.train_tasks, params.get('train_collects', [{}]), train_defaults,
      config, params)

  test_defaults = defaults.copy(_unlocked=True)
  test_defaults.prefix = 'test'
  test_defaults.mode = 'test'
  test_defaults.save_episode_dir = config.test_dir
  test_defaults.planner = params.get('test_planner', 'policy_mode')
  test_defaults.objective = params.get(
      'test_planner_objective', defaults.objective)
  test_defaults.action_noise_scale = params.get('test_action_noise', 0.0)
  test_defaults.action_noise_ramp = 0
  test_defaults.action_noise_min = 0.0
  test_defaults.action_noise_factors = params.get(
      'train_action_noise_factors', None)
  config.test_collects = _active_collection(
      config.test_tasks, params.get('test_collects', [{}]), test_defaults,
      config, params)
  return config


def _initial_collection(config, params):
  num_seed_episodes = int(params.get('num_seed_episodes', 5))
  num_seed_steps = int(params.get('num_seed_steps', 2500))
  sims = tools.AttrDict()
  for task in config.train_tasks:
    sims['train-' + task.name] = tools.AttrDict(
        task=task,
        mode='train',
        save_episode_dir=config.train_dir,
        num_episodes=num_seed_episodes,
        num_steps=num_seed_steps,
        give_rewards=params.get('seed_episode_rewards', True))
  for task in config.test_tasks:
    sims['test-' + task.name] = tools.AttrDict(
        task=task,
        mode='test',
        save_episode_dir=config.test_dir,
        num_episodes=num_seed_episodes,
        num_steps=num_seed_steps,
        give_rewards=True)
  return sims


def _active_collection(tasks, collects, defaults, config, params):
  sims = tools.AttrDict()
  for task in tasks:
    for user_collect in collects:
      for key in user_collect:
        if key not in defaults:
          message = 'Invalid key {} in activation collection config.'
          raise KeyError(message.format(key))
      collect = tools.AttrDict(defaults, _unlocked=True)
      collect.update(user_collect)
      collect.planner = _define_planner(
          collect.planner, collect.horizon, config, params)
      collect.objective = tools.bind(
          getattr(objectives_lib, collect.objective), params=params)
      if collect.give_rewards:
        collect.task = task
      else:
        env_ctor = tools.bind(
            lambda ctor: control.wrappers.NoRewardHint(ctor()),
            task.env_ctor)
        collect.task = tasks_lib.Task(task.name, env_ctor)
      collect.exploration = tools.AttrDict(
          scale=collect.action_noise_scale,
          type=collect.action_noise_type,
          schedule=tools.bind(
              tools.schedule.linear,
              ramp=collect.action_noise_ramp,
              min=collect.action_noise_min),
          factors=collect.action_noise_factors)
      name = '{}_{}_{}'.format(collect.prefix, collect.name, task.name)
      assert name not in sims, (set(sims.keys()), name)
      sims[name] = collect
  return sims


def _define_planner(planner, horizon, config, params):
  if planner == 'cem':
    planner_fn = tools.bind(
        control.planning.cross_entropy_method,
        beams=params.get('planner_beams', 1000),
        iterations=params.get('planner_iterations', 10),
        topk=params.get('planner_topk', 100),
        horizon=horizon)
  elif planner == 'policy_sample':
    planner_fn = tools.bind(
        control.planning.action_head_policy,
        strategy='sample',
        config=config)
  elif planner == 'policy_mode':
    planner_fn = tools.bind(
        control.planning.action_head_policy,
        strategy='mode',
        config=config)
  else:
    raise NotImplementedError(planner)
  return planner_fn
