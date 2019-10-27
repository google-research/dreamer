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
import os

import numpy as np

from dreamer import control
from dreamer import tools


Task = collections.namedtuple('Task', 'name, env_ctor, state_components')


def dummy(config, params):
  action_repeat = params.get('action_repeat', 1)
  env_ctor = lambda: control.wrappers.ActionRepeat(
      control.DummyEnv(), action_repeat)
  return Task('dummy', env_ctor, [])


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'cartpole', 'balance', config, params, action_repeat)
  return Task('cartpole_balance', env_ctor, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'cartpole', 'swingup', config, params, action_repeat)
  return Task('cartpole_swingup', env_ctor, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, 'finger', 'spin', config, params, action_repeat)
  return Task('finger_spin', env_ctor, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'cheetah', 'run', config, params, action_repeat)
  return Task('cheetah_run', env_ctor, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'ball_in_cup', 'catch', config, params, action_repeat)
  return Task('cup_catch', env_ctor, state_components)


def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['height', 'orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'walker', 'walk', config, params, action_repeat)
  return Task('walker_walk', env_ctor, state_components)


def walker_run(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['height', 'orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'walker', 'run', config, params, action_repeat)
  return Task('walker_run', env_ctor, state_components)


def walker_stand(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['height', 'orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'walker', 'stand', config, params, action_repeat)
  return Task('walker_stand', env_ctor, state_components)


def humanoid_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = [
      'reward', 'com_velocity', 'extremities', 'head_height', 'joint_angles',
      'torso_vertical', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'humanoid', 'walk', config, params, action_repeat)
  return Task('humanoid_walk', env_ctor, state_components)


def reacher_easy(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity', 'to_target']
  env_ctor = tools.bind(
      _dm_control_env, 'reacher', 'easy', config, params, action_repeat)
  return Task('reacher_easy', env_ctor, state_components)


def reacher_hard(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity', 'to_target']
  env_ctor = tools.bind(
      _dm_control_env, 'reacher', 'hard', config, params, action_repeat)
  return Task('reacher_hard', env_ctor, state_components)


def hopper_stand(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, 'hopper', 'stand', config, params, action_repeat)
  return Task('hopper_stand', env_ctor, state_components)


def hopper_hop(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity', 'touch']
  env_ctor = tools.bind(
      _dm_control_env, 'hopper', 'hop', config, params, action_repeat)
  return Task('hopper_hop', env_ctor, state_components)


def fish_upright(config, params):
  action_repeat = params.get('action_repeat', 2)
  env_ctor = tools.bind(
      _dm_control_env, 'fish', 'upright', config, params, action_repeat)
  return Task('fish_upright', env_ctor, [])


def pointmass_easy(config, params):
  action_repeat = params.get('action_repeat', 2)
  env_ctor = tools.bind(
      _dm_control_env, 'point_mass', 'easy', config, params, action_repeat)
  return Task('pointmass_easy', env_ctor, [])


def manipulator_bring(config, params):
  action_repeat = params.get('action_repeat', 2)
  env_ctor = tools.bind(
      _dm_control_env, 'manipulator', 'bring_ball', config, params,
      action_repeat)
  return Task('manipulator_bring_ball', env_ctor, [])


def pendulum_swingup(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['orientation', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'pendulum', 'swingup', config, params, action_repeat)
  return Task('pendulum_swingup', env_ctor, state_components)


def finger_turn_easy(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = [
      'reward', 'position', 'velocity', 'touch', 'target_position',
      'dist_to_target']
  env_ctor = tools.bind(
      _dm_control_env, 'finger', 'turn_easy', config, params, action_repeat)
  return Task('finger_turn_easy', env_ctor, state_components)


def finger_turn_hard(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = [
      'reward', 'position', 'velocity', 'touch', 'target_position',
      'dist_to_target']
  env_ctor = tools.bind(
      _dm_control_env, 'finger', 'turn_hard', config, params, action_repeat)
  return Task('finger_turn_hard', env_ctor, state_components)


def cartpole_balance_sparse(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'cartpole', 'balance_sparse', config, params,
      action_repeat)
  return Task(
      'cartpole_balance_sparse', env_ctor, state_components)


def cartpole_swingup_sparse(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['position', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'cartpole', 'swingup_sparse', config, params,
      action_repeat)
  return Task(
      'cartpole_swingup_sparse', env_ctor, state_components)


def quadruped_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = [
      'reward', 'egocentric_state', 'force_torque', 'imu', 'torso_upright',
      'torso_velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'quadruped', 'walk', config, params, action_repeat,
      camera_id=2, normalize_actions=True)
  return Task('quadruped_walk', env_ctor, state_components)


def quadruped_run(config, params):
  action_repeat = params.get('action_repeat', 2)
  env_ctor = tools.bind(
      _dm_control_env, 'quadruped', 'run', config, params, action_repeat,
      camera_id=2, normalize_actions=True)
  return Task('quadruped_run', env_ctor, [])


def quadruped_escape(config, params):
  action_repeat = params.get('action_repeat', 2)
  env_ctor = tools.bind(
      _dm_control_env, 'quadruped', 'escape', config, params, action_repeat,
      camera_id=2, normalize_actions=True)
  return Task('quadruped_escape', env_ctor, [])


def quadruped_fetch(config, params):
  action_repeat = params.get('action_repeat', 2)
  env_ctor = tools.bind(
      _dm_control_env, 'quadruped', 'fetch', config, params, action_repeat,
      camera_id=2, normalize_actions=True)
  return Task('quadruped_fetch', env_ctor, [])


def acrobot_swingup(config, params):
  action_repeat = params.get('action_repeat', 2)
  state_components = ['orientations', 'velocity']
  env_ctor = tools.bind(
      _dm_control_env, 'acrobot', 'swingup', config, params, action_repeat)
  return Task('acrobot_swingup', env_ctor, state_components)


DMLAB_TASKS = {
    'dmlab_collect': 'rooms_collect_good_objects_train',
    'dmlab_collect_few': 'explore_object_rewards_few',
    'dmlab_compare': 'psychlab_sequential_comparison',
    'dmlab_doors_large': 'explore_obstructed_goals_large',
    'dmlab_doors_small': 'explore_obstructed_goals_small',
    'dmlab_explore_small': 'explore_object_locations_small',
    'dmlab_find_large': 'explore_goal_locations_large',
    'dmlab_find_small': 'explore_goal_locations_small',
    'dmlab_keys': 'rooms_keys_doors_puzzle',
    'dmlab_lasertag_1': 'lasertag_one_opponent_small',
    'dmlab_lasertag_3': 'lasertag_three_opponents_small',
    'dmlab_recognize': 'psychlab_visual_search',
    'dmlab_watermaze': 'rooms_watermaze',
}


for name, level in DMLAB_TASKS.items():
  def task_fn(name, level, config, params):
    action_repeat = params.get('action_repeat', 4)
    env_ctor = tools.bind(_dm_lab_env, level, config, params, action_repeat)
    return Task(name, env_ctor, [])
  locals()[name] = tools.bind(task_fn, name, level)


ATARI_TASKS = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis',
    'BankHeist', 'BattleZone', 'BeamRider', 'Berzerk', 'Bowling', 'Boxing',
    'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber', 'Defender',
    'DemonAttack', 'DoubleDunk', 'Enduro', 'FishingDerby', 'Freeway',
    'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond',
    'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
    'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'PrivateEye', 'Qbert',
    'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'Skiing', 'Solaris',
    'SpaceInvaders', 'StarGunner', 'Surround', 'Tennis', 'TimePilot',
    'Tutankham', 'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor',
    'YarsRevenge', 'Zaxxon']
ATARI_TASKS = {'atari_{}'.format(game.lower()): game for game in ATARI_TASKS}


for name, game in ATARI_TASKS.items():
  def task_fn(name, game, config, params, mode):
    action_repeat = params.get('action_repeat', 4)
    env_ctor = tools.bind(
        _atari_env, game, mode, config, params, action_repeat)
    return Task(name, env_ctor, [])
  locals()[name] = tools.bind(task_fn, name, game)


def gym_cheetah(config, params):
  # Works with `isolate_envs: process`.
  action_repeat = params.get('action_repeat', 1)
  state_components = ['state']
  env_ctor = tools.bind(
      _gym_env, 'HalfCheetah-v3', config, params, action_repeat)
  return Task('gym_cheetah', env_ctor, state_components)


def gym_racecar(config, params):
  # Works with `isolate_envs: thread`.
  action_repeat = params.get('action_repeat', 1)
  env_ctor = tools.bind(
      _gym_env, 'CarRacing-v0', config, params, action_repeat,
      select_obs=[], obs_is_image=False, render_mode='state_pixels')
  return Task('gym_racing', env_ctor, [])


def _dm_control_env(
    domain, task, config, params, action_repeat, camera_id=None,
    normalize_actions=False):
  if camera_id is None:
    camera_id = int(params.get('camera_id', 0))
  size = params.get('render_size', 64)
  env = control.wrappers.DeepMindControl(
      domain, task, (size, size), camera_id=camera_id)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  if params.get('normalize_actions', normalize_actions):
    env = control.wrappers.NormalizeActions(env)
  env = control.wrappers.PixelObservations(
      env, (size, size), np.uint8, 'image')
  return _common_env(env, config, params)


def _dm_lab_env(level, config, params, action_repeat):
  runfiles_path = params.get('dmlab_runfiles_path', None)
  if runfiles_path:
    runfiles_path = os.path.join(os.environ['BORG_ALLOC_DIR'], runfiles_path)
  # Typical render sizes are 84x84 and 72x96. We're currently using 64x64.
  size = params.get('render_size', 64)
  env = control.wrappers.DeepMindLabyrinth(
      level, 'train', (size, size), action_repeat=action_repeat,
      runfiles_path=runfiles_path)
  # Agent outputs correct actions but random collect does not.
  env = control.wrappers.OneHotAction(env, strict=False)
  return _common_env(env, config, params)


def _atari_env(game, mode, config, params, action_repeat):
  assert mode in ('train', 'test')
  # Typical render size is 84x84. We're currently using 64x64.
  size = params.get('render_size', 64)
  env = control.wrappers.Atari(
      game, action_repeat, (size, size),
      grayscale=params.get('atari_grayscale', False),
      noops=params.get('atari_noops', 30),
      life_done=params.get('atari_lifes', True) and mode == 'train',
      sticky_actions=params.get('atari_sticky', True))
  # Agent outputs correct actions but random collect does not.
  env = control.wrappers.OneHotAction(env, strict=False)
  train_max_length = params.get('atari_train_max_length', None)
  if mode == 'train' and train_max_length:
    env = control.wrappers.MaximumDuration(env, train_max_length)
  return _common_env(env, config, params)


def _gym_env(
    name, config, params, action_repeat, select_obs=None, obs_is_image=False,
    render_mode='rgb_array'):
  import gym
  env = gym.make(name)
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.NormalizeActions(env)
  if obs_is_image:
    env = control.wrappers.ObservationDict(env, 'image')
    env = control.wrappers.ObservationToRender(env)
  else:
    env = control.wrappers.ObservationDict(env, 'state')
  if select_obs is not None:
    env = control.wrappers.SelectObservations(env, select_obs)
  size = params.get('render_size', 64)
  env = control.wrappers.PixelObservations(
      env, (size, size), np.uint8, 'image', render_mode)
  return _common_env(env, config, params)


def _common_env(env, config, params):
  env = control.wrappers.MinimumDuration(env, config.batch_shape[0])
  max_length = params.get('max_length', None)
  if max_length:
    env = control.wrappers.MaximumDuration(env, max_length)
  env = control.wrappers.ConvertTo32Bit(env)
  return env
