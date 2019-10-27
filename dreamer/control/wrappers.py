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

import atexit
import datetime
import io
import os
import sys
import threading
import traceback
import uuid

import gym
import gym.spaces
import numpy as np
import skimage.transform
import tensorflow as tf

from dreamer import tools


class ObservationDict(object):

  def __init__(self, env, key='observ'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class ConcatObservation(object):

  def __init__(self, env, keys):
    self._env = env
    self._keys = keys

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    spaces = [spaces[key] for key in self._keys]
    low = np.concatenate([space.low for space in spaces], 0)
    high = np.concatenate([space.high for space in spaces], 0)
    dtypes = [space.dtype for space in spaces]
    if not all(dtype == dtypes[0] for dtype in dtypes):
      message = 'Spaces must have the same data type; are {}.'
      raise KeyError(message.format(', '.join(str(x) for x in dtypes)))
    return gym.spaces.Box(low, high, dtype=dtypes[0])

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = self._select_keys(obs)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = self._select_keys(obs)
    return obs

  def _select_keys(self, obs):
    return np.concatenate([obs[key] for key in self._keys], 0)


class SelectObservations(object):

  def __init__(self, env, keys):
    self._env = env
    self._keys = keys

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    return gym.spaces.Dict({key: spaces[key] for key in self._keys})

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action, *args, **kwargs):
    obs, reward, done, info = self._env.step(action, *args, **kwargs)
    obs = {key: obs[key] for key in self._keys}
    return obs, reward, done, info

  def reset(self, *args, **kwargs):
    obs = self._env.reset(*args, **kwargs)
    obs = {key: obs[key] for key in self._keys}
    return obs


class PixelObservations(object):

  def __init__(
      self, env, size=(64, 64), dtype=np.uint8, key='image',
      render_mode='rgb_array'):
    assert isinstance(env.observation_space, gym.spaces.Dict)
    self._env = env
    self._size = size
    self._dtype = dtype
    self._key = key
    self._render_mode = render_mode

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    high = {np.uint8: 255, np.float: 1.0}[self._dtype]
    image = gym.spaces.Box(0, high, self._size + (3,), dtype=self._dtype)
    spaces = self._env.observation_space.spaces.copy()
    assert self._key not in spaces
    spaces[self._key] = image
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs[self._key] = self._render_image()
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._render_image()
    return obs

  def _render_image(self):
    image = self._env.render(self._render_mode)
    if image.shape[:2] != self._size:
      kwargs = dict(
          output_shape=self._size, mode='edge', order=1, preserve_range=True)
      image = skimage.transform.resize(image, **kwargs).astype(image.dtype)
    if self._dtype and image.dtype != self._dtype:
      if image.dtype in (np.float32, np.float64) and self._dtype == np.uint8:
        image = (image * 255).astype(self._dtype)
      elif image.dtype == np.uint8 and self._dtype in (np.float32, np.float64):
        image = image.astype(self._dtype) / 255
      else:
        message = 'Cannot convert observations from {} to {}.'
        raise NotImplementedError(message.format(image.dtype, self._dtype))
    return image


class ObservationToRender(object):

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._image = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    return gym.spaces.Dict({})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._image = obs.pop(self._key)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    self._image = obs.pop(self._key)
    return obs

  def render(self, *args, **kwargs):
    return self._image


class OverwriteRender(object):

  def __init__(self, env, render_fn):
    self._env = env
    self._render_fn = render_fn
    self._env.render('rgb_array')  # Set up viewer.

  def __getattr__(self, name):
    return getattr(self._env, name)

  def render(self, *args, **kwargs):
    return self._render_fn(self._env, *args, **kwargs)


class ActionRepeat(object):

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      observ, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return observ, total_reward, done, info


class NormalizeActions(object):

  def __init__(self, env):
    self._env = env
    low, high = env.action_space.low, env.action_space.high
    self._enabled = np.logical_and(np.isfinite(low), np.isfinite(high))
    self._low = np.where(self._enabled, low, -np.ones_like(low))
    self._high = np.where(self._enabled, high, np.ones_like(low))

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    space = self._env.action_space
    low = np.where(self._enabled, -np.ones_like(space.low), space.low)
    high = np.where(self._enabled, np.ones_like(space.high), space.high)
    return gym.spaces.Box(low, high, dtype=space.dtype)

  def step(self, action):
    action = (action + 1) / 2 * (self._high - self._low) + self._low
    return self._env.step(action)


class DeepMindControl(object):

  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, domain, task, render_size=(64, 64), camera_id=0):
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._render_size = render_size
    self._camera_id = camera_id

  @property
  def observation_space(self):
    components = {}
    for key, value in self._env.observation_spec().items():
      components[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    return gym.spaces.Dict(components)

  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return gym.spaces.Box(
        action_spec.minimum, action_spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': time_step.discount}
    if done:
      info['done_reason'] = 'timeout'
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    return dict(time_step.observation)

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._env.physics.render(
        *self._render_size, camera_id=self._camera_id)


class DeepMindLabyrinth(object):

  ACTION_SET_DEFAULT = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (0, 0, 0, -1, 0, 0, 0),   # Backward
      (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
      (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
      (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
      (20, 0, 0, 1, 0, 0, 0),   # Look Right + Forward
      (0, 0, 0, 0, 1, 0, 0),    # Fire
  )

  ACTION_SET_MEDIUM = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (0, 0, 0, -1, 0, 0, 0),   # Backward
      (0, 0, -1, 0, 0, 0, 0),   # Strafe Left
      (0, 0, 1, 0, 0, 0, 0),    # Strafe Right
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
      (0, 0, 0, 0, 0, 0, 0),    # Idle.
  )

  ACTION_SET_SMALL = (
      (0, 0, 0, 1, 0, 0, 0),    # Forward
      (-20, 0, 0, 0, 0, 0, 0),  # Look Left
      (20, 0, 0, 0, 0, 0, 0),   # Look Right
  )

  def __init__(
      self, level, mode, render_size=(64, 64), action_repeat=4,
      action_set=ACTION_SET_DEFAULT, level_cache=None, seed=None,
      runfiles_path=None):
    assert mode in ('train', 'test')
    import deepmind_lab
    if runfiles_path:
      print('Setting DMLab runfiles path:', runfiles_path)
      deepmind_lab.set_runfiles_path(runfiles_path)
    self._config = {}
    self._config['width'] = render_size[0]
    self._config['height'] = render_size[1]
    self._config['logLevel'] = 'WARN'
    if mode == 'test':
      self._config['allowHoldOutLevels'] = 'true'
      self._config['mixerSeed'] = 0x600D5EED
    self._action_repeat = action_repeat
    self._random = np.random.RandomState(seed)
    self._env = deepmind_lab.Lab(
        level=level,
        observations=['RGB_INTERLEAVED'],
        config={k: str(v) for k, v in self._config.items()},
        level_cache=level_cache)
    self._action_set = action_set
    self._last_image = None
    self._done = True

  @property
  def observation_space(self):
    shape = (self._config['height'], self._config['width'], 3)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return gym.spaces.Discrete(len(self._action_set))

  def reset(self):
    self._done = False
    self._env.reset(seed=self._random.randint(0, 2 ** 31 - 1))
    obs = self._get_obs()
    return obs

  def step(self, action):
    raw_action = np.array(self._action_set[action], np.intc)
    reward = self._env.step(raw_action, num_steps=self._action_repeat)
    self._done = not self._env.is_running()
    obs = self._get_obs()
    return obs, reward, self._done, {}

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._last_image

  def close(self):
    self._env.close()

  def _get_obs(self):
    if self._done:
      image = 0 * self._last_image
    else:
      image = self._env.observations()['RGB_INTERLEAVED']
    self._last_image = image
    return {'image': image}


class LocalLevelCache(object):

  def __init__(self, cache_dir='/tmp/level_cache'):
    self._cache_dir = cache_dir
    tf.gfile.MakeDirs(cache_dir)

  def fetch(self, key, pk3_path):
    path = os.path.join(self._cache_dir, key)
    if tf.gfile.Exists(path):
      tf.gfile.Copy(path, pk3_path, overwrite=True)
      return True
    return False

  def write(self, key, pk3_path):
    path = os.path.join(self._cache_dir, key)
    if not tf.gfile.Exists(path):
      tf.gfile.Copy(pk3_path, path)


class Atari(object):

  # LOCK = multiprocessing.Lock()
  LOCK = threading.Lock()

  def __init__(
      self, name, action_repeat=4, size=(84, 84), grayscale=True, noops=30,
      life_done=False, sticky_actions=True):
    import gym
    version = 0 if sticky_actions else 4
    with self.LOCK:
      self._env = gym.make('{}NoFrameskip-v{}'.format(name, version))
    self._action_repeat = action_repeat
    self._size = size
    self._grayscale = grayscale
    self._noops = noops
    self._life_done = life_done
    self._lives = None
    shape = self._env.observation_space.shape[:2] + (() if grayscale else (3,))
    self._buffers = [np.empty(shape, dtype=np.uint8) for _ in range(2)]
    self._random = np.random.RandomState(seed=None)

  @property
  def observation_space(self):
    shape = self._size + (1 if self._grayscale else 3,)
    space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    return gym.spaces.Dict({'image': space})

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      self._env.reset()
    # Use at least one no-op.
    noops = self._random.randint(1, self._noops) if self._noops > 1 else 1
    for _ in range(noops):
      done = self._env.step(0)[2]
      if done:
        with self.LOCK:
          self._env.reset()
    self._lives = self._env.ale.lives()
    if self._grayscale:
      self._env.ale.getScreenGrayscale(self._buffers[0])
    else:
      self._env.ale.getScreenRGB2(self._buffers[0])
    self._buffers[1].fill(0)
    return self._get_obs()

  def step(self, action):
    total_reward = 0.0
    for step in range(self._action_repeat):
      _, reward, done, info = self._env.step(action)
      total_reward += reward
      if self._life_done:
        lives = self._env.ale.lives()
        done = done or lives < self._lives
        self._lives = lives
      if done:
        # In principle, the loop could exit before two valid frames have been
        # rendered.
        break
      elif step >= self._action_repeat - 2:
        index = step - (self._action_repeat - 2)
        if self._grayscale:
          self._env.ale.getScreenGrayscale(self._buffers[index])
        else:
          self._env.ale.getScreenRGB2(self._buffers[index])
    obs = self._get_obs()
    return obs, total_reward, done, info

  def render(self, mode):
    return self._env.render(mode)

  def _get_obs(self):
    if self._action_repeat > 1:
      np.maximum(self._buffers[0], self._buffers[1], out=self._buffers[0])
    image = skimage.transform.resize(
        self._buffers[0], output_shape=self._size, mode='edge', order=1,
        preserve_range=True)
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = image[:, :, None] if self._grayscale else image
    return {'image': image}


class OneHotAction(object):

  def __init__(self, env, strict=True):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env
    self._strict = strict

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    return gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)

  def step(self, action):
    index = np.argmax(action).astype(int)
    if self._strict:
      reference = np.zeros_like(action)
      reference[index] = 1
      assert np.allclose(reference, action), action
    return self._env.step(index)

  def reset(self):
    return self._env.reset()


class MaximumDuration(object):

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    if self._step is None:
      raise RuntimeError('Must reset environment.')
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
      if 'done_reason' not in info:
        info['done_reason'] = 'timeout'
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class MinimumDuration(object):

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step < self._duration:
      done = False
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ProcessObservation(object):

  def __init__(self, env, process_fn):
    self._env = env
    self._process_fn = process_fn

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    return tools.nested.map(
        lambda box: gym.spaces.Box(
            self._process_fn(box.low),
            self._process_fn(box.high),
            dtype=self._process_fn(box.low).dtype),
        self._env.observation_space)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    observ = self._process_fn(observ)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    observ = self._process_fn(observ)
    return observ


class PadActions(object):

  def __init__(self, env, spaces):
    self._env = env
    self._action_space = self._pad_box_space(spaces)

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action, *args, **kwargs):
    action = action[:len(self._env.action_space.low)]
    return self._env.step(action, *args, **kwargs)

  def reset(self, *args, **kwargs):
    return self._env.reset(*args, **kwargs)

  def _pad_box_space(self, spaces):
    assert all(len(space.low.shape) == 1 for space in spaces)
    length = max(len(space.low) for space in spaces)
    low, high = np.inf * np.ones(length), -np.inf * np.ones(length)
    for space in spaces:
      low[:len(space.low)] = np.minimum(space.low, low[:len(space.low)])
      high[:len(space.high)] = np.maximum(space.high, high[:len(space.high)])
    return gym.spaces.Box(low, high, dtype=np.float32)


class ObservationDropout(object):

  def __init__(self, env, key, prob):
    self._env = env
    self._key = key
    self._prob = prob
    self._random = np.random.RandomState(seed=0)

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    observ = self._process_fn(observ)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    observ = self._process_fn(observ)
    return observ

  def _process_fn(self, observ):
    if self._random.uniform(0, 1) < self._prob:
      observ[self._key] *= 0
    return observ


class CollectDataset(object):

  def __init__(self, env, outdir):
    self._env = env
    self._outdir = outdir and os.path.expanduser(outdir)
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action, *args, **kwargs):
    if kwargs.get('blocking', True):
      transition = self._env.step(action, *args, **kwargs)
      return self._process_step(action, *transition)
    else:
      future = self._env.step(action, *args, **kwargs)
      return lambda: self._process_step(action, *future())

  def reset(self, *args, **kwargs):
    if kwargs.get('blocking', True):
      observ = self._env.reset(*args, **kwargs)
      return self._process_reset(observ)
    else:
      future = self._env.reset(*args, **kwargs)
      return lambda: self._process_reset(future())

  def _process_step(self, action, observ, reward, done, info):
    transition = self._process_observ(observ).copy()
    transition['action'] = action
    transition['reward'] = reward
    if done:
      reason = info.get('done_reason', 'termination')
      transition['pcont'] = dict(termination=0.0, timeout=1.0)[reason]
    else:
      transition['pcont'] = 1.0
    self._episode.append(transition)
    if done:
      episode = self._get_episode()
      if self._outdir:
        filename = self._get_filename(episode)
        self._write(episode, filename)
    return observ, reward, done, info

  def _process_reset(self, observ):
    # Resetting the environment provides the observation for time step zero.
    # The action and reward are not known for this time step, so we zero them.
    transition = self._process_observ(observ).copy()
    transition['action'] = np.zeros_like(self.action_space.low)
    transition['reward'] = 0.0
    transition['pcont'] = 1.0
    self._episode = [transition]
    return observ

  def _process_observ(self, observ):
    if not isinstance(observ, dict):
      observ = {'observ': observ}
    return observ

  def _get_filename(self, episode):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = len(episode['reward'])
    filename = '{}-{}-{}.npz'.format(timestamp, identifier, length)
    filename = os.path.join(self._outdir, filename)
    return filename

  def _get_episode(self):
    episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
    episode = {k: np.array(v) for k, v in episode.items()}
    for key, sequence in episode.items():
      if sequence.dtype not in (np.uint8, np.float32, np.float64, np.bool):
        message = "Sequence for key {} is of unexpected type {}:\n{}"
        raise RuntimeError(message.format(key, sequence.dtype, sequence))
    return episode

  def _write(self, episode, filename):
    if not tf.gfile.Exists(self._outdir):
      tf.gfile.MakeDirs(self._outdir)
    with io.BytesIO() as file_:
      np.savez_compressed(file_, **episode)
      file_.seek(0)
      with tf.gfile.Open(filename, 'w') as ff:
        ff.write(file_.read())
    folder = os.path.basename(self._outdir)
    name = os.path.splitext(os.path.basename(filename))[0]
    word = 'with' if np.sum(episode.get('reward_mask', 1)) > 0 else 'without'
    score = episode['reward'].sum()
    message = 'Recorded episode {} of length {} {} score of {:.1f} to {}.'
    print(message.format(name, len(episode['action']), word, score, folder))


class NoRewardHint(object):

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces.copy()
    low = np.zeros(1, dtype=np.float32)
    high = np.ones(1, dtype=np.float32)
    spaces['reward_mask'] = gym.spaces.Box(low, high)
    return gym.spaces.Dict(spaces)

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs['reward_mask'] = np.zeros(1, dtype=np.float32)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs['reward_mask'] = np.zeros(1, dtype=np.float32)
    return obs


class ConvertTo32Bit(object):

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    observ = tools.nested.map(self._convert_observ, observ)
    reward = self._convert_reward(reward)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    observ = tools.nested.map(self._convert_observ, observ)
    return observ

  def _convert_observ(self, observ):
    if not np.isfinite(observ).all():
      raise ValueError('Infinite observation encountered.')
    if observ.dtype == np.float64:
      return observ.astype(np.float32)
    if observ.dtype == np.int64:
      return observ.astype(np.int32)
    return observ

  def _convert_reward(self, reward):
    if not np.isfinite(reward).all():
      raise ValueError('Infinite reward encountered.')
    return np.array(reward, dtype=np.float32)


class Async(object):

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, constructor, strategy='thread'):
    if strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    self._strategy = strategy
    self._conn, conn = mp.Pipe()
    self._process = mp.Process(target=self._worker, args=(constructor, conn))
    atexit.register(self.close)
    self._process.start()
    self._observ_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._observ_space:
      self._observ_space = self.__getattr__('observation_space')
    return self._observ_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=True):
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    try:
      message, payload = self._conn.recv()
    except (OSError, EOFError):
      raise RuntimeError('Lost connection to environment worker.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, constructor, conn):
    try:
      env = constructor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      try:
        conn.send((self._EXCEPTION, stacktrace))
      except Exception:
        print('Failed to send exception back to main process.')
    try:
      conn.close()
    except Exception:
      print('Failed to properly close connection.')
