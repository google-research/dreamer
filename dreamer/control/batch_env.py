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

import gym
import numpy as np
import tensorflow as tf


class TFBatchEnv(object):

  def __init__(self, envs, blocking):
    self._batch_env = PyBatchEnv(envs, blocking, flatten=True)
    spaces = self._batch_env.observation_space.spaces
    self._dtypes = [self._parse_dtype(spaces[key]) for key in self._keys[:-2]]
    self._dtypes += [tf.float32, tf.bool]  # Reward and done flag.
    self._shapes = [self._parse_shape(spaces[key]) for key in self._keys[:-2]]
    self._shapes += [(), ()]  # Reward and done flag.

  def __getattr__(self, name):
    return getattr(self._batch_env, name)

  def __len__(self):
    return len(self._batch_env)

  def __getitem__(self, index):
    return self._batch_env[index]

  def step(self, action):
    output = tf.py_func(
        self._batch_env.step, [action], self._dtypes, name='step')
    return self._process_output(output, len(self._batch_env))

  def reset(self, indices=None):
    if indices is None:
      indices = tf.range(len(self._batch_env))
    output = tf.py_func(
        self._batch_env.reset, [indices], self._dtypes, name='reset')
    return self._process_output(output, None)

  def _process_output(self, output, batch_size):
    for tensor, shape in zip(output, self._shapes):
      tensor.set_shape((batch_size,) + shape)
    return {key: tensor for key, tensor in zip(self._keys, output)}

  def _parse_dtype(self, space):
    if isinstance(space, gym.spaces.Discrete):
      return tf.int32
    if isinstance(space, gym.spaces.Box):
      if space.low.dtype == np.uint8:
        return tf.uint8
      else:
        return tf.float32
    raise NotImplementedError()

  def _parse_shape(self, space):
    if isinstance(space, gym.spaces.Discrete):
      return ()
    if isinstance(space, gym.spaces.Box):
      return space.shape
    raise NotImplementedError("Unsupported space '{}.'".format(space))


class PyBatchEnv(object):

  def __init__(self, envs, blocking, flatten=False):
    observ_space = envs[0].observation_space
    if not all(env.observation_space == observ_space for env in envs):
      raise ValueError('All environments must use the same observation space.')
    action_space = envs[0].action_space
    if not all(env.action_space == action_space for env in envs):
      raise ValueError('All environments must use the same observation space.')
    self._envs = envs
    self._blocking = blocking
    self._flatten = flatten
    self._keys = list(sorted(observ_space.spaces.keys())) + ['reward', 'done']

  def __len__(self):
    return len(self._envs)

  def __getitem__(self, index):
    return self._envs[index]

  def __getattr__(self, name):
    return getattr(self._envs[0], name)

  def step(self, actions):
    for index, (env, action) in enumerate(zip(self._envs, actions)):
      if not env.action_space.contains(action):
        message = 'Invalid action for batch index {}: {}'
        raise ValueError(message.format(index, action))
    if self._blocking:
      transitions = [
          env.step(action)
          for env, action in zip(self._envs, actions)]
    else:
      transitions = [
          env.step(action, blocking=False)
          for env, action in zip(self._envs, actions)]
      transitions = [transition() for transition in transitions]
    outputs = {key: [] for key in self._keys}
    for observ, reward, done, _ in transitions:
      for key, value in observ.items():
        outputs[key].append(np.array(value))
      outputs['reward'].append(np.array(reward, np.float32))
      outputs['done'].append(np.array(done, np.bool))
    outputs = {key: np.stack(value) for key, value in outputs.items()}
    if self._flatten:
      outputs = tuple(outputs[key] for key in self._keys)
    return outputs

  def reset(self, indices=None):
    if indices is None:
      indices = range(len(self._envs))
    if self._blocking:
      observs = [self._envs[index].reset() for index in indices]
    else:
      observs = [self._envs[index].reset(blocking=False) for index in indices]
      observs = [observ() for observ in observs]
    outputs = {key: [] for key in self._keys}
    for observ in observs:
      for key, value in observ.items():
        outputs[key].append(np.array(value))
      outputs['reward'].append(np.array(0.0, np.float32))
      outputs['done'].append(np.array(False, np.bool))
    outputs = {key: np.stack(value) for key, value in outputs.items()}
    if self._flatten:
      outputs = tuple(outputs[key] for key in self._keys)
    return outputs

  def close(self):
    for env in self._envs:
      if hasattr(env, 'close'):
        env.close()
