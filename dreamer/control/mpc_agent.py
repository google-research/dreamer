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

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from dreamer.tools import nested


class MPCAgent(object):

  def __init__(self, batch_env, step, is_training, should_log, config):
    self._step = step  # Trainer step, not environment step.
    self._is_training = is_training
    self._should_log = should_log
    self._config = config
    self._cell = config.cell
    self._num_envs = len(batch_env)
    state = self._cell.zero_state(self._num_envs, tf.float32)
    var_like = lambda x: tf.get_local_variable(
        x.name.split(':')[0].replace('/', '_') + '_var',
        shape=x.shape,
        initializer=lambda *_, **__: tf.zeros_like(x), use_resource=True)
    self._state = nested.map(var_like, state)
    batch_action_shape = (self._num_envs,) + batch_env.action_space.shape
    self._prev_action = tf.get_local_variable(
        'prev_action_var', shape=batch_action_shape,
        initializer=lambda *_, **__: tf.zeros(batch_action_shape),
        use_resource=True)

  def reset(self, agent_indices):
    state = nested.map(
        lambda tensor: tf.gather(tensor, agent_indices),
        self._state)
    reset_state = nested.map(
        lambda var, val: tf.scatter_update(var, agent_indices, 0 * val),
        self._state, state, flatten=True)
    reset_prev_action = self._prev_action.assign(
        tf.zeros_like(self._prev_action))
    return tf.group(reset_prev_action, *reset_state)

  def step(self, agent_indices, observ):
    observ = self._config.preprocess_fn(observ)
    # Converts observ to sequence.
    observ = nested.map(lambda x: x[:, None], observ)
    embedded = self._config.encoder(observ)[:, 0]
    state = nested.map(
        lambda tensor: tf.gather(tensor, agent_indices),
        self._state)
    prev_action = self._prev_action + 0
    with tf.control_dependencies([prev_action]):
      use_obs = tf.ones(tf.shape(agent_indices), tf.bool)[:, None]
      _, state = self._cell((embedded, prev_action, use_obs), state)
    action = self._config.planner(
        self._cell, self._config.objective, state,
        embedded.shape[1:].as_list(),
        prev_action.shape[1:].as_list())
    action = action[:, 0]
    if self._config.exploration:
      expl = self._config.exploration
      scale = tf.cast(expl.scale, tf.float32)[None]  # Batch dimension.
      if expl.schedule:
        scale *= expl.schedule(self._step)
      if expl.factors:
        scale *= np.array(expl.factors)
      if expl.type == 'additive_normal':
        action = tfd.Normal(action, scale[:, None]).sample()
      elif expl.type == 'epsilon_greedy':
        random_action = tf.one_hot(
            tfd.Categorical(0 * action).sample(), action.shape[-1])
        switch = tf.cast(tf.less(
            tf.random.uniform((self._num_envs,)),
            scale), tf.float32)[:, None]
        action = switch * random_action + (1 - switch) * action
      else:
        raise NotImplementedError(expl.type)
    action = tf.clip_by_value(action, -1, 1)
    remember_action = self._prev_action.assign(action)
    remember_state = nested.map(
        lambda var, val: tf.scatter_update(var, agent_indices, val),
        self._state, state, flatten=True)
    with tf.control_dependencies(remember_state + (remember_action,)):
      return tf.identity(action)
