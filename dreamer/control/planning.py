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

import tensorflow as tf

from dreamer import tools


def cross_entropy_method(
    cell, objective, state, obs_shape, action_shape, horizon, graph,
    beams=1000, topk=100, iterations=10, min_action=-1, max_action=1):
  obs_shape, action_shape = tuple(obs_shape), tuple(action_shape)
  batch = tools.shape(tools.nested.flatten(state)[0])[0]
  initial_state = tools.nested.map(lambda tensor: tf.tile(
      tensor, [beams] + [1] * (tensor.shape.ndims - 1)), state)
  extended_batch = tools.shape(tools.nested.flatten(initial_state)[0])[0]
  use_obs = tf.zeros([extended_batch, horizon, 1], tf.bool)
  obs = tf.zeros((extended_batch, horizon) + obs_shape)

  def iteration(index, mean, stddev):
    # Sample action proposals from belief.
    normal = tf.random_normal((batch, beams, horizon) + action_shape)
    action = normal * stddev[:, None] + mean[:, None]
    action = tf.clip_by_value(action, min_action, max_action)
    # Evaluate proposal actions.
    action = tf.reshape(
        action, (extended_batch, horizon) + action_shape)
    (_, state), _ = tf.nn.dynamic_rnn(
        cell, (0 * obs, action, use_obs), initial_state=initial_state)
    return_ = objective(state)
    return_ = tf.reshape(return_, (batch, beams))
    # Re-fit belief to the best ones.
    _, indices = tf.nn.top_k(return_, topk, sorted=False)
    indices += tf.range(batch)[:, None] * beams
    best_actions = tf.gather(action, indices)
    mean, variance = tf.nn.moments(best_actions, 1)
    stddev = tf.sqrt(variance + 1e-6)
    return index + 1, mean, stddev

  mean = tf.zeros((batch, horizon) + action_shape)
  stddev = tf.ones((batch, horizon) + action_shape)
  _, mean, std = tf.while_loop(
      lambda index, mean, stddev: index < iterations, iteration,
      (0, mean, stddev), back_prop=False)
  return mean


def action_head_policy(
    cell, objective, state, obs_shape, action_shape, graph, config, strategy):
  features = cell.features_from_state(state)
  policy = graph.heads.action(features)
  if strategy == 'sample':
    action = policy.sample()
  elif strategy == 'mode':
    action = policy.mode()
  else:
    raise NotImplementedError(strategy)
  plan = action[:, None, :]
  return plan
