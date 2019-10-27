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

from dreamer.control import batch_env
from dreamer.control import wrappers


def create_batch_env(env_ctor, num_envs, isolate_envs):
  envs, blockings = zip(*[
      create_env(env_ctor, isolate_envs)
      for _ in range(num_envs)])
  assert all(blocking == blockings[0] for blocking in blockings)
  return batch_env.TFBatchEnv(envs, blockings[0])


def create_env(env_ctor, isolate_envs):
  if isolate_envs == 'none':
    env = env_ctor()
    blocking = True
  elif isolate_envs == 'thread':
    env = wrappers.Async(env_ctor, 'thread')
    blocking = False
  elif isolate_envs == 'process':
    env = wrappers.Async(env_ctor, 'process')
    blocking = False
  else:
    raise NotImplementedError(isolate_envs)
  return env, blocking


def simulate(agent, env, episodes=None, steps=None):

  def pred(step, episode, *args, **kwargs):
    if episodes is None:
      return tf.less(step, steps)
    if steps is None:
      return tf.less(episode, episodes)
    return tf.logical_or(tf.less(episode, episodes), tf.less(step, steps))

  def reset(mask, scores, lengths, previous):
    indices = tf.where(mask)[:, 0]
    reset_agent = agent.reset(indices)
    values = env.reset(indices)
    # This would be shorter but gives an internal TensorFlow error.
    # previous = {
    #     key: tf.tensor_scatter_update(
    #         previous[key], indices[:, None], values[key])
    #     for key in previous}
    idx = tf.cast(indices[:, None], tf.int32)
    previous = {
        key: tf.where(
            mask,
            tf.scatter_nd(idx, values[key], tf.shape(previous[key])),
            previous[key])
        for key in previous}
    scores = tf.where(mask, tf.zeros_like(scores), scores)
    lengths = tf.where(mask, tf.zeros_like(lengths), lengths)
    with tf.control_dependencies([reset_agent]):
      return tf.identity(scores), tf.identity(lengths), previous

  def body(step, episode, scores, lengths, previous, ta_out, ta_sco, ta_len):
    # Reset episodes and agents if necessary.
    reset_mask = tf.cond(
        tf.equal(step, 0),
        lambda: tf.ones(len(env), tf.bool),
        lambda: previous['done'])
    reset_mask.set_shape([len(env)])
    scores, lengths, previous = tf.cond(
        tf.reduce_any(reset_mask),
        lambda: reset(reset_mask, scores, lengths, previous),
        lambda: (scores, lengths, previous))
    step_indices = tf.range(len(env))
    action = agent.step(step_indices, previous)
    values = env.step(action)
    # Update book keeping variables.
    done_indices = tf.cast(tf.where(values['done']), tf.int32)
    step += tf.shape(step_indices)[0]
    episode += tf.shape(done_indices)[0]
    scores += values['reward']
    lengths += tf.shape(step_indices)[0]
    # Write transitions, scores, and lengths to tensor arrays.
    ta_out = {
        key: array.write(array.size(), values[key])
        for key, array in ta_out.items()}
    ta_sco = tf.cond(
        tf.greater(tf.shape(done_indices)[0], 0),
        lambda: ta_sco.write(
            ta_sco.size(), tf.gather(scores, done_indices)[:, 0]),
        lambda: ta_sco)
    ta_len = tf.cond(
        tf.greater(tf.shape(done_indices)[0], 0),
        lambda: ta_len.write(
            ta_len.size(), tf.gather(lengths, done_indices)[:, 0]),
        lambda: ta_len)
    return step, episode, scores, lengths, values, ta_out, ta_sco, ta_len

  initial = env.reset()
  ta_out = {
      key: tf.TensorArray(
          value.dtype, 0, True, element_shape=value.shape)
      for key, value in initial.items()}
  ta_sco = tf.TensorArray(
      tf.float32, 0, True, element_shape=[None], infer_shape=False)
  ta_len = tf.TensorArray(
      tf.int32, 0, True, element_shape=[None], infer_shape=False)
  zero_scores = tf.zeros(len(env), tf.float32, name='scores')
  zero_lengths = tf.zeros(len(env), tf.int32, name='lengths')
  ta_out, ta_sco, ta_len = tf.while_loop(
      pred, body,
      (0, 0, zero_scores, zero_lengths, initial, ta_out, ta_sco, ta_len),
      parallel_iterations=1, back_prop=False)[-3:]
  transitions = {key: array.stack() for key, array in ta_out.items()}
  transitions = {
      key: tf.transpose(value, [1, 0] + list(range(2, value.shape.ndims)))
      for key, value in transitions.items()}
  scores = ta_sco.concat()
  lengths = ta_len.concat()
  return scores, lengths, transitions
