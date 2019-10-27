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
from tensorflow_probability import distributions as tfd

from dreamer import tools
from dreamer.models import base


class RSSM(base.Base):

  def __init__(
      self, state_size, belief_size, embed_size,
      future_rnn=True, mean_only=False, min_stddev=0.1, activation=tf.nn.elu,
      num_layers=1):
    self._state_size = state_size
    self._belief_size = belief_size
    self._embed_size = embed_size
    self._future_rnn = future_rnn
    self._cell = tf.contrib.rnn.GRUBlockCell(self._belief_size)
    self._kwargs = dict(units=self._embed_size, activation=activation)
    self._mean_only = mean_only
    self._min_stddev = min_stddev
    self._num_layers = num_layers
    super(RSSM, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))

  @property
  def state_size(self):
    return {
        'mean': self._state_size,
        'stddev': self._state_size,
        'sample': self._state_size,
        'belief': self._belief_size,
        'rnn_state': self._belief_size,
    }

  @property
  def feature_size(self):
    return self._belief_size + self._state_size

  def dist_from_state(self, state, mask=None):
    if mask is not None:
      stddev = tools.mask(state['stddev'], mask, value=1)
    else:
      stddev = state['stddev']
    dist = tfd.MultivariateNormalDiag(state['mean'], stddev)
    return dist

  def features_from_state(self, state):
    return tf.concat([state['sample'], state['belief']], -1)

  def divergence_from_states(self, lhs, rhs, mask=None):
    lhs = self.dist_from_state(lhs, mask)
    rhs = self.dist_from_state(rhs, mask)
    divergence = tfd.kl_divergence(lhs, rhs)
    if mask is not None:
      divergence = tools.mask(divergence, mask)
    return divergence

  def _transition(self, prev_state, prev_action, zero_obs):
    hidden = tf.concat([prev_state['sample'], prev_action], -1)
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    belief, rnn_state = self._cell(hidden, prev_state['rnn_state'])
    if self._future_rnn:
      hidden = belief
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
    stddev += self._min_stddev
    if self._mean_only:
      sample = mean
    else:
      sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
    return {
        'mean': mean,
        'stddev': stddev,
        'sample': sample,
        'belief': belief,
        'rnn_state': rnn_state,
    }

  def _posterior(self, prev_state, prev_action, obs):
    prior = self._transition_tpl(prev_state, prev_action, tf.zeros_like(obs))
    hidden = tf.concat([prior['belief'], obs], -1)
    for _ in range(self._num_layers):
      hidden = tf.layers.dense(hidden, **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
    stddev += self._min_stddev
    if self._mean_only:
      sample = mean
    else:
      sample = tfd.MultivariateNormalDiag(mean, stddev).sample()
    return {
        'mean': mean,
        'stddev': stddev,
        'sample': sample,
        'belief': prior['belief'],
        'rnn_state': prior['rnn_state'],
    }
