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

from dreamer import distributions
from dreamer import tools


def feed_forward(
    features, data_shape, num_layers=2, activation=tf.nn.relu,
    mean_activation=None, stop_gradient=False, trainable=True, units=100,
    std=1.0, low=-1.0, high=1.0, dist='normal', min_std=1e-2, init_std=1.0):
  hidden = features
  if stop_gradient:
    hidden = tf.stop_gradient(hidden)
  for _ in range(num_layers):
    hidden = tf.layers.dense(hidden, units, activation, trainable=trainable)
  mean = tf.layers.dense(
      hidden, int(np.prod(data_shape)), mean_activation, trainable=trainable)
  mean = tf.reshape(mean, tools.shape(features)[:-1] + data_shape)
  if std == 'learned':
    std = tf.layers.dense(
        hidden, int(np.prod(data_shape)), None, trainable=trainable)
    init_std = np.log(np.exp(init_std) - 1)
    std = tf.nn.softplus(std + init_std) + min_std
    std = tf.reshape(std, tools.shape(features)[:-1] + data_shape)
  if dist == 'normal':
    dist = tfd.Normal(mean, std)
    dist = tfd.Independent(dist, len(data_shape))
  elif dist == 'deterministic':
    dist = tfd.Deterministic(mean)
    dist = tfd.Independent(dist, len(data_shape))
  elif dist == 'binary':
    dist = tfd.Bernoulli(mean)
    dist = tfd.Independent(dist, len(data_shape))
  elif dist == 'trunc_normal':
    # https://www.desmos.com/calculator/rnksmhtgui
    dist = tfd.TruncatedNormal(mean, std, low, high)
    dist = tfd.Independent(dist, len(data_shape))
  elif dist == 'tanh_normal':
    # https://www.desmos.com/calculator/794s8kf0es
    dist = distributions.TanhNormal(mean, std)
  elif dist == 'tanh_normal_tanh':
    # https://www.desmos.com/calculator/794s8kf0es
    mean = 5.0 * tf.tanh(mean / 5.0)
    dist = distributions.TanhNormal(mean, std)
  elif dist == 'onehot_score':
    dist = distributions.OneHot(mean, gradient='score')
  elif dist == 'onehot_straight':
    dist = distributions.OneHot(mean, gradient='straight')
  else:
    raise NotImplementedError(dist)
  return dist
