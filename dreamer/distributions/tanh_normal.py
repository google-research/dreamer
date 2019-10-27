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
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class TanhNormal(object):

  def __init__(self, mean, std, samples=100):
    dist = tfd.Normal(mean, std)
    dist = tfd.TransformedDistribution(dist, TanhBijector())
    dist = tfd.Independent(dist, 1)
    self._dist = dist
    self._samples = samples

  @property
  def name(self):
    return 'TanhNormalDistribution'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def mean(self):
    samples = self._dist.sample(self._samples)
    return tf.reduce_mean(samples, 0)

  def stddev(self):
    samples = self._dist.sample(self._samples)
    mean = tf.reduce_mean(samples, 0, keep_dims=True)
    return tf.reduce_mean(tf.pow(samples - mean, 2), 0)

  def mode(self):
    samples = self._dist.sample(self._samples)
    logprobs = self._dist.log_prob(samples)
    mask = tf.one_hot(tf.argmax(logprobs, axis=0), self._samples, axis=0)
    return tf.reduce_sum(samples * mask[..., None], 0)

  def entropy(self):
    sample = self._dist.sample(self._samples)
    logprob = self.log_prob(sample)
    return -tf.reduce_mean(logprob, 0)


class TanhBijector(tfp.bijectors.Bijector):

  def __init__(self, validate_args=False, name='tanh'):
    super(TanhBijector, self).__init__(
        forward_min_event_ndims=0,
        validate_args=validate_args,
        name=name)

  def _forward(self, x):
    return tf.nn.tanh(x)

  def _inverse(self, y):
    precision = 0.99999997
    clipped = tf.where(
        tf.less_equal(tf.abs(y), 1.),
        tf.clip_by_value(y, -precision, precision), y)
    # y = tf.stop_gradient(clipped) + y - tf.stop_gradient(y)
    return tf.atanh(clipped)

  def _forward_log_det_jacobian(self, x):
    log2 = tf.math.log(tf.constant(2.0, dtype=x.dtype))
    return 2.0 * (log2 - x - tf.nn.softplus(-2.0 * x))
