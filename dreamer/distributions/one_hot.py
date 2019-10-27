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


class OneHot(object):

  def __init__(self, logits=None, probs=None, gradient='score'):
    self._gradient = gradient
    self._dist = tfd.Categorical(logits=logits, probs=probs)
    self._num_classes = self.mean().shape[-1].value

  @property
  def name(self):
    return 'OneHotDistribution'

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def prob(self, events):
    indices = tf.argmax(events, axis=-1)
    return self._dist.prob(indices)

  def log_prob(self, events):
    indices = tf.argmax(events, axis=-1)
    return self._dist.log_prob(indices)

  def mean(self):
    return self._dist.probs_parameter()

  def stddev(self):
    values = tf.one_hot(tf.range(self._num_classes), self._num_classes)
    distances = tf.reduce_sum((values - self.mean()[..., None, :]) ** 2, -1)
    return tf.sqrt(distances)

  def mode(self):
    return tf.one_hot(self._dist.mode(), self._num_classes)

  def sample(self, amount=None):
    amount = [amount] if amount else []
    sample = tf.one_hot(self._dist.sample(*amount), self._num_classes)
    if self._gradient == 'score':  # Implemented as DiCE.
      logp = self.log_prob(sample)[..., None]
      sample *= tf.exp(logp - tf.stop_gradient(logp))
    elif self._gradient == 'straight':  # Gradient for all classes.
      probs = self._dist.probs_parameter()
      sample += probs - tf.stop_gradient(probs)
    else:
      raise NotImplementedError(self._gradient)
    return sample
