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


def encoder(obs, keys=None, num_layers=3, units=300, activation=tf.nn.relu):
  if not keys:
    keys = [key for key in obs.keys() if key != 'image']
  inputs = tf.concat([obs[key] for key in keys], -1)
  hidden = tf.reshape(inputs, [-1] + inputs.shape[2:].as_list())
  for _ in range(num_layers):
    hidden = tf.layers.dense(hidden, units, activation)
  hidden = tf.reshape(hidden, tools.shape(inputs)[:2] + [
      hidden.shape[1].value])
  return hidden
