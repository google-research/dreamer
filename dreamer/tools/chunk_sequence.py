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

from dreamer.tools import nested


def chunk_sequence(sequence, chunk_length, randomize=True, num_chunks=None):
  if 'length' in sequence:
    length = sequence.pop('length')
  else:
    length = tf.shape(nested.flatten(sequence)[0])[0]
  if randomize:
    if not num_chunks:
      num_chunks = tf.maximum(1, length // chunk_length - 1)
    else:
      num_chunks = num_chunks + 0 * length
    used_length = num_chunks * chunk_length
    max_offset = length - used_length
    offset = tf.random_uniform((), 0, max_offset + 1, dtype=tf.int32)
  else:
    if num_chunks is None:
      num_chunks = length // chunk_length
    else:
      num_chunks = num_chunks + 0 * length
    used_length = num_chunks * chunk_length
    offset = 0
  clipped = nested.map(
      lambda tensor: tensor[offset: offset + used_length],
      sequence)
  chunks = nested.map(
      lambda tensor: tf.reshape(
          tensor, [num_chunks, chunk_length] + tensor.shape[1:].as_list()),
      clipped)
  return chunks
