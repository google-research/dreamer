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

import functools
import os
import random
import time

from scipy.ndimage import interpolation
import numpy as np
import tensorflow as tf

from dreamer.tools import attr_dict
from dreamer.tools import chunk_sequence


def numpy_episodes(
    train_dir, test_dir, shape, reader=None, loader=None, num_chunks=None,
    preprocess_fn=None, gpu_prefetch=False):
  reader = reader or episode_reader
  loader = loader or cache_loader
  try:
    dtypes, shapes = _read_spec(reader, train_dir)
  except ZeroDivisionError:
    dtypes, shapes = _read_spec(reader, test_dir)
  train = tf.data.Dataset.from_generator(
      functools.partial(loader, reader, train_dir),
      dtypes, shapes)
  test = tf.data.Dataset.from_generator(
      functools.partial(cache_loader, reader, test_dir, shape[0]),
      dtypes, shapes)
  chunking = lambda x: tf.data.Dataset.from_tensor_slices(
      chunk_sequence.chunk_sequence(x, shape[1], True, num_chunks))
  train = train.flat_map(chunking)
  train = train.batch(shape[0], drop_remainder=True)
  if preprocess_fn:
    train = train.map(preprocess_fn, tf.data.experimental.AUTOTUNE)
  if gpu_prefetch:
    train = train.apply(tf.data.experimental.copy_to_device('/gpu:0'))
  train = train.prefetch(tf.data.experimental.AUTOTUNE)
  test = test.flat_map(chunking)
  test = test.batch(shape[0], drop_remainder=True)
  if preprocess_fn:
    test = test.map(preprocess_fn, tf.data.experimental.AUTOTUNE)
  if gpu_prefetch:
    test = test.apply(tf.data.experimental.copy_to_device('/gpu:0'))
  test = test.prefetch(tf.data.experimental.AUTOTUNE)
  return attr_dict.AttrDict(train=train, test=test)


def cache_loader(reader, directory, every):
  cache = {}
  while True:
    episodes = _sample(cache.values(), every)
    for episode in _permuted(episodes, every):
      yield episode
    filenames = tf.gfile.Glob(os.path.join(directory, '*.npz'))
    filenames = [filename for filename in filenames if filename not in cache]
    for filename in filenames:
      cache[filename] = reader(filename)


def recent_loader(reader, directory, every):
  recent = {}
  cache = {}
  while True:
    episodes = []
    episodes += _sample(recent.values(), every // 2)
    episodes += _sample(cache.values(), every // 2)
    for episode in _permuted(episodes, every):
      yield episode
    cache.update(recent)
    recent = {}
    filenames = tf.gfile.Glob(os.path.join(directory, '*.npz'))
    filenames = [filename for filename in filenames if filename not in cache]
    for filename in filenames:
      recent[filename] = reader(filename)


def window_loader(reader, directory, window, every):
  cache = {}
  while True:
    episodes = _sample(cache.values(), every)
    for episode in _permuted(episodes, every):
      yield episode
    filenames = tf.gfile.Glob(os.path.join(directory, '*.npz'))
    filenames = sorted(filenames)[-window:]
    for filename in filenames:
      if filename not in cache:
        cache[filename] = reader(filename)
    for key in list(cache.keys()):
      if key not in filenames:
        del cache[key]


def reload_loader(reader, directory):
  directory = os.path.expanduser(directory)
  while True:
    filenames = tf.gfile.Glob(os.path.join(directory, '*.npz'))
    random.shuffle(filenames)
    for filename in filenames:
      yield reader(filename)


def dummy_loader(reader, directory):
  random = np.random.RandomState(seed=0)
  dtypes, shapes, length = _read_spec(reader, directory, True, True)
  while True:
    episode = {}
    for key in dtypes:
      dtype, shape = dtypes[key], (length,) + shapes[key][1:]
      if dtype in (np.float32, np.float64):
        episode[key] = random.uniform(0, 1, shape).astype(dtype)
      elif dtype in (np.int32, np.int64, np.uint8):
        episode[key] = random.uniform(0, 255, shape).astype(dtype)
      else:
        raise NotImplementedError('Unsupported dtype {}.'.format(dtype))
    yield episode


def episode_reader(
    filename, resize=None, max_length=None, action_noise=None,
    clip_rewards=False, pcont_scale=None):
  try:
    with tf.gfile.Open(filename, 'rb') as file_:
      episode = np.load(file_)
  except (IOError, ValueError):
    # Try again one second later, in case the file was still being written.
    time.sleep(1)
    with tf.gfile.Open(filename, 'rb') as file_:
      episode = np.load(file_)
  episode = {key: _convert_type(episode[key]) for key in episode.keys()}
  episode['return'] = np.cumsum(episode['reward'])
  if 'reward_mask' not in episode:
    episode['reward_mask'] = np.ones_like(episode['reward'])[..., None]
  if max_length:
    episode = {key: value[:max_length] for key, value in episode.items()}
  if resize and resize != 1:
    factors = (1, resize, resize, 1)
    episode['image'] = interpolation.zoom(episode['image'], factors)
  if action_noise:
    seed = np.fromstring(filename, dtype=np.uint8)
    episode['action'] += np.random.RandomState(seed).normal(
        0, action_noise, episode['action'].shape)
  if clip_rewards is False:
    pass
  elif clip_rewards == 'sign':
    episode['reward'] = np.sign(episode['reward'])
  elif clip_rewards == 'tanh':
    episode['reward'] = np.tanh(episode['reward'])
  else:
    raise NotImplementedError(clip_rewards)
  if pcont_scale is not None:
    episode['pcont'] *= pcont_scale
  return episode


def _read_spec(
    reader, directory, return_length=False, numpy_types=False):
  episodes = reload_loader(reader, directory)
  episode = next(episodes)
  episodes.close()
  dtypes = {key: value.dtype for key, value in episode.items()}
  if not numpy_types:
    dtypes = {key: tf.as_dtype(value) for key, value in dtypes.items()}
  shapes = {key: value.shape for key, value in episode.items()}
  shapes = {key: (None,) + shape[1:] for key, shape in shapes.items()}
  if return_length:
    length = len(episode[list(shapes.keys())[0]])
    return dtypes, shapes, length
  else:
    return dtypes, shapes


def _convert_type(array):
  if array.dtype == np.float64:
    return array.astype(np.float32)
  if array.dtype == np.int64:
    return array.astype(np.int32)
  return array


def _sample(sequence, amount):
  sequence = list(sequence)
  amount = min(amount, len(sequence))
  return random.sample(sequence, amount)


def _permuted(sequence, amount):
  sequence = list(sequence)
  if not sequence:
    return
  index = 0
  while True:
    for element in np.random.permutation(sequence):
      if index >= amount:
        return
      yield element
      index += 1
