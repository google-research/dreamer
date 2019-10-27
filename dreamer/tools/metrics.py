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

from concurrent import futures
import collections
import datetime
import itertools
import json
import pathlib
import re
import uuid

# import imageio
import numpy as np
import tensorflow as tf


class InGraphMetrics:

  def __init__(self, metrics):
    self._metrics = metrics

  def set_tags(self, **tags):
    keys, values = zip(*sorted(tags.items(), key=lambda x: x[0]))
    def inner(*values):
      parsed = []
      for index, value in enumerate(values):
        if value.dtype == tf.string:
          parsed.append(value.numpy().decode('utf-8'))
        elif value.dtype in (tf.float32, tf.float64):
          parsed.append(float(value.numpy()))
        elif value.dtype in (tf.int32, tf.int64):
          parsed.append(int(value.numpy()))
        else:
          raise NotImplementedError(value.dtype)
      tags = dict(zip(keys, parsed))
      return self._metrics.set_tags(**tags)
    # String tensors in tf.py_function are only supported on CPU.
    with tf.device('/cpu:0'):
      return tf.py_function(inner, values, [], 'set_tags')

  def reset_tags(self):
    return tf.py_function(
        self._metrics.reset_tags, [], [], 'reset_tags')

  def flush(self):
    def inner():
      _ = self._metrics.flush()
    return tf.py_function(inner, [], [], 'flush')

  def add_scalar(self, name, value):
    assert len(value.shape) == 0, (name, value)
    def inner(value):
      self._metrics.add_scalar(name, value.numpy())
    return tf.py_function(inner, [value], [], 'add_scalar_' + name)

  def add_scalars(self, name, value):
    assert len(value.shape) == 1, (name, value)
    def inner(value):
      self._metrics.add_scalars(name, value.numpy())
    return tf.py_function(inner, [value], [], 'add_scalars_' + name)

  def add_tensor(self, name, value):
    def inner(value):
      self._metrics.add_tensor(name, value.numpy())
    return tf.py_function(inner, [value], [], 'add_tensor_' + name)


class Metrics:

  def __init__(self, directory, workers=None):
    assert workers is None or isinstance(workers, int)
    self._directory = pathlib.Path(directory).expanduser()
    self._records = collections.defaultdict(collections.deque)
    self._created_directories = set()
    self._values = {}
    self._tags = {}
    self._writers = {
        'npy': np.save,
        # 'png': imageio.imwrite,
        # 'jpg': imageio.imwrite,
        # 'bmp': imageio.imwrite,
        # 'gif': functools.partial(imageio.mimwrite, fps=30),
        # 'mp4': functools.partial(imageio.mimwrite, fps=30),
    }
    if workers:
      self._pool = futures.ThreadPoolExecutor(max_workers=workers)
    else:
      self._pool = None
    self._last_futures = []

  @property
  def names(self):
    names = set(self._records.keys())
    for catalogue in self._directory.glob('**/records.jsonl'):
      names.add(self._path_to_name(catalogue.parent))
    return sorted(names)

  def set_tags(self, **kwargs):
    reserved = ('value', 'name')
    if any(key in reserved for key in kwargs):
      message = "Reserved keys '{}' and cannot be used for tags"
      raise KeyError(message.format(', '.join(reserved)))
    for key, value in kwargs.items():
      key = json.loads(json.dumps(key))
      value = json.loads(json.dumps(value))
      self._tags[key] = value

  def reset_tags(self):
    self._tags = {}

  def add_scalar(self, name, value):
    self._validate_name(name)
    record = self._tags.copy()
    record['value'] = float(value)
    self._records[name].append(record)

  def add_scalars(self, name, values):
    self._validate_name(name)
    for value in values:
      record = self._tags.copy()
      record['value'] = float(value)
      self._records[name].append(record)

  def add_tensor(self, name, value, format='npy'):
    assert format in ('npy',)
    self._validate_name(name)
    record = self._tags.copy()
    record['filename'] = self._random_filename('npy')
    self._values[record['filename']] = value
    self._records[name].append(record)

  def add_image(self, name, value, format='png'):
    assert format in ('png', 'jpg', 'bmp')
    self._validate_name(name)
    record = self._tags.copy()
    record['filename'] = self._random_filename(format)
    self._values[record['filename']] = value
    self._records[name].append(record)

  def add_video(self, name, value, format='gif'):
    assert format in ('gif', 'mp4')
    self._validate_name(name)
    record = self._tags.copy()
    record['filename'] = self._random_filename(format)
    self._values[record['filename']] = value
    self._records[name].append(record)

  def flush(self, blocking=None):
    if blocking is False and not self._pool:
      message = 'Create with workers argument for non-blocking flushing.'
      raise ValueError(message)
    for future in self._last_futures or []:
      try:
        future.result()
      except Exception as e:
        message = 'Previous asynchronous flush failed.'
        raise RuntimeError(message) from e
      self._last_futures = None
    jobs = []
    for name in self._records:
      records = list(self._records[name])
      if not records:
        continue
      self._records[name].clear()
      filename = self._name_to_path(name) / 'records.jsonl'
      jobs += [(self._append_catalogue, filename, records)]
      for record in records:
        jobs.append((self._write_file, name, record))
    if self._pool and not blocking:
      futures = []
      for job in jobs:
        futures.append(self._pool.submit(*job))
      self._last_futures = futures.copy()
      return futures
    else:
      for job in jobs:
        job[0](*job[1:])

  def query(self, pattern=None, **tags):
    pattern = pattern and re.compile(pattern)
    for name in self.names:
      if pattern and not pattern.search(name):
        continue
      for record in self._read_records(name):
        if {k: v for k, v in record.items() if k in tags} != tags:
          continue
        record['name'] = name
        yield record

  def _validate_name(self, name):
    assert isinstance(name, str) and name
    if re.search(r'[^a-z0-9/_-]+', name):
      message = (
          "Invalid metric name '{}'. Names must contain only lower case "
          "letters, digits, dashes, underscores, and forward slashes.")
      raise NameError(message.format(name))

  def _name_to_path(self, name):
    return self._directory.joinpath(*name.split('/'))

  def _path_to_name(self, path):
    return '/'.join(path.relative_to(self._directory).parts)

  def _read_records(self, name):
    path = self._name_to_path(name)
    catalogue = path / 'records.jsonl'
    records = self._records[name]
    if catalogue.exists():
      records = itertools.chain(records, self._load_catalogue(catalogue))
    for record in records:
      if 'filename' in record:
        record['filename'] = path / record['filename']
      yield record

  def _write_file(self, name, record):
    if 'filename' not in record:
      return
    format = pathlib.Path(record['filename']).suffix.lstrip('.')
    if format not in self._writers:
      raise TypeError('Trying to write unknown format {}'.format(format))
    value = self._values.pop(record['filename'])
    filename = self._name_to_path(name) / record['filename']
    self._ensure_directory(filename.parent)
    self._writers[format](filename, value)

  def _load_catalogue(self, filename):
    rows = [json.loads(line) for line in filename.open('r')]
    message = 'Metrics files do not contain lists of mappings'
    if not isinstance(rows, list):
      raise TypeError(message)
    if not all(isinstance(row, dict) for row in rows):
      raise TypeError(message)
    return rows

  def _append_catalogue(self, filename, records):
    self._ensure_directory(filename.parent)
    records = [
        collections.OrderedDict(sorted(record.items(), key=lambda x: x[0]))
        for record in records]
    content = ''.join([json.dumps(record) + '\n' for record in records])
    with filename.open('a') as f:
      f.write(content)

  def _ensure_directory(self, directory):
    if directory in self._created_directories:
      return
    # We first attempt to create the directory and afterwards add it to the set
    # of created directories. This means multiple workers could attempt to
    # create the directory at the same time, which is better than one trying to
    # create a file while another has not yet created the directory.
    directory.mkdir(parents=True, exist_ok=True)
    self._created_directories.add(directory)

  def _random_filename(self, extension):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4()).replace('-', '')
    filename = '{}-{}.{}'.format(timestamp, identifier, extension)
    return filename
