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

import datetime
import itertools
import os
import pickle
import six
import sys
import threading
import time
import traceback
import uuid

import numpy as np
import tensorflow as tf


class StopExperiment(Exception):
  pass


class WorkerConflict(Exception):
  pass


class SkipRun(Exception):
  pass


class Experiment(object):

  def __init__(
      self, basedir, process_fn, start_fn=None, resume_fn=None,
      num_runs=None, worker_name=None, ping_every=30, resume_runs=True):
    self._basedir = basedir
    self._process_fn = process_fn
    self._start_fn = start_fn
    self._resume_fn = resume_fn
    self._num_runs = num_runs
    self._worker_name = worker_name or str(uuid.uuid4())
    self._ping_every = ping_every
    self._ping_stale = ping_every and 2 * ping_every
    self._resume_runs = resume_runs

  def __iter__(self):
    for current_run in self._generate_run_numbers():
      logdir = self._basedir and os.path.join(
          self._basedir, '{:05}'.format(current_run))
      try:
        run = Run(
            logdir, self._process_fn, self._start_fn, self._resume_fn,
            self._worker_name, self._ping_every, self._ping_stale,
            self._resume_runs)
        yield run
      except SkipRun:
        continue
      except StopExperiment:
        print('Stopping.')
        break
    print('All runs completed.')

  def _generate_run_numbers(self):
    if self._num_runs:
      # Don't wait initially and see if there are runs that are already stale.
      runs = np.random.permutation(range(self._num_runs))
      for run in runs:
        yield run + 1
      # At the end, wait for all dead runs to become stale, and pick them up.
      # This is necessary for complete runs of workers that died very recently.
      if self._ping_stale:
        time.sleep(self._ping_stale)
        for run in runs:
          yield run + 1
    else:
      # For infinite runs, we want to always finish started jobs first.
      # Therefore, we need to wait for them to become stale in the beginning.
      if self._ping_stale:
        time.sleep(self._ping_stale)
      for run in itertools.count():
        yield run + 1


class Run(object):

  def __init__(
      self, logdir, process_fn, start_fn, resume_fn, worker_name,
      ping_every=30, ping_stale=60, reuse_if_exists=True):
    self._logdir = os.path.expanduser(logdir)
    self._process_fn = process_fn
    self._worker_name = worker_name
    self._ping_every = ping_every
    self._ping_stale = ping_stale
    self._logger = self._create_logger()
    try:
      if self._should_start():
        self._claim()
        self._logger.info('Start.')
        self._init_fn = start_fn
      elif reuse_if_exists and self._should_resume():
        self._claim()
        self._logger.info('Resume.')
        self._init_fn = resume_fn
      else:
        raise SkipRun
    except WorkerConflict:
      self._logger.info('Leave to other worker.')
      raise SkipRun
    self._thread = None
    self._running = [True]
    self._thread = threading.Thread(target=self._store_ping_thread)
    self._thread.daemon = True  # Terminate with main thread.
    self._thread.start()

  def __iter__(self):
    try:
      args = self._init_fn and self._init_fn(self._logdir)
      if args is None:
        args = ()
      if not isinstance(args, tuple):
        args = (args,)
      for value in self._process_fn(self._logdir, *args):
        if not self._running[0]:
          break
        yield value
      self._logger.info('Done.')
      self._store_done()
    except WorkerConflict:
      self._logging.warn('Unexpected takeover.')
      raise SkipRun
    except Exception as e:
      exc_info = sys.exc_info()
      self._handle_exception(e)
      six.reraise(*exc_info)
    finally:
      self._running[0] = False
      self._thread and self._thread.join()

  def _should_start(self):
    if not self._logdir:
      return True
    if tf.gfile.Exists(os.path.join(self._logdir, 'PING')):
      return False
    if tf.gfile.Exists(os.path.join(self._logdir, 'DONE')):
      return False
    return True

  def _should_resume(self):
    if not self._logdir:
      return False
    if tf.gfile.Exists(os.path.join(self._logdir, 'DONE')):
      # self._logger.debug('Already done.')
      return False
    if not tf.gfile.Exists(os.path.join(self._logdir, 'PING')):
      # self._logger.debug('Not started yet.')
      return False
    last_worker, last_ping = self._read_ping()
    if last_worker != self._worker_name and last_ping < self._ping_stale:
      # self._logger.debug('Already in progress.')
      return False
    return True

  def _claim(self):
    if not self._logdir:
      return False
    self._store_ping(overwrite=True)
    if self._ping_every:
      time.sleep(self._ping_every)
    if self._read_ping()[0] != self._worker_name:
      raise WorkerConflict
    self._store_ping()

  def _store_done(self):
    if not self._logdir:
      return
    with tf.gfile.Open(os.path.join(self._logdir, 'DONE'), 'w') as file_:
      file_.write('\n')

  def _store_fail(self, message):
    if not self._logdir:
      return
    with tf.gfile.Open(os.path.join(self._logdir, 'FAIL'), 'w') as file_:
      file_.write(message + '\n')

  def _read_ping(self):
    if not tf.gfile.Exists(os.path.join(self._logdir, 'PING')):
      return None, None
    try:
      with tf.gfile.Open(os.path.join(self._logdir, 'PING'), 'rb') as file_:
        last_worker, last_ping = pickle.load(file_)
      duration = (datetime.datetime.utcnow() - last_ping).total_seconds()
      return last_worker, duration
    except (EOFError, IOError, tf.errors.NotFoundError):
      raise WorkerConflict

  def _store_ping(self, overwrite=False):
    if not self._logdir:
      return
    try:
      last_worker, _ = self._read_ping()
      if last_worker is None:
        self._logger.info("Create directory '{}'.".format(self._logdir))
        tf.gfile.MakeDirs(self._logdir)
      elif last_worker != self._worker_name and not overwrite:
        raise WorkerConflict
      # self._logger.debug('Store ping.')
      with tf.gfile.Open(os.path.join(self._logdir, 'PING'), 'wb') as file_:
        pickle.dump((self._worker_name, datetime.datetime.utcnow()), file_)
    except (EOFError, IOError, tf.errors.NotFoundError):
      raise WorkerConflict

  def _store_ping_thread(self):
    if not self._ping_every:
      return
    try:
      last_write = time.time()
      self._store_ping(self._logdir)
      while self._running[0]:
        if time.time() >= last_write + self._ping_every:
            last_write = time.time()
            self._store_ping(self._logdir)
        # Only wait short times to quickly react to abort.
        time.sleep(0.01)
    except WorkerConflict:
      self._running[0] = False

  def _handle_exception(self, exception):
    message = ''.join(traceback.format_exception(*sys.exc_info()))
    self._logger.warning('Exception:\n{}'.format(message))
    self._logger.warning('Failed.')
    try:
      self._store_done()
      self._store_fail(message)
    except Exception:
      message = ''.join(traceback.format_exception(*sys.exc_info()))
      template = 'Exception in exception handler:\n{}'
      self._logger.warning(template.format(message))

  def _create_logger(self):
    run_name = self._logdir and os.path.basename(self._logdir)
    methods = {}
    for name in 'debug info warning'.split():
      methods[name] = lambda unused_self, message: print(
          'Worker {} run {}: {}'.format(self._worker_name, run_name, message))
    return type('PrefixedLogger', (object,), methods)()
