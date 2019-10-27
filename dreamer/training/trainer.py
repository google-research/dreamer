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

import collections
import os

import tensorflow as tf

from dreamer import tools


_Phase = collections.namedtuple(
    'Phase',
    'name, writer, op, batch_size, steps, feed, report_every, log_every,'
    'checkpoint_every, restore_every')


class Trainer(object):

  def __init__(self, logdir, config=None):
    self._logdir = logdir
    self._global_step = tf.train.get_or_create_global_step()
    self._step = tf.placeholder(tf.int32, name='step')
    self._phase = tf.placeholder(tf.string, name='phase')
    self._log = tf.placeholder(tf.bool, name='log')
    self._report = tf.placeholder(tf.bool, name='report')
    self._reset = tf.placeholder(tf.bool, name='reset')
    self._phases = []
    # Checkpointing.
    self._loaders = []
    self._savers = []
    self._logdirs = []
    self._checkpoints = []
    self._config = config or tools.AttrDict()

  @property
  def global_step(self):
    return self._global_step

  @property
  def step(self):
    return self._step

  @property
  def phase(self):
    return self._phase

  @property
  def log(self):
    return self._log

  @property
  def reset(self):
    return self._reset

  def add_saver(
      self, include=r'.*', exclude=r'.^', logdir=None, load=True, save=True,
      checkpoint=None):
    variables = tools.filter_variables(include, exclude)
    saver = tf.train.Saver(variables, max_to_keep=1)
    if load:
      self._loaders.append(saver)
    if save:
      self._savers.append(saver)
    self._logdirs.append(logdir or self._logdir)
    if checkpoint is None and self._config.checkpoint_to_load:
      self._checkpoints.append(
          os.path.join(self._logdirs[-1], self._config.checkpoint_to_load))
    else:
      self._checkpoints.append(checkpoint)

  def add_phase(
      self, name, steps, score, summary, batch_size=1,
      report_every=None, log_every=None, checkpoint_every=None,
      restore_every=None, feed=None):
    score = tf.convert_to_tensor(score, tf.float32)
    summary = tf.convert_to_tensor(summary, tf.string)
    feed = feed or {}
    if not score.shape.ndims:
      score = score[None]
    writer = self._logdir and tf.summary.FileWriter(
        os.path.join(self._logdir, name),
        tf.get_default_graph(), flush_secs=30)
    op = self._define_step(name, batch_size, score, summary)
    self._phases.append(_Phase(
        name, writer, op, batch_size, int(steps), feed, report_every,
        log_every, checkpoint_every, restore_every))

  def run(self, max_step=None, sess=None, unused_saver=None):
    for _ in self.iterate(max_step, sess):
      pass

  def iterate(self, max_step=None, sess=None):
    sess = sess or self._create_session()
    with sess:
      self._initialize_variables(
          sess, self._loaders, self._logdirs, self._checkpoints)
      sess.graph.finalize()
      while True:
        global_step = sess.run(self._global_step)
        if max_step and global_step >= max_step:
          break
        phase, epoch, steps_in = self._find_current_phase(global_step)
        phase_step = epoch * phase.steps + steps_in
        if steps_in % phase.steps < phase.batch_size:
          message = '\n' + ('-' * 50) + '\n'
          message += 'Epoch {} phase {} (phase step {}, global step {}).'
          print(message.format(epoch + 1, phase.name, phase_step, global_step))
        # Populate book keeping tensors.
        phase.feed[self._step] = phase_step
        phase.feed[self._phase] = phase.name
        phase.feed[self._reset] = (steps_in < phase.batch_size)
        phase.feed[self._log] = phase.writer and self._is_every_steps(
            phase_step, phase.batch_size, phase.log_every)
        phase.feed[self._report] = self._is_every_steps(
            phase_step, phase.batch_size, phase.report_every)
        summary, mean_score, global_step = sess.run(phase.op, phase.feed)
        if self._is_every_steps(
            phase_step, phase.batch_size, phase.checkpoint_every):
          for saver in self._savers:
            self._store_checkpoint(sess, saver, global_step)
        if self._is_every_steps(
            phase_step, phase.batch_size, phase.report_every):
          print('Score {}.'.format(mean_score))
          yield mean_score
        if summary and phase.writer:
          # We want smaller phases to catch up at the beginnig of each epoch so
          # that their graphs are aligned.
          longest_phase = max(phase_.steps for phase_ in self._phases)
          summary_step = epoch * longest_phase + steps_in
          phase.writer.add_summary(summary, summary_step)
        if self._is_every_steps(
            phase_step, phase.batch_size, phase.restore_every):
          self._initialize_variables(
              sess, self._loaders, self._logdirs, self._checkpoints)

  def _is_every_steps(self, phase_step, batch, every):
    if not every:
      return False
    covered_steps = range(phase_step, phase_step + batch)
    return any((step + 1) % every == 0 for step in covered_steps)

  def _find_current_phase(self, global_step):
    epoch_size = sum(phase.steps for phase in self._phases)
    epoch = int(global_step // epoch_size)
    steps_in = global_step % epoch_size
    for phase in self._phases:
      if steps_in < phase.steps:
        return phase, epoch, steps_in
      steps_in -= phase.steps

  def _define_step(self, name, batch_size, score, summary):
    with tf.variable_scope('phase_{}'.format(name)):
      score_mean = tools.StreamingMean((), tf.float32, 'score_mean')
      score.set_shape((None,))
      with tf.control_dependencies([score, summary]):
        submit_score = score_mean.submit(score)
      with tf.control_dependencies([submit_score]):
        mean_score = tf.cond(self._report, score_mean.clear, float)
        summary = tf.cond(
            self._report,
            lambda: tf.summary.merge([summary, tf.summary.scalar(
                name + '/score', mean_score, family='trainer')]),
            lambda: summary)
        next_step = self._global_step.assign_add(batch_size)
      with tf.control_dependencies([summary, mean_score, next_step]):
        return (
            tf.identity(summary),
            tf.identity(mean_score),
            tf.identity(next_step))

  def _create_session(self):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    try:
      return tf.Session('local', config=config)
    except tf.errors.NotFoundError:
      return tf.Session(config=config)

  def _initialize_variables(self, sess, savers, logdirs, checkpoints):
    sess.run(tf.group(
        tf.local_variables_initializer(),
        tf.global_variables_initializer()))
    assert len(savers) == len(logdirs) == len(checkpoints)
    for i, (saver, logdir, checkpoint) in enumerate(
        zip(savers, logdirs, checkpoints)):
      logdir = os.path.expanduser(logdir)
      state = tf.train.get_checkpoint_state(logdir)
      if checkpoint:
        checkpoint = os.path.join(logdir, checkpoint)
      if not checkpoint and state and state.model_checkpoint_path:
        checkpoint = state.model_checkpoint_path
      if checkpoint:
        saver.restore(sess, checkpoint)

  def _store_checkpoint(self, sess, saver, global_step):
    if not self._logdir or not saver:
      return
    tf.gfile.MakeDirs(self._logdir)
    filename = os.path.join(self._logdir, 'model.ckpt')
    saver.save(sess, filename, global_step)
