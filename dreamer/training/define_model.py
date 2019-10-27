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
import functools

import tensorflow as tf

from dreamer import tools
from dreamer.training import define_summaries
from dreamer.training import utility


def define_model(logdir, metrics, data, trainer, config):
  print('Build TensorFlow compute graph.')
  dependencies = []
  cleanups = []
  step = trainer.step
  global_step = trainer.global_step
  phase = trainer.phase
  timestamp = tf.py_func(
      lambda: datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S'),
      [], tf.string)
  dependencies.append(metrics.set_tags(
      global_step=global_step, step=step, phase=phase,
      time=timestamp))

  # Instantiate network blocks. Note, this initialization would be expensive
  # when using tf.function since it would run at every step.
  try:
    cell = config.cell()
  except TypeError:
    cell = config.cell(action_size=data['action'].shape[-1].value)
  kwargs = dict(create_scope_now_=True)
  encoder = tf.make_template('encoder', config.encoder, **kwargs)
  heads = tools.AttrDict(_unlocked=True)
  raw_dummy_features = cell.features_from_state(
      cell.zero_state(1, tf.float32))[:, None]
  for key, head in config.heads.items():
    name = 'head_{}'.format(key)
    kwargs = dict(create_scope_now_=True)
    if key in data:
      kwargs['data_shape'] = data[key].shape[2:].as_list()
    if key == 'action_target':
      kwargs['data_shape'] = data['action'].shape[2:].as_list()
    if key == 'cpc':
      kwargs['data_shape'] = [cell.feature_size]
      dummy_features = encoder(data)[:1, :1]
    else:
      dummy_features = raw_dummy_features
    heads[key] = tf.make_template(name, head, **kwargs)
    heads[key](dummy_features)  # Initialize weights.

  # Update target networks.
  if 'value_target' in heads:
    dependencies.append(tools.track_network(
        trainer, config.batch_shape[0],
        r'.*/head_value/.*', r'.*/head_value_target/.*',
        config.value_target_period, config.value_target_update))
  if 'value_target_2' in heads:
    dependencies.append(tools.track_network(
        trainer, config.batch_shape[0],
        r'.*/head_value/.*', r'.*/head_value_target_2/.*',
        config.value_target_period, config.value_target_update))
  if 'action_target' in heads:
    dependencies.append(tools.track_network(
        trainer, config.batch_shape[0],
        r'.*/head_action/.*', r'.*/head_action_target/.*',
        config.action_target_period, config.action_target_update))

  # Apply and optimize model.
  embedded = encoder(data)
  with tf.control_dependencies(dependencies):
    embedded = tf.identity(embedded)
  graph = tools.AttrDict(locals())
  prior, posterior = tools.unroll.closed_loop(
      cell, embedded, data['action'], config.debug)
  objectives = utility.compute_objectives(
      posterior, prior, data, graph, config)
  summaries, grad_norms = utility.apply_optimizers(
      objectives, trainer, config)
  dependencies += summaries

  # Active data collection.
  with tf.variable_scope('collection'):
    with tf.control_dependencies(dependencies):  # Make sure to train first.
      for name, params in config.train_collects.items():
        schedule = tools.schedule.binary(
            step, config.batch_shape[0],
            params.steps_after, params.steps_every, params.steps_until)
        summary, _ = tf.cond(
            tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
            functools.partial(
                utility.simulate, metrics, config, params, graph, cleanups,
                gif_summary=False, name=name),
            lambda: (tf.constant(''), tf.constant(0.0)),
            name='should_collect_' + name)
        summaries.append(summary)
        dependencies.append(summary)

  # Compute summaries.
  graph = tools.AttrDict(locals())
  summary, score = tf.cond(
      trainer.log,
      lambda: define_summaries.define_summaries(graph, config, cleanups),
      lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
      name='summaries')
  summaries = tf.summary.merge([summaries, summary])
  dependencies.append(utility.print_metrics(
      {ob.name: ob.value for ob in objectives},
      step, config.print_metrics_every, 2, 'objectives'))
  dependencies.append(utility.print_metrics(
      grad_norms, step, config.print_metrics_every, 2, 'grad_norms'))
  dependencies.append(tf.cond(trainer.log, metrics.flush, tf.no_op))
  with tf.control_dependencies(dependencies):
    score = tf.identity(score)
  return score, summaries, cleanups
