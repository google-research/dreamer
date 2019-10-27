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

import argparse
import functools
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent))

import ruamel.yaml as yaml
import tensorflow as tf

from dreamer import tools
from dreamer import training
from dreamer.scripts import configs


def process(logdir, args):
  with args.params.unlocked:
    args.params.logdir = logdir
  config = configs.make_config(args.params)
  logdir = pathlib.Path(logdir)
  metrics = tools.Metrics(logdir / 'metrics', workers=5)
  training.utility.collect_initial_episodes(metrics, config)
  tf.reset_default_graph()
  dataset = tools.numpy_episodes.numpy_episodes(
      config.train_dir, config.test_dir, config.batch_shape,
      reader=config.data_reader,
      loader=config.data_loader,
      num_chunks=config.num_chunks,
      preprocess_fn=config.preprocess_fn,
      gpu_prefetch=config.gpu_prefetch)
  metrics = tools.InGraphMetrics(metrics)
  build_graph = tools.bind(training.define_model, logdir, metrics)
  for score in training.utility.train(build_graph, dataset, logdir, config):
    yield score


def main(args):
  experiment = training.Experiment(
      args.logdir,
      process_fn=functools.partial(process, args=args),
      num_runs=args.num_runs,
      ping_every=args.ping_every,
      resume_runs=args.resume_runs)
  for run in experiment:
    for unused_score in run:
      pass


if __name__ == '__main__':
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=pathlib.Path, required=True)
  parser.add_argument('--params', default='{}')
  parser.add_argument('--num_runs', type=int, default=1)
  parser.add_argument('--ping_every', type=int, default=0)
  parser.add_argument('--resume_runs', type=boolean, default=True)
  parser.add_argument('--dmlab_runfiles_path', default=None)
  args_, remaining = parser.parse_known_args()
  params_ = args_.params.replace('#', ',').replace('\\', '')
  args_.params = tools.AttrDict(yaml.safe_load(params_))
  if args_.dmlab_runfiles_path:
    with args_.params.unlocked:
      args_.params.dmlab_runfiles_path = args_.dmlab_runfiles_path
    assert args_.params.dmlab_runfiles_path  # Mark as accessed.
  args_.logdir = args_.logdir and os.path.expanduser(args_.logdir)
  remaining.insert(0, sys.argv[0])
  tf.app.run(lambda _: main(args_), remaining)
