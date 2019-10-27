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

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import tensorflow as tf
from dreamer import tools
from dreamer.scripts import train


class DreamerTest(tf.test.TestCase):

  def test_dreamer(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['dreamer', 'debug'],
            tasks=['dummy'],
            isolate_envs='none',
            max_steps=30,
            train_planner='policy_sample',
            test_planner='policy_mode',
            planner_objective='reward_value',
            action_head=True,
            value_head=True,
            imagination_horizon=3),
        ping_every=0,
        resume_runs=False)
    train.main(args)

  def test_dreamer_discrete(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['dreamer', 'debug'],
            tasks=['dummy'],
            isolate_envs='none',
            max_steps=30,
            train_planner='policy_sample',
            test_planner='policy_mode',
            planner_objective='reward_value',
            action_head=True,
            value_head=True,
            imagination_horizon=3,
            action_head_dist='onehot_score',
            action_noise_type='epsilon_greedy'),
        ping_every=0,
        resume_runs=False)
    train.main(args)

  def test_dreamer_target(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['dreamer', 'debug'],
            tasks=['dummy'],
            isolate_envs='none',
            max_steps=30,
            train_planner='policy_sample',
            test_planner='policy_mode',
            planner_objective='reward_value',
            action_head=True,
            value_head=True,
            value_target_head=True,
            imagination_horizon=3),
        ping_every=0,
        resume_runs=False)
    train.main(args)

  def test_no_value(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['actor', 'debug'],
            tasks=['dummy'],
            isolate_envs='none',
            max_steps=30,
            imagination_horizon=3),
        ping_every=0,
        resume_runs=False)
    train.main(args)

  def test_planet(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['planet', 'debug'],
            tasks=['dummy'],
            isolate_envs='none',
            max_steps=30,
            planner_horizon=3),
        ping_every=0,
        resume_runs=False)
    train.main(args)


if __name__ == '__main__':
  tf.test.main()
