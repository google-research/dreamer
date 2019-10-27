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


class IsolateEnvsTest(tf.test.TestCase):

  def test_dummy_thread(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['dreamer', 'debug'],
            tasks=['dummy'],
            isolate_envs='thread',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    train.main(args)

  def test_dm_control_thread(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['dreamer', 'debug'],
            tasks=['cup_catch'],
            isolate_envs='thread',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    train.main(args)

  def test_atari_thread(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        params=tools.AttrDict(
            defaults=['dreamer', 'debug'],
            tasks=['atari_pong'],
            isolate_envs='thread',
            action_head_dist='onehot_score',
            action_noise_type='epsilon_greedy',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    train.main(args)

  # def test_dmlab_thread(self):
  #   args = tools.AttrDict(
  #       logdir=self.get_temp_dir(),
  #       num_runs=1,
  #       params=tools.AttrDict(
  #           defaults=['dreamer', 'debug'],
  #           tasks=['dmlab_collect'],
  #           isolate_envs='thread',
  #           action_head_dist='onehot_score',
  #           action_noise_type='epsilon_greedy',
  #           max_steps=30),
  #       ping_every=0,
  #       resume_runs=False)
  #   train.main(args)


if __name__ == '__main__':
  tf.test.main()
