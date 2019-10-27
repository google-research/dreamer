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
from dreamer.control import temporal_difference as td


def reward(state, graph, params):
  features = graph.cell.features_from_state(state)
  reward = graph.heads.reward(features).mean()
  return tf.reduce_sum(reward, 1)


def reward_value(state, graph, params):
  features = graph.cell.features_from_state(state)
  reward = graph.heads.reward(features).mean()
  value = graph.heads.value(features).mean()
  value *= tools.schedule.linear(
      graph.step, params.get('objective_value_ramp', 0))
  return_ = td.lambda_return(
      reward[:, :-1], value[:, :-1], value[:, -1],
      params.get('planner_discount', 0.99),
      params.get('planner_lambda', 0.95),
      axis=1)
  return return_[:, 0]
