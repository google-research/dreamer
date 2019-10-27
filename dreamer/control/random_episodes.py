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

from dreamer import control


def random_episodes(
    env_ctor, num_episodes, num_steps, outdir=None, isolate_envs='none'):
  # If using environment processes or threads, we should also use them here to
  # avoid loading their dependencies into the global name space. This way,
  # their imports will be isolated from the main process and later created envs
  # do not inherit them via global state but import their own copies.
  env, _ = control.create_env(env_ctor, isolate_envs)
  env = control.wrappers.CollectDataset(env, outdir)
  episodes = [] if outdir else None
  while num_episodes > 0 or num_steps > 0:
    policy = lambda env, obs: env.action_space.sample()
    done = False
    obs = env.reset()
    while not done:
      action = policy(env, obs)
      obs, _, done, info = env.step(action)
    episode = env._get_episode()
    episodes.append(episode)
    num_episodes -= 1
    num_steps -= len(episode['reward'])
  try:
    env.close()
  except AttributeError:
    pass
  return episodes
