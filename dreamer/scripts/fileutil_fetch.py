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

import argparse
import pathlib
import re
import threading

import sh


def execute_commands(commands, parallel):
  semaphore = threading.Semaphore(parallel)
  def done_fn(cmd, success, exit_code):
    print(cmd._foo)
    semaphore.release()
  running = []
  for command in commands:
    semaphore.acquire()
    running.append(command(_bg=True, _done=done_fn))
    running[-1]._foo = command._foo
  failures = 0
  outputs = []
  for command in running:
    try:
      command.wait()
      outputs.append(command.stdout.decode('utf-8'))
    except sh.ErrorReturnCode as e:
      print(e)
      failures += 1
  print('')
  return outputs, failures


def main(args):
  pattern = re.compile(args.pattern)
  filenames = sh.fileutil.ls('-R', args.indir)
  filenames = [filename.strip() for filename in filenames]
  filenames = [filename for filename in filenames if pattern.search(filename)]
  print('Found', len(filenames), 'filenames.')
  commands = []
  for index, filename in enumerate(filenames):
    relative = pathlib.Path(filename).relative_to(args.indir)
    destination = args.outdir / relative
    if not args.overwrite and destination.exists():
      continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = [filename, destination]
    if args.overwrite:
      flags = ['-f'] + flags
    command = sh.fileutil.cp.bake(*flags)
    command._foo = f'{index + 1}/{len(filenames)} {relative}'
    commands.append(command)
  print(f'Executing {args.parallel} in parallel.')
  execute_commands(commands, args.parallel)


if __name__ == '__main__':
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--indir', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--parallel', type=int, default=100)
  # parser.add_argument('--pattern', type=str, default='.*/records.yaml$')
  parser.add_argument(
      '--pattern', type=str,
      default='.*/(return|length)/records.jsonl$')
  parser.add_argument('--subdir', type=boolean, default=True)
  parser.add_argument('--overwrite', type=boolean, default=False)
  _args = parser.parse_args()
  if _args.subdir:
    _args.outdir /= _args.indir.stem
  main(_args)
