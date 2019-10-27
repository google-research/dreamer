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

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import summary_op_util


def encode_gif(images, fps):
  from subprocess import Popen, PIPE
  h, w, c = images[0].shape
  cmd = [
      'ffmpeg', '-y',
      '-f', 'rawvideo',
      '-vcodec', 'rawvideo',
      '-r', '%.02f' % fps,
      '-s', '%dx%d' % (w, h),
      '-pix_fmt', {1: 'gray', 3: 'rgb24'}[c],
      '-i', '-',
      '-filter_complex',
      '[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
      '-r', '%.02f' % fps,
      '-f', 'gif',
      '-']
  proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  for image in images:
    proc.stdin.write(image.tostring())
  out, err = proc.communicate()
  if proc.returncode:
    err = '\n'.join([' '.join(cmd), err.decode('utf8')])
    raise IOError(err)
  del proc
  return out


def py_gif_summary(tag, images, max_outputs, fps):
  is_bytes = isinstance(tag, bytes)
  if is_bytes:
    tag = tag.decode('utf-8')
  images = np.asarray(images)
  if images.dtype != np.uint8:
    raise ValueError('Tensor must have dtype uint8 for gif summary.')
  if images.ndim != 5:
    raise ValueError('Tensor must be 5-D for gif summary.')
  batch_size, _, height, width, channels = images.shape
  if channels not in (1, 3):
    raise ValueError('Tensors must have 1 or 3 channels for gif summary.')
  summ = tf.Summary()
  num_outputs = min(batch_size, max_outputs)
  for i in range(num_outputs):
    image_summ = tf.Summary.Image()
    image_summ.height = height
    image_summ.width = width
    image_summ.colorspace = channels  # 1: grayscale, 3: RGB
    try:
      image_summ.encoded_image_string = encode_gif(images[i], fps)
    except (IOError, OSError) as e:
      print(
          'Unable to encode images to a gif string because either ffmpeg is '
          'not installed or ffmpeg returned an error: %s. Falling back to an '
          'image summary of the first frame in the sequence.', e)
      try:
        from PIL import Image  # pylint: disable=g-import-not-at-top
        import io  # pylint: disable=g-import-not-at-top
        with io.BytesIO() as output:
          Image.fromarray(images[i][0]).save(output, 'PNG')
          image_summ.encoded_image_string = output.getvalue()
      except Exception:
        print('Gif summaries requires ffmpeg or PIL to be installed: %s', e)
        image_summ.encoded_image_string = (
            ''.encode('utf-8') if is_bytes else '')
    if num_outputs == 1:
      summ_tag = tag
    else:
      summ_tag = '{}/{}'.format(tag, i)
    summ.value.add(tag=summ_tag, image=image_summ)
  summ_str = summ.SerializeToString()
  return summ_str


def gif_summary(name, tensor, max_outputs, fps, collections=None, family=None):
  tensor = tf.convert_to_tensor(tensor)
  if tensor.dtype in (tf.float32, tf.float64):
    tensor = tf.cast(255.0 * tensor, tf.uint8)
  with summary_op_util.summary_scope(
      name, family, values=[tensor]) as (tag, scope):
    val = tf.py_func(
        py_gif_summary,
        [tag, tensor, max_outputs, fps],
        tf.string,
        stateful=False,
        name=scope)
    summary_op_util.collect(val, collections, [tf.GraphKeys.SUMMARIES])
  return val
