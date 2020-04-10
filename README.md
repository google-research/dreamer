# Dream to Control

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, Mohammad Norouzi

**Note:** This is the original implementation. To build upon Dreamer, we
recommend the newer implementation of [Dreamer in TensorFlow
2](https://github.com/danijar/dreamer). It is substantially simpler
and faster while replicating the results.

<img width="100%" src="https://imgur.com/x4NUHXl.gif">

Implementation of Dreamer, the reinforcement learning agent introduced in
[Dream to Control: Learning Behaviors by Latent Imagination][paper]. Dreamer
learns long-horizon behaviors from images purely by latent imagination. For
this, it backpropagates value estimates through trajectories imagined in the
compact latent space of a learned world model. Dreamer solves visual control
tasks using substantilly fewer episodes than strong model-free agents.

If you find this open source release useful, please reference in your paper:

```
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```

## Method

![Dreamer model diagram](https://imgur.com/JrXC4rh.png)

Dreamer learns a world model from past experience that can predict into the
future. It then learns action and value models in its compact latent space. The
value model optimizes Bellman consistency of imagined trajectories. The action
model maximizes value estimates by propgating their analytic gradients back
through imagined trajectories. When interacting with the environment, it simply
executes the action model.

Find out more:

- [Project website][website]
- [PDF paper][paper]

[website]: https://danijar.com/dreamer
[paper]: https://arxiv.org/pdf/1912.01603.pdf

## Instructions

To train an agent, install the dependencies and then run one of these commands:

```sh
python3 -m dreamer.scripts.train --logdir ./logdir/debug \
  --params '{defaults: [dreamer, debug], tasks: [dummy]}' \
  --num_runs 1000 --resume_runs False
```

```sh
python3 -m dreamer.scripts.train --logdir ./logdir/control \
  --params '{defaults: [dreamer], tasks: [walker_run]}'
```

```sh
python3 -m dreamer.scripts.train --logdir ./logdir/atari \
  --params '{defaults: [dreamer, pcont, discrete, atari], tasks: [atari_boxing]}'
```

```sh
python3 -m dreamer.scripts.train --logdir ./logdir/dmlab \
  --params '{defaults: [dreamer, discrete], tasks: [dmlab_collect]}'
```

The available tasks are listed in `scripts/tasks.py`. The hyper parameters can
be found in `scripts/configs.py`.

Tips:

- Add `debug` to the list of defaults to use a smaller config and reach
  the code you're developing more quickly.
- Add the flags `--resume_runs False` and `--num_runs 1000`
  to automatically create unique logdirs.
- To train the baseline without value function, add `value_head: False` to the
  params.
- To train PlaNet, add `train_planner: cem, test_planner: cem,
  planner_objective: reward, action_head: False, value_head: False,
  imagination_horizon: 0` to the params.

## Dependencies

The code was tested under Ubuntu 18 and uses these packages:
tensorflow-gpu==1.13.1, tensorflow_probability==0.6.0, dm_control (`egl`
[rendering option][rendering] recommended), gym, imageio, matplotlib,
ruamel.yaml, scikit-image, scipy.

[rendering]: https://github.com/deepmind/dm_control#rendering

Disclaimer: This is not an official Google product.
