# Jaxman: Jax-based implementation for multi-agent navigation
<!-- |Grid| Diff Drive| Continous|
| ------ | ------ | ------ | -->
<img src=assets/grid.gif width=250>
<img src=assets/diff_drive.gif width=250>
<img src=assets/continuous.gif width=250>

JAXMAN is a JAX-based library for multi-agent navigation. Our library can create environments with three different dynamics.

## Installation
### **venv**
```console
$ python -m venv .**venv**
$ source .venv/bin/activate
(.venv) $ pip install -e .[dev]
```

### Docker container
```console
$ docker-compose build
$ docker-compose up -d dev
$ docker-compose exec dev bash
```

### Docker container with CUDA enabled
```console
$ docker-compose up -d dev-gpu
$ docker-compose exec dev-gpu bash
```

and update JAX modules in the container...

```console
# pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Tutorial
- [1. Environment](tutorial/1.%20Environment.ipynb)
- [2. RL Agent](tutorial/2.RL%20Agent.ipynb)

## Tests
Test code is located in `tests` and can test environment dynamics and RL agent features by `pytest -v`

## Experiment and Evaluation
After the setup, you can run experiment as follow (expected run in docker-container)
```console
# python scripts/train_rl.py # train RL agent in grid environment
# python scripts/train_rl.py env.is_diff_drive=True # train RL agent in diff drive environmnet
# python scripts/train_rl.py env.is_discrete=False # train RL agent in continous environmnet
# python scripts/train_rl.py env.num_agents=10 # train RL agent in grid environment with 10 agent
```