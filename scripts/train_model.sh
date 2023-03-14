#!/bin/bash

# test
CUDA_VISIBLE_DEVICES=0 /venv/bin/python scripts/train_basemodel.py env.map_size=10 env.num_items=7 env.num_agents=1 env.is_respawn=True
CUDA_VISIBLE_DEVICES=0 /venv/bin/python scripts/train_basemodel.py env.map_size=10 env.num_items=7 env.num_agents=1 env.is_respawn=True model.use_maxmin_dqn=True train.is_pal=False train.use_k_step_learning=False train.use_ddqn=False