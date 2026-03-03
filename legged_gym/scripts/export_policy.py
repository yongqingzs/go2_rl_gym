import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, task_registry, Logger
from legged_gym.utils.exporter import export_policy_as_jit, export_policy_as_onnx, export_policy_as_pkl

import numpy as np
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 7
    env_cfg.terrain.num_cols = 7
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_pd_gains = False
    env_cfg.domain_rand.randomize_motor_zero_offset = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)
    
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    if hasattr(runner.alg, 'actor_critic'):
        model = runner.alg.actor_critic
    else:
        model = runner.alg.model
    export_policy_as_jit(model, path)
    # export_policy_as_onnx(model, path)
    # export_policy_as_pkl(model, path)
    print('Exported policy as jit script / onnx to: ', path)


if __name__ == '__main__':
    EXPORT_POLICY = True
    args = get_args()
    args.headless = True
    play(args)
