from go1_gym.envs.go1.obstacle_avoidance import ObstacleAvoidance
import isaacgym

assert isaacgym

import os

import torch
import numpy as np

import glob
import os
import pickle as pkl

from go1_gym.envs import *
from go1_gym.envs.base.world_config import Cfg
from go1_gym.envs.go1.go1_config import config_go1
from go1_gym.envs.go1.velocity_tracking import VelocityTrackingEasyEnv

from tqdm import tqdm

should_random_init = False
has_obstacles = True

if should_random_init:
    load_label = "/common/home/st1122/Projects/walk-these-ways/runs/gait-conditioned-agility/2023-11-18/train/030705.545678"
elif has_obstacles:
    load_label = '/common/home/st1122/Projects/walk-these-ways/runs/gait-conditioned-agility/2023-12-07/train/210931.056253'
else:
    load_label = "/common/home/st1122/Projects/walk-these-ways/runs/gait-conditioned-agility/static_init_velocity"



def load_torque_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_latest.jit')
    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit')

    def policy(obs_hist, info={}):
        """
        Converts the observation into a latent vector
        and then passes it to the body to get the action
        """
        latent = adaptation_module.forward(obs_hist.to('cpu'))
        action = body.forward(torch.cat((obs_hist.to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    return policy

def load_velocity_policy(logdir):
    body = torch.jit.load(logdir + '/checkpoints/body_best.jit')

    def policy(obs, info={}):
        """
        Converts the observation into a latent vector
        and then passes it to the body to get the action
        """

        action = body.forward(obs["obs_history_vel"].to('cpu', dtype=torch.float))

    
        return action.cpu().detach()

    return policy


def load_env(label, headless, random_init):
    try:
        contents = os.listdir(label)
    except:
        contents = []
    if 'parameters.pkl' not in contents:
        dirs = glob.glob(os.path.join(os.path.dirname(__file__), f"../runs/{label}/*"))
        logdir = sorted(dirs)[0]
    else:
        logdir = label

    policy = load_velocity_policy(logdir)
    torque_policy = load_torque_policy('/common/home/st1122/Projects/walk-these-ways/runs/gait-conditioned-agility/pretrain-v0/train/025417.456545')

    with open(os.path.join(logdir, "parameters.pkl"), 'rb') as file:
        pkl_cfg = pkl.load(file)
        print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        print(cfg.keys())

        for key, value in cfg.items():
            if hasattr(Cfg, key):
                for key2, value2 in cfg[key].items():
                    setattr(getattr(Cfg, key), key2, value2)

    # turn off DR for evaluation script
    Cfg.domain_rand.push_robots = False
    Cfg.domain_rand.randomize_friction = False
    Cfg.domain_rand.randomize_gravity = False
    Cfg.domain_rand.randomize_restitution = False
    Cfg.domain_rand.randomize_motor_offset = False
    Cfg.domain_rand.randomize_motor_strength = False
    Cfg.domain_rand.randomize_friction_indep = False
    Cfg.domain_rand.randomize_ground_friction = False
    Cfg.domain_rand.randomize_base_mass = False
    Cfg.domain_rand.randomize_Kd_factor = False
    Cfg.domain_rand.randomize_Kp_factor = False
    Cfg.domain_rand.randomize_joint_friction = False
    Cfg.domain_rand.randomize_com_displacement = False

    Cfg.env.num_recording_envs = 1
    Cfg.env.num_envs = 1
    Cfg.env.record_video = 1
    Cfg.terrain.num_rows = 5
    Cfg.terrain.num_cols = 5
    Cfg.terrain.border_size = 0
    Cfg.terrain.center_robots = True
    Cfg.terrain.center_span = 1
    Cfg.terrain.teleport_robots = True

    Cfg.domain_rand.lag_timesteps = 6
    Cfg.domain_rand.randomize_lag_timesteps = True
    Cfg.control.control_type = "actuator_net"

    Cfg.init_state.pos = [0, -1.5, .5]  # x,y,z [m]
    Cfg.init_state.rot = [0.0, 0.0, 0.7, 0.7]

    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper

    env = ObstacleAvoidance(
        sim_device='cuda:0', 
        headless=headless, 
        cfg=Cfg, 
        torque_policy=torque_policy, 
        random_init=random_init,
        has_obstacles = has_obstacles
    )
    env = HistoryWrapper(env)

    return env, policy


def play_go1(headless=True):
    from ml_logger import logger

    from pathlib import Path
    from go1_gym import MINI_GYM_ROOT_DIR
    import glob
    import os

    env, vel_policy = load_env(load_label, headless, random_init=should_random_init)

    num_eval_steps = 750

    env.start_recording()
    env.record_now = True
    env.complete_video_frames = []

    obs = env.reset()

    for i in tqdm(range(num_eval_steps)):
        vel_actions = vel_policy(obs)
        torque_actions = env.convert_vel_to_action(vel_actions, obs['obs_history'])

        obs, rew, done, info = env.step(torque_actions)

        if done:
            print('Success!')
            break
    
    video_frames = env.complete_video_frames
    if len(video_frames) == 0:
        video_frames = env.video_frames
    if len(video_frames) == 0:
        video_frames = env.env.video_frames

    logger.save_video(video_frames, "videos/plan.mp4", fps=1 / env.dt)


if __name__ == '__main__':
    # to see the environment rendering, set headless=False
    play_go1(headless=False)
