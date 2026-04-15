import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import sys
import time

_WINDOWS = sys.platform.startswith("win")

def _torcs_launch(vision=False):
    """Start TORCS.  On Windows the user must do this manually."""
    if _WINDOWS:
        print("\nWINDOWS: Please ensure TORCS is running.")
        print("  Race -> Practice / Quick Race -> New Race, then click Accept.\n")
        return
    os.system("pkill torcs")
    time.sleep(0.5)
    flags = "-nofuel -nodamage -nolaptime"
    if vision:
        flags += " -vision"
    os.system(f"torcs {flags} &")
    time.sleep(0.5)
    os.system("sh autostart.sh")
    time.sleep(0.5)

def _torcs_kill():
    """Kill TORCS.  On Windows this is a no-op (user manages the process)."""
    if _WINDOWS:
        return
    os.system("pkill torcs")


class TorcsEnv:
    terminal_judge_start = 500 
    termination_limit_progress = 5  
    
    default_speed = 320.0 

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False):

        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.initial_run = True

        _torcs_launch(vision=self.vision)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    def step(self, u):
        client = self.client
        this_action = self.agent_to_torcs(u)
        action_torcs = client.R.d
        action_torcs['steer'] = this_action['steer']
        action_torcs['accel'] = this_action.get('accel', 0.0)
        action_torcs['brake'] = this_action.get('brake', 0.0)
        action_torcs['gear']  = this_action.get('gear', client.R.d.get('gear', 1))

        # Save previous obs for reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update
        client.respond_to_server()
        client.get_servers_input()
        obs = client.S.d
        self.observation = self.make_observaton(obs)

        # Reward setting
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp*np.cos(obs['angle'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement
        episode_terminate = False
        if track.min() < 0:
            reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0:
            episode_terminate = True
            client.R.d['meta'] = True

        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], {}

    def reset(self, relaunch=False):
        self.time_step = 0
        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        self.client.MAX_STEPS = np.inf
        client = self.client
        client.get_servers_input()
        obs = client.S.d
        self.observation = self.make_observaton(obs)
        self.last_u = None
        self.initial_reset = False
        return self.get_obs()

    def end(self):
        _torcs_kill()

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
        _torcs_kill()
        time.sleep(0.5)
        _torcs_launch(vision=self.vision)

    def agent_to_torcs(self, u):
        return {
            'steer': float(u[0]),
            'accel': float(u[1]) if len(u) > 1 else 0.0,
            'brake': float(u[2]) if len(u) > 2 else 0.0,
            'gear':  int(u[3])   if len(u) > 3 else 1,
        }

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observaton(self, raw_obs):
        if self.vision is False:
            names = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel']
            Observation = col.namedtuple('Observaion', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel', 'img']
            Observation = col.namedtuple('Observaion', names)
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)