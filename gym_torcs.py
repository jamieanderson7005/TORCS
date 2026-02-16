import gym
import numpy as np
import snakeoil3_gym as snakeoil3
import collections as col
import os
import time
import subprocess 

class TorcsEnv:
    def __init__(self, vision=False, throttle=False):
        self.vision = vision
        self.throttle = throttle
        # Ensure this path is correct for your system!
        self.torcs_path = r"C:\Users\User\Downloads\torcs\torcs\wtorcs.exe" 

    def launch_torcs(self):
        self.kill_torcs()
        print(f"Launching TORCS...")
        torcs_dir = os.path.dirname(self.torcs_path)
        try:
            # Launch without shell=True for better process control
            subprocess.Popen([self.torcs_path, "-nofuel", "-nodamage", "-nolaptime"], 
                             cwd=torcs_dir)
        except Exception as e:
            print(f"Error launching TORCS: {e}")
        
        time.sleep(2.0)
        print(">>> ACTION: In TORCS, select RACE -> QUICK RACE -> NEW RACE <<<")

    def kill_torcs(self):
        os.system('taskkill /F /IM wtorcs.exe /T > NUL 2>&1')
        time.sleep(1.0)

    def step(self, u):
        client = self.client
        client.R.d['steer'] = u[0]
        client.R.d['accel'] = u[1] if self.throttle else 0.2

        # Simple Automatic Gearbox
        speed = client.S.d.get('speedX', 0)
        if speed < 5: client.R.d['gear'] = 1
        elif speed < 50: client.R.d['gear'] = 2
        else: client.R.d['gear'] = 3

        client.respond_to_server()
        client.get_servers_input()

        raw_obs = client.S.d
        names = ['speedX', 'speedY', 'speedZ', 'rpm', 'track', 'wheelSpinVel']
        Observation = col.namedtuple('Observation', names)
        
        # Check if race is over
        done = client.S.d.get('meta', 0) == 1
        
        obs = Observation(*[raw_obs.get(name, 0) for name in names])
        return obs, raw_obs.get('speedX', 0), done, {}

    def reset(self, relaunch=False):
        if relaunch:
            self.launch_torcs()
        
        self.client = snakeoil3.Client(p=3001, vision=self.vision)
        return self.step([0,0])[0]

    def end(self):
        self.kill_torcs()