from gym_torcs import TorcsEnv
from ddpg_agent import Actor, Critic, ReplayBuffer
import torch
import numpy as np

# 1. Setup Environment
# Set throttle=True so the AI learns to accelerate, not just steer
env = TorcsEnv(vision=False, throttle=True, gear_change=False)
state_dim = 29 # Sensors
action_dim = 3 # Steer, Accel, Brake

actor = Actor(state_dim, action_dim)
replay_buffer = ReplayBuffer()
optimizer = torch.optim.Adam(actor.parameters(), lr=0.001)

# 2. The Training Loop
for episode in range(500):
    ob = env.reset(relaunch=(episode % 3 == 0))
    # Flatten the observation components into one vector
    state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    for step in range(500):
        # AI decides what to do
        state_tensor = torch.FloatTensor(state)
        action = actor(state_tensor).detach().numpy()
        
        # Add exploration noise (so the car tries new things)
        action += np.random.normal(0, 0.1, size=action_dim)

        # Execute in TORCS
        ob_next, reward, done, _ = env.step(action)
        
        # Prepare next state
        next_state = np.hstack((ob_next.angle, ob_next.track, ob_next.trackPos, ob_next.speedX, ob_next.speedY, ob_next.speedZ, ob_next.wheelSpinVel/100.0, ob_next.rpm))
        
        # Save to memory
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer.buffer) > 64:
            # HERE: This is where you would call your optimizer.step() 
            # to update the neural network weights.
            pass

        if done: break