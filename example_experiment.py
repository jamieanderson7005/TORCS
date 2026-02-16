from gym_torcs import TorcsEnv
from sample_agent import Agent
import numpy as np

# 1. Disable vision initially (it is very heavy and can cause lag/crashes)
vision = False 

# 2. Initialize environment with throttle enabled
env = TorcsEnv(vision=vision, throttle=True)
agent = Agent(1) 

print("--- TORCS EXPERIMENT STARTING ---")

try:
    for episode in range(5):
        print(f"Starting Episode: {episode}")
        
        # Reset and relaunch TORCS on first episode to ensure a fresh connection
        ob = env.reset(relaunch=(episode == 0))

        # 3. Set a massive step limit so the script doesn't close while you are in menus
        for step in range(50000): 
            # Define Action: [Steering, Acceleration]
            # action[0] = 0.0 (Straight) | action[1] = 0.5 (Half Throttle)
            action = np.array([0.0, 0.5]) 
            
            # Send action to environment
            ob, reward, done, _ = env.step(action)
            
            # Print telemetry every 100 steps to confirm connection
            if step % 100 == 0:
                print(f"Step: {step} | Speed: {ob.speedX:.2f} | RPM: {ob.rpm:.0f}")
            
            if done:
                break
                
except KeyboardInterrupt:
    print("\nExperiment stopped by user.")

finally:
    env.end()
    print("Cleanup complete.")