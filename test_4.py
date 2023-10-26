import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

env = gym.make("LunarLander-v2")
env.reset()

model = PPO.load('trained_model.zip', env=env)  # Load your trained model here

episodes = 10

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        actions, states = model.predict(obs, deterministic = True)
        actions = np.clip(actions, env.action_space.low, env.action_space.high) # reason - model.predict() returns actions in range [-1,1] from model trained with range [0,1]
        obs, rewards, done, info = env.step(actions)

env.close()