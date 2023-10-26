import gymnasium as gym
from stable_baselines3 import PPO
# from stable_baselines3.common.envs import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# stwórz środowisko
env = gym.make("LunarLander-v2")    #zmiana środowiska na "CarRacing-v2", "LunarLander-v2", 'CartPole-v1'
env = Monitor(env, filename=None, allow_early_resets=True)  # To monitor and log training progress

# inicjacja agenta PPO - jeden z algorytmów uczenia ze wzmocnieniem
model = PPO(      #zmiana PPO na A2C realizuje inny algorytm
    'MlpPolicy',
    env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.0,
    verbose=1
)

model.learn(total_timesteps=int(100000)) # uczenie agenta 

model_name = 'trained_model2.zip'  # zapis wartości wynikowych z procesu uczenia
model.save(model_name)

eval_env = Monitor(gym.make("LunarLander-v2"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")