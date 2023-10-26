import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from huggingface_sb3 import package_to_hub
from huggingface_hub import notebook_login

# Utworzenie srodowiska
env = gym.make('LunarLander-v2')
env = Monitor(env, filename=None, allow_early_resets=True)
env_id = "LunarLander-v2"


# inicjacja agenta PPO - jeden z algorytmów uczenia ze wzmocnieniem
model = PPO(
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


model.learn(total_timesteps=int(1000000)) # uczenie agenta 

# Zapisanie modelu
model_name = "Starship"
model.save(model_name)



# Ocena modelu, przyznanie nagrod
eval_env = Monitor(gym.make("LunarLander-v2", render_mode="rgb_array"))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")




# Przeslanie modelu do huggingface.co w celu wizualizacji jego uczenia
model_architecture = "PPO"
repo_id = "theteroles/asci"
commit_message = "Upload PPO LunarLander-v2 trained agent"




package_to_hub(model=model, 
               model_name=model_name, 
               model_architecture=model_architecture, 
               env_id=env_id, 
               eval_env=eval_env, 
               repo_id=repo_id, 
               commit_message=commit_message)



#huggingface skrypt
#huggingface-cli login
#token - https://huggingface.co/settings/tokens
#https://huggingface.co/theteroles/asci