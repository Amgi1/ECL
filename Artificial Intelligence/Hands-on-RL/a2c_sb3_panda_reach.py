import gymnasium as gym
from stable_baselines3 import A2C
import panda_gym
from huggingface_sb3 import push_to_hub
import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MultiInputPolicy",
    "total_timesteps": 500000,
    "env_name": "PandaReachJointsDense-v3",
}
run = wandb.init(
    project="PandaReach-A2C",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)


# Create the environment
env = gym.make(config["env_name"])#, render_mode="human")

# Create the A2C model
model = A2C(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/panda_reach_{run.id}")

# Train the model
model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(gradient_save_freq=100, 
            model_save_path=f"models/panda_reach_{run.id}", verbose=2,)
            )

run.finish()

model.save("a2c_panda_reach")

push_to_hub(
    repo_id="Amnyfr/hands-on-rl",
    filename="a2c_panda_reach.zip",
    commit_message="Commit model A2C PandaReach",)
