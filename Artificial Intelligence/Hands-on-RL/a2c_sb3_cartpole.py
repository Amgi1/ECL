from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from huggingface_sb3 import push_to_hub
import wandb
from wandb.integration.sb3 import WandbCallback


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}
run = wandb.init(
    project="CartPole-A2C",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)
# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=1)
vec_env = VecVideoRecorder(
            vec_env,
            f"videos/{run.id}",
            record_video_trigger=lambda x: x % 2000 == 0,
            video_length=200,
        )

model = A2C("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=25000, 
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,)
        )
model.save("a2c_cartpole")

run.finish()

push_to_hub(
    repo_id="Amnyfr/hands-on-rl",
    filename="a2c_cartpole.zip",
    commit_message="Commit model A2C BE1",
)
