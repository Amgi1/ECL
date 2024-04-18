# TD 1 : Hands-On Reinforcement Learning

## Description
This project contains 3 python scripts that serve as an initiation to Reinforcement Learning. The relevant python libraries used for reinforcement learning in this project are gym, stable-baselines 3 and panda-gym.

The models and relevant training data were saved using Hugging Face (https://huggingface.co/Amnyfr/hands-on-rl/tree/main) and Weights and biases (https://wandb.ai/amnyfr_team/). An issue with Weights and Biases (potentially related to https://github.com/wandb/wandb/issues/1370) stopped all training data from being saved. A solution I found was to run the python scripts with administrator privileges.

All training was done on CPU.


## Installation

We recommend to use Python virtual environnements to install the required modules : https://docs.python.org/3/library/venv.html

First, install Pytorch : https://pytorch.org/get-started/locally.

Then install the following modules :


```sh
pip install gym==0.26.2
```

Install also pyglet for the rendering.

```sh
pip install pyglet==2.0.10
```

If needed 

```sh
pip install pygame==2.5.2
```

```sh
pip install PyQt5
```

Install stable baselines.
```sh
pip install stable-baselines3
pip install moviepy
```

Install Hugging Face to upload models.

```sh
pip install huggingface-sb3==2.3.1
```

Install tensorboard and Weights and Biases to monitor training.

```sh
pip install wandb tensorboard
```

## Models

### Reinforce

Python file reinforce_cartpole.py uses the policy gradient method to solve the CartPole environment.

This is done using the gym environment CartPole-v1. The policy is learned by a fully-connected neural network over 500 episodes.

![img](<Plot reward cartpole.png>)

Plot of cartpole reward evolution during training

### A2C SB3 Cartpole

As a follow-up, we solve the CartPole environment using the Advantage Actor Critic method in python file a2c_sb3_cartpole.py.

Training is done using Stable Baseline 3 tool A2C over 25000 timesteps.

The trained model is saved at https://huggingface.co/Amnyfr/hands-on-rl/blob/main/a2c_cartpole.zip and the training data at https://wandb.ai/amnyfr_team/CartPole-A2C.

### A2C SB3 PandaReach

We now solve in file a2c_sb3_panda_reach.py the PandaReach environment from the panda_gym library using the aforementionned Advantage Actor Critic method.

The number of episodes was set at 500000.

The trained model is saved at https://huggingface.co/Amnyfr/hands-on-rl/blob/main/a2c_panda_reach.zip and the training data at https://wandb.ai/amnyfr_team/PandaReach-A2C.


## Authors and acknowledgment
Author: Amaury Giard

Based on project MSO_3_4-TD1 by Emmanuel Dellandrea.

Work in class done under supervision by Léo Schneider.
