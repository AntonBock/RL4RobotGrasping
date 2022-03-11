import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3

training = True
checkpoint = False
network_type = "sac"
# Define the models (stochastic and deterministic models) for the agent using helper classes 
# and programming with two approaches (layer by layer and torch.nn.Sequential class).
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class StochasticActor(GaussianModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        super().__init__(observation_space, action_space, device, clip_actions,
                         clip_log_std, min_log_std, max_log_std)

        self.linear_layer_1 = nn.Linear(self.num_observations, 512)
        self.linear_layer_2 = nn.Linear(512, 256)
        self.linear_layer_3 = nn.Linear(256, 128)
        self.linear_layer_4 = nn.Linear(128, 64)
        self.mean_action_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, states, taken_actions):
        x = F.elu(self.linear_layer_1(states))
        x = F.elu(self.linear_layer_2(x))
        x = F.elu(self.linear_layer_3(x))
        x = F.elu(self.linear_layer_4(x))
        return torch.tanh(self.mean_action_layer(x)), self.log_std_parameter

class DeterministicActor(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions = False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 512)
        self.linear_layer_2 = nn.Linear(512, 256)
        self.linear_layer_3 = nn.Linear(256, 128)
        self.linear_layer_4 = nn.Linear(128, 64)
        self.mean_action_layer = nn.Linear(64, self.num_actions)

    def compute(self, states, taken_actions):
        x = F.elu(self.linear_layer_1(states))
        x = F.elu(self.linear_layer_2(x))
        x = F.elu(self.linear_layer_3(x))
        x = F.elu(self.linear_layer_4(x))
        return torch.tanh(self.mean_action_layer(x))

class Critic(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 1))

    def compute(self, states, taken_actions):
        return self.net(torch.cat([states, taken_actions], dim=1))


# Load and wrap the Isaac Gym environment.
env = load_isaacgym_env_preview3(task_name="FrankaCabinet", isaacgymenvs_path="./env")
env = wrap_env(env)

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=8000, num_envs=env.num_envs, device=device)

# Instantiate the agent's models (function approximators).

if network_type == "sac":
    networks_current = {"policy": StochasticActor(env.observation_space, env.action_space, device, clip_actions=True),
                "critic_1": Critic(env.observation_space, env.action_space, device) if training else None,
                "critic_2": Critic(env.observation_space, env.action_space, device) if training else None,
                "target_critic_1": Critic(env.observation_space, env.action_space, device) if training else None,
                "target_critic_2": Critic(env.observation_space, env.action_space, device) if training else None}

print(f"NETWORK: {networks_current}")    

if checkpoint:
    networks_current["policy"].load("./runs/22-02-23_09-47-54-394557_SAC/checkpoints/50000_policy.pt") 
else:
    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for network in networks_current.values():
        network.init_parameters(method_name="normal_", mean=0.0, std=0.1) 


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 512
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 0
cfg_sac["learn_entropy"] = True
# logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 25
cfg_sac["experiment"]["checkpoint_interval"] = 1000


agent = SAC(networks=networks_current,
            memory=memory, 
            cfg=cfg_sac, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "headless": True, "progress_interval": 200}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training

if training:
    trainer.train()
else:
    trainer.eval()


