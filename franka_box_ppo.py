import isaacgym

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import GaussianModel, DeterministicModel
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2, load_isaacgym_env_preview3

tt = True
checkpoint = False

# Define the models (stochastic and deterministic models) for the agent using helper classes 
# and programming with two approaches (layer by layer and torch.nn.Sequential class).
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
class Policy(GaussianModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        super().__init__(observation_space, action_space, device, clip_actions,
                         clip_log_std, min_log_std, max_log_std)

        self.linear_layer_1 = nn.Linear(self.num_observations, 512) #32
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

class Value(DeterministicModel):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        super().__init__(observation_space, action_space, device, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 1))

    def compute(self, states, taken_actions):
        return self.net(states)


# Load and wrap the Isaac Gym environment.
    
env = load_isaacgym_env_preview3(task_name="FrankaCabinet", isaacgymenvs_path="./env")
env = wrap_env(env)

device = env.device

# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

# Instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#models-networks

networks_ppo = {"policy": Policy(env.observation_space, env.action_space, device, clip_actions=True),
            "value": (Value(env.observation_space, env.action_space, device) if tt else None)}

if checkpoint:
    networks_ppo["policy"].load("./runs/policy.pt") 
else:
    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for network in networks_ppo.values():
        network.init_parameters(method_name="normal_", mean=0.0, std=0.1) 

         

# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.ppo.html#configuration-and-hyperparameters
cfg_ppo = PPO_DEFAULT_CONFIG.copy()
cfg_ppo["learning_starts"] = 0
cfg_ppo["random_timesteps"] = 0
cfg_ppo["rollouts"] = 16
cfg_ppo["learning_epochs"] = 8
cfg_ppo["mini_batches"] = 2

cfg_ppo["discount_factor"] = 0.99
cfg_ppo["lambda"] = 0.99
cfg_ppo["policy_learning_rate"] = 0.0005
cfg_ppo["value_learning_rate"] = 0.0005

cfg_ppo["grad_norm_clip"] = 0.5
cfg_ppo["ratio_clip"] = 0.2
cfg_ppo["value_clip"] = 0.2
cfg_ppo["clip_predicted_values"] = False

cfg_ppo["entropy_loss_scale"] = 0.0
cfg_ppo["value_loss_scale"] = 2.0

cfg_ppo["kl_threshold"] = 0
# logging to TensorBoard and write checkpoints each 16 and 1000 timesteps respectively
cfg_ppo["experiment"]["write_interval"] = 50
cfg_ppo["experiment"]["checkpoint_interval"] = 1000

agent = PPO(models=networks_ppo,
            memory=(memory if tt else None), 
            cfg=cfg_ppo, 
            observation_space=env.observation_space, 
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000000, "progress_interval": 500}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training

if tt:
    trainer.train()
else:
    trainer.eval()


