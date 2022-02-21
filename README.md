# RL4RobotGrasping

<<<<<<< Updated upstream
# Demo directory
Find media files [here](https://drive.google.com/drive/folders/1iZSdf_ES50ebloyL4NZgCL3jxnOBp3L2?usp=sharing).


=======
>>>>>>> Stashed changes
# Install

## Prerequisites 
- Ubuntu 18.04 or 20.04.
- Python 3.6, 3.7 or 3.8.
- Minimum recommended NVIDIA driver version:
  + Linux: 470

### NVidia Isaac Gym
Download Isaac gym preview 3 from their website:
https://developer.nvidia.com/isaac-gym

#### Install in a new conda environment

In the root directory, run:

    ./create_conda_env_rlgpu.sh

This will create a new conda env called ``rlgpu``, which you can activate by running:

    conda activate rlgpu

#### Uninstall conda enviroment

To uninstall, run:
<<<<<<< Updated upstream

    conda remove --name rlgpu --all

=======

    conda remove --name rlgpu --all

>>>>>>> Stashed changes
For troubleshooting the Isaac gym install refer to the ``docs`` folder in Isaac Gym

### Install Isaac gym enviroments
Clone/Download the IsaacGymEnv repo(https://github.com/NVIDIA-Omniverse/IsaacGymEnvs):

    git clone https://github.com/NVIDIA-Omniverse/IsaacGymEnvs
    cd IsaacGymEnvs

To install:

    pip install -e .

### SKRL - Reinforcement Learning library
Clone/Download the github repo(https://github.com/Toni-SM/skrl):

    git clone https://github.com/Toni-SM/skrl.git
    cd skrl

To install:

    pip install -e .

### Validate install
To validate the install try to run(make sure the conda env `rlgpu` is active):

    python isaacgym_cartpole_test.py
