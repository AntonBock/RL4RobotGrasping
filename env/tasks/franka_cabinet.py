# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import random

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


class FrankaCabinet(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        # print("Running __init__")
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.height_reward_scale = self.cfg["env"]["heightRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.randPos = self.cfg["env"]["randomPropPosition"]
        self.randProp = self.cfg["env"]["propSelect"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        

        # prop dimensions
        self.prop_width = 0.05
        self.prop_height = 0.05
        self.prop_length = 0.05
        self.prop_spacing = 0.06

        num_obs = 22
        num_acts = 9

        self.cfg["env"]["numObservations"] = 22
        self.cfg["env"]["numActions"] = 9

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)


        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([0, 0, 0, -1.0, 0, 1.1, 0.0, 0.035, 0.035], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.prop_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.prop_dof_pos = self.prop_dof_state[..., 0] 
        self.prop_dof_vel = self.prop_dof_state[..., 1] 

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 1:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (1 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        # print("Finished __init__")
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # print("Finished __init__2")

    def create_sim(self):
        print("Creating sim")
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        print("Finished creating sim")


    def _create_ground_plane(self):
        print("Creating ground plane")
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        print("Finished creating ground plane")
    


    def _create_envs(self, num_envs, spacing, num_per_row):
        print("Creating envs")
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda.urdf"
        box_asset_file = "urdf/cube/cube.urdf"
        cyl_asset_file = "urdf/cylinder/cylinder.urdf"
        sphere_asset_file = "urdf/sphere/sphere.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            box_asset_file = self.cfg["env"]["asset"].get("assetFileNameBox", box_asset_file)
            cyl_asset_file = self.cfg["env"]["asset"].get("assetFileNameCyl", cyl_asset_file)
            sphere_asset_file = self.cfg["env"]["asset"].get("assetFileNameSphere", sphere_asset_file)
            

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)


        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)


        print("num franka bodies: ", self.num_franka_bodies)
        print("num franka dofs: ", self.num_franka_dofs)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = franka_dof_damping[i]
            else:
                franka_dof_props['stiffness'][i] = 7000.0
                franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
            self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[[7, 8]] = 0.1
        franka_dof_props['effort'][7] = 200
        franka_dof_props['effort'][8] = 200

        # create prop assets
        # box_opts = gymapi.AssetOptions()
        # box_opts.density = 400
        # prop_asset = self.gym.create_box(self.sim, self.prop_width, self.prop_height, self.prop_width, box_opts)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)


        box_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file)
        cyl_asset = self.gym.load_asset(self.sim, asset_root, cyl_asset_file)
        sphere_asset = self.gym.load_asset(self.sim, asset_root, sphere_asset_file)








        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_prop_bodies = 1 #self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = 1 #self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + self.num_props * num_prop_bodies 
        max_agg_shapes = num_franka_shapes + self.num_props * num_prop_shapes 

        self.frankas = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        
        print("Iterating through environments")
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)




            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.num_props > 0:
                self.prop_start.append(self.gym.get_sim_actor_count(self.sim))



                props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                prop_count = 0
                for j in range(props_per_row):
                    prop_up = yzmin + j * self.prop_spacing
                    for k in range(props_per_row):
                        if prop_count >= self.num_props:
                            break
                        # propx = xmin + k * self.prop_spacing

                        random.seed()

                        prop_state_pose = gymapi.Transform()

                        roll = 0
                        pitch = 0
                        yaw = 0
                        prop_state_pose.p.x = 0.5 # drawer_pose.p.x + propx
                        # propz, propy = 0, prop_up
                        prop_state_pose.p.y = 0.0 #drawer_pose.p.y + propy
                        prop_state_pose.p.z = 0.026 # drawer_pose.p.z + propz


                        # Use random positioning?

                        if self.randPos:
                            prop_state_pose.p.x = randrange_float(0.40, 0.80, self.prop_spacing)
                            # propz, propy = 0, p
                            prop_state_pose.p.y = randrange_float(-0.50, 0.50, self.prop_spacing)
                            prop_state_pose.p.z = 0.05
                            yaw = random.uniform(0.00000, 6.28318)
            

                        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                        

                        prop_state_pose.r = gymapi.Quat(qx, qy, qz, qw)

                        # choice of prop



                        if self.randProp == "box":
                            prop_actor = self.gym.create_actor(env_ptr, box_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        elif self.randProp == "cyl":
                            prop_actor = self.gym.create_actor(env_ptr, cyl_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        elif self.randProp == "sphere":
                            prop_actor = self.gym.create_actor(env_ptr, sphere_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        elif self.randProp == "rand":
                            x = chooseProp()
                            if x==0: temp_asset=box_asset
                            elif x==1: temp_asset=cyl_asset
                            elif x==2: temp_asset=sphere_asset
                            prop_actor = self.gym.create_actor(env_ptr, temp_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)

                        # xtra_actor = self.gym.create_actor(env_ptr, xtra_asset, prop_state_pose, "xtra{}".format(prop_count), i, 0, 0)
                        # prop_actor = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1
                        

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x , prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
                        
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            # self.cabinets.append(cabinet_actor)
            print("Environment no. ", i+1, " created")
        
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        #self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.prop_handle = self.gym.find_actor_rigid_body_handle(env_ptr, prop_actor, "prop_box")
        # self.gym.set_actor_scale(env_ptr, self.prop_handle, 5.2)
        # self.prop_handle = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
        self.lfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_leftfinger")
        self.rfinger_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_rightfinger")
        self.default_prop_states = to_torch(self.default_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)
        print("Finished creating environments")
        
        self.init_data()

    def init_data(self):
        print("Initialising data")
        # Franka
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_link7")
        lfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_leftfinger")
        rfinger = self.gym.find_actor_rigid_body_handle(self.envs[0], self.frankas[0], "panda_rightfinger")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        self.lfinger_pose = self.gym.get_rigid_transform(self.envs[0], lfinger)
        self.rfinger_pose = self.gym.get_rigid_transform(self.envs[0], rfinger)

        finger_pose = gymapi.Transform()
        finger_pose.p = (self.lfinger_pose.p + self.rfinger_pose.p) * 0.5
        finger_pose.r = self.lfinger_pose.r

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        franka_local_grasp_pose = hand_pose_inv * finger_pose
        franka_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        self.franka_local_grasp_pos = to_torch([franka_local_grasp_pose.p.x, franka_local_grasp_pose.p.y,
                                                franka_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.franka_local_grasp_rot = to_torch([franka_local_grasp_pose.r.x, franka_local_grasp_pose.r.y,
                                                franka_local_grasp_pose.r.z, franka_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))


        # Prop instead of drawer
        prop_local_grasp_pose = gymapi.Transform()
        #prop_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.0, grasp_pose_axis, 0.0))
        #prop_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.prop_local_grasp_pos = to_torch([prop_local_grasp_pose.p.x, prop_local_grasp_pose.p.y,
                                                prop_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.prop_local_grasp_rot = to_torch([prop_local_grasp_pose.r.x, prop_local_grasp_pose.r.y,
                                                prop_local_grasp_pose.r.z, prop_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))
        

        # Unknown
        self.gripper_forward_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        #self.drawer_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.prop_inward_axis = to_torch([-1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.gripper_up_axis = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        #self.drawer_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))
        self.prop_up_axis = to_torch([0, 0, 1], device=self.device).repeat((self.num_envs, 1))

        self.franka_grasp_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_grasp_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_grasp_rot[..., -1] = 1  # xyzw

        # Drawer version
        # self.drawer_grasp_pos = torch.zeros_like(self.drawer_local_grasp_pos)
        # self.drawer_grasp_rot = torch.zeros_like(self.drawer_local_grasp_rot)
        # self.drawer_grasp_rot[..., -1] = 1

        # Prop version
        self.prop_grasp_pos = torch.zeros_like(self.prop_local_grasp_pos)
        self.prop_grasp_rot = torch.zeros_like(self.prop_local_grasp_rot)
        self.prop_grasp_rot[..., -1] = 1


        self.franka_lfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_rfinger_pos = torch.zeros_like(self.franka_local_grasp_pos)
        self.franka_lfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        self.franka_rfinger_rot = torch.zeros_like(self.franka_local_grasp_rot)
        print("Finished initialising data")

    def compute_reward(self, actions):
 
        self.gym.refresh_net_contact_force_tensor(self.sim)
        _force_vec = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_vec = gymtorch.wrap_tensor(_force_vec)
        force_vec = force_vec.view(-1, 11, 3)
        force_vec_left = force_vec.select(1, 8).cpu().numpy()
        force_vec_right = force_vec.select(1, 9).cpu().numpy()
        #print(f"FORCES:{force_vec_left}")
        
        right_forces = []
        left_forces = []
        force_tens = torch.empty(self.num_envs, dtype=torch.int16, device=0)
        for i in range(self.num_envs):

            left_f = gymapi.Vec3()
            left_f.x = force_vec_left[i][0]
            left_f.y = force_vec_left[i][1]
            left_f.z = force_vec_left[i][2]
            right_f = gymapi.Vec3()
            right_f.x = force_vec_right[i][0]
            right_f.y = force_vec_right[i][1]
            right_f.z = force_vec_right[i][2]

            left_force = self.lfinger_pose.transform_vector(left_f)
            right_force = self.rfinger_pose.transform_vector(right_f)
            
            left_forces.append(left_force)
            right_forces.append(right_force)

            if abs(left_force.y)> 1 and abs(right_force.y)> 1:
                force_tens[i] = 1
            else:
                force_tens[i] = 0
                

        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions, #self.cabinet_dof_pos,
            self.franka_grasp_pos, self.prop_grasp_pos, self.franka_grasp_rot, self.prop_grasp_rot,
            self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.gripper_forward_axis, self.prop_inward_axis, self.gripper_up_axis, self.prop_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.height_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length, force_tens
        )
        # print("Finished computing reward")

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        prop_pos = self.rigid_body_states[:, self.prop_handle][:, 0:3]
        prop_rot = self.rigid_body_states[:, self.prop_handle][:, 3:7]

        self.franka_grasp_rot[:], self.franka_grasp_pos[:], self.prop_grasp_rot[:], self.prop_grasp_pos[:] = \
            compute_grasp_transforms(hand_rot, hand_pos, self.franka_local_grasp_rot, self.franka_local_grasp_pos,
                                     prop_rot, prop_pos, self.prop_local_grasp_rot, self.prop_local_grasp_pos
                                     )

        self.franka_lfinger_pos = self.rigid_body_states[:, self.lfinger_handle][:, 0:3]
        self.franka_rfinger_pos = self.rigid_body_states[:, self.rfinger_handle][:, 0:3]
        self.franka_lfinger_rot = self.rigid_body_states[:, self.lfinger_handle][:, 3:7]
        self.franka_rfinger_rot = self.rigid_body_states[:, self.rfinger_handle][:, 3:7]

        dof_pos_scaled = (2.0 * (self.franka_dof_pos - self.franka_dof_lower_limits)
                          / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) - 1.0)

        testArray = np.zeros([16,4])
        to_target = self.prop_grasp_pos - self.franka_grasp_pos # Distance to target
        self.obs_buf = torch.cat((dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, to_target, self.prop_grasp_pos[:, 2].unsqueeze(-1)), dim=-1) #self.prop_dof_vel[:, 3].unsqueeze(-1)),

        # print("Box position: ", prop_pos)
        # print("Finished computing observations")
        return self.obs_buf

    def reset_idx(self, env_ids):
        # print("Running reset_idx")
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        
        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos
        
        # reset cabinet
        #self.cabinet_dof_state[env_ids, :] = torch.zeros_like(self.cabinet_dof_state[env_ids])

        # reset props (Random)
        if self.randPos:

            self.rand_prop_states = []
            for i in range(self.num_envs):
                if self.num_props > 0:


                    props_per_row = int(np.ceil(np.sqrt(self.num_props)))
                    xmin = -0.5 * self.prop_spacing * (props_per_row - 1)
                    yzmin = -0.5 * self.prop_spacing * (props_per_row - 1)

                    prop_count = 0
                    for j in range(props_per_row):
                        prop_up = yzmin + j * self.prop_spacing
                        for k in range(props_per_row):
                            if prop_count >= self.num_props:
                                break
                            # propx = xmin + k * self.prop_spacing

                            random.seed()

                            prop_state_pose = gymapi.Transform()


                            prop_state_pose.p.x = randrange_float(0.40, 0.80, self.prop_spacing) # returns 2.4
                            # propz, propy = 0, p
                            prop_state_pose.p.y = randrange_float(-0.50, 0.50, self.prop_spacing)
                            prop_state_pose.p.z = 0.05
                            
                            roll = 0
                            pitch = 0
                            yaw = random.uniform(0.00000, 6.28318)

                            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                            

                            prop_state_pose.r = gymapi.Quat(qx, qy, qz, qw)
                            prop_count += 1

                            prop_idx = j * props_per_row + k
                            self.rand_prop_states.append([prop_state_pose.p.x , prop_state_pose.p.y, prop_state_pose.p.z,
                                                            prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                            0, 0, 0, 0, 0, 0])

                            # if self.randProp == "rand":
                            #     x = chooseProp()
                            #     if x==0: temp_asset=box_asset
                            #     elif x==1: temp_asset=cyl_asset
                            #     elif x==2: temp_asset=sphere_asset
                            #     prop_actor = self.gym.create_actor(env_ptr, temp_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
        
            self.default_prop_states = to_torch(self.rand_prop_states, device=self.device, dtype=torch.float).view(self.num_envs, self.num_props, 13)


        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 1:].flatten()
            self.prop_states[env_ids] = self.default_prop_states[env_ids]
            # print("Prop pos: ", self.default_prop_states[env_ids])

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(prop_indices), len(prop_indices))
        
        multi_env_ids_int32 = self.global_indices[env_ids, :1].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.franka_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        # print("Finished reset_idx")

    def pre_physics_step(self, actions):
        # print("Pre_physx")
        # print("Prop_grasp_pos: ", self.prop_grasp_pos)
        self.actions = actions.clone().to(self.device)
        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))
        # print("Prop grasp_pos: ", self.prop_grasp_pos[0])
        # print("Finished Pre_physx")

    def post_physics_step(self):
        # print("Post physx")
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                px = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_grasp_pos[i] + quat_apply(self.franka_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_grasp_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # # Edit?
                # px = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.drawer_grasp_pos[i] + quat_apply(self.drawer_grasp_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.drawer_grasp_pos[i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_lfinger_pos[i] + quat_apply(self.franka_lfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_lfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])

                px = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.franka_rfinger_pos[i] + quat_apply(self.franka_rfinger_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.franka_rfinger_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [1, 0, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0, 1, 0])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0, 0, 1])
    # print("Finished post physx")

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

def chooseProp():
    return np.random.randint(0,3)


#####################################################################
###=========================jit functions=========================###
#####################################################################
rewarded = False
@torch.jit.script
def compute_franka_reward(
    reset_buf, progress_buf, actions,
    franka_grasp_pos, prop_grasp_pos, franka_grasp_rot, prop_grasp_rot,
    franka_lfinger_pos, franka_rfinger_pos,
    gripper_forward_axis, prop_inward_axis, gripper_up_axis, prop_up_axis,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, height_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length, grip_forces
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor) -> Tuple[Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(franka_grasp_pos - prop_grasp_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward) * dist_reward_scale

    lfinger_dist = torch.norm(franka_lfinger_pos - prop_grasp_pos, p=2, dim=-1)
    rfinger_dist = torch.norm(franka_rfinger_pos - prop_grasp_pos, p=2, dim=-1)

    lfinger_reward = 1.0 / (1.0 + lfinger_dist ** 2)
    lfinger_reward *= lfinger_reward
    lfinger_reward = torch.where(d <= 0.05, lfinger_reward * 2, lfinger_reward) * finger_dist_reward_scale

    rfinger_reward = 1.0 / (1.0 + rfinger_dist ** 2)
    rfinger_reward *= rfinger_reward
    rfinger_reward = torch.where(d <= 0.05, rfinger_reward * 2, rfinger_reward) * finger_dist_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale

    # How high the box has been lifted
    height_reward = torch.where(prop_grasp_pos[:, 2]>10, 5, 0) * height_reward_scale  # drawer_top_joint
    
    finger_dist = torch.norm(franka_lfinger_pos - franka_rfinger_pos, p=2, dim=-1)
   

    close_reward = torch.where(d <= 0.3, 0.05, 0.0)

    dist_reward = torch.where(d <= 0.06, 10, 0)
    height_reward = torch.where(prop_grasp_pos[:, 2]>4, 500, 0)
        
    
    time_penalty = 0.1

    grip_reward = grip_forces * 100

    rewards = dist_reward + close_reward + grip_reward - time_penalty

    #reset_buf = torch.ones_like(reset_buf) if height_reward > 100 else reset_buf
    #reset_buf = torch.where(d <= 0.06, torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf



@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos,
                             prop_rot, prop_pos, prop_local_grasp_rot, prop_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_franka_rot, global_franka_pos = tf_combine(
        hand_rot, hand_pos, franka_local_grasp_rot, franka_local_grasp_pos)
    global_prop_rot, global_prop_pos = tf_combine(
        prop_rot, prop_pos, prop_local_grasp_rot, prop_local_grasp_pos)

    return global_franka_rot, global_franka_pos, global_prop_rot, global_prop_pos #global_drawer_rot, global_drawer_pos
