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
import imageio

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask

from numpy.random import choice
from numpy.random.mtrand import triangular
from scipy import interpolate
from isaacgym.terrain_utils import *
from math import sqrt



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
        self.randTerrain = self.cfg["env"]["randTerrain"]
        self.randFrankaPos = self.cfg["env"]["randStartPos"]
        self.randDynamics = self.cfg["env"]["randDynamics"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.using_camera = self.cfg["env"]["enableCameraSensors"]
        self.img_save = self.cfg["env"]["saveImages"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        # prop dimensions
        self.prop_width = 0.05
        self.prop_height = 0.05
        self.prop_length = 0.05
        self.prop_spacing = 0.06

        self.cam_width = 84
        self.cam_height = 84
        self.cam_pixels = self.cam_width*self.cam_height
        self.non_cam_observations = 19

        self.cfg["env"]["numObservations"] = self.cam_pixels + self.non_cam_observations if self.using_camera else 19
        self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reward state
        self.reward_state = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.success_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.fail_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.success_timer = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self.global_timer = 0

        # create some wrapper tensors for different slices

        self.franka_default_dof_pos = to_torch([0, 0, 0, -1.0, 0, 1.1, 0.0, 0.035, 0.035], device=self.device) 
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.franka_dof_pos = self.franka_dof_state[..., 0]
        self.franka_dof_vel = self.franka_dof_state[..., 1]
        self.prop_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.prop_dof_pos = self.prop_dof_state[..., 0] 
        self.prop_dof_vel = self.prop_dof_state[..., 1]
        self.wall_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_franka_dofs:]
        self.wall_dof_pos = self.wall_dof_state[..., 0]
        self.wall_dof_vel = self.wall_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        if self.num_props > 0:
            self.prop_states = self.root_state_tensor[:, 1:]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.franka_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        # print("Finished __init__")
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # print("Finished __init__2")

    def create_sim(self):
        print("Creating sim")
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8164 #Gravity in Suldrup, DK. Source: https://www.space.dtu.dk/english/-/media/Institutter/Space/English/reports/technical_reports/tech_no_6.ashx, page 7
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
            
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        print("Finished creating sim")


    def _create_ground_plane(self):
        if self.randTerrain==False:
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
        wall_asset_file = "urdf/wall/wall.urdf"
        # rock_asset_file = "urdf/rock2/rock2.urdf"
        # sphere_asset_file = "urdf/donut_1/donut.urdf"

        self.rockList = []
        d = {}

      
        for i in range(len(os.listdir("assets/rocks/urdf"))):            
            rock_asset_file = os.path.join("rocks/urdf/",f"{i}.urdf")         
            rock_asset_file = self.cfg["env"]["asset"].get("assetFileNameRock", rock_asset_file)  

            rock_asset = self.gym.load_asset(self.sim, asset_root, rock_asset_file)
            self.rockList.append(rock_asset)
  


        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.convex_decomposition_from_submeshes=False
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True


        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)
            box_asset_file = self.cfg["env"]["asset"].get("assetFileNameBox", box_asset_file)
            cyl_asset_file = self.cfg["env"]["asset"].get("assetFileNameCyl", cyl_asset_file)
            sphere_asset_file = self.cfg["env"]["asset"].get("assetFileNameSphere", sphere_asset_file)
            wall_asset_file = self.cfg["env"]["asset"].get("assetFileNameWall", wall_asset_file)





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


        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)




        wall_asset_options = gymapi.AssetOptions()
        wall_asset_options.density = 10.0
        wall_asset_options.fix_base_link = True

        box_asset = self.gym.load_asset(self.sim, asset_root, box_asset_file)
        cyl_asset = self.gym.load_asset(self.sim, asset_root, cyl_asset_file)
        sphere_asset = self.gym.load_asset(self.sim, asset_root, sphere_asset_file)
        wall_asset = self.gym.load_asset(self.sim, asset_root, wall_asset_file, asset_options)

        self.num_wall_dofs = self.gym.get_asset_dof_count(wall_asset)

        wall_dof_props = self.gym.get_asset_dof_properties(wall_asset)
        for i in range(self.num_wall_dofs):
            wall_dof_props['damping'][i] = 10.0

        # compute aggregate size
        num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        num_franka_shapes = self.gym.get_asset_rigid_shape_count(franka_asset)
        num_prop_bodies = 1 #self.gym.get_asset_rigid_body_count(prop_asset)
        num_prop_shapes = 1 #self.gym.get_asset_rigid_shape_count(prop_asset)
        max_agg_bodies = num_franka_bodies + 1 + self.num_props * num_prop_bodies #+1 for wall
        max_agg_shapes = num_franka_shapes + 1 + self.num_props * num_prop_shapes 

        self.frankas = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        self.cams = []
        self.cam_tensors = []
        
        print("Iterating through environments")
        tx = -spacing
        ty= -spacing*3

        xount = 0
        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0)


            if self.randDynamics:
                randDyn = randrange_float(0.99, 1.01, 0.00001)
            else:
                randDyn = 1
            
            franka_dof_stiffness = to_torch([400*randDyn, 400*randDyn, 400*randDyn, 400*randDyn, 400*randDyn, 400*randDyn, 400*randDyn, 1.0e6, 1.0e6], dtype=torch.float, device=self.device)
            franka_dof_damping = to_torch([80*randDyn, 80*randDyn, 80*randDyn, 80*randDyn, 80*randDyn, 80*randDyn, 80*randDyn, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)

            for mm in range(self.num_franka_dofs):
                franka_dof_props['stiffness'][mm] = franka_dof_stiffness[mm]
                franka_dof_props['damping'][mm] = franka_dof_damping[mm]

            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)


            #print("env_ptr: ", env_ptr)

            #Terrain generation
            if self.randTerrain:
                terrain_width = spacing*2
                terrain_length = spacing*2
                horizontal_scale = 0.25  # [m]
                vertical_scale = 0.005  # [m]
                num_rows = int(terrain_width/horizontal_scale)
                num_cols = int(terrain_length/horizontal_scale)
                heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)


                subt=SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)


                heightfield = random_uniform_terrain(subt, min_height=-0.2, max_height=0.0, step=0.01, downsampled_scale=0.5).height_field_raw #ds=0.5

                # add the terrain as a triangle mesh
                vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
                tm_params = gymapi.TriangleMeshParams()
                tm_params.nb_vertices = vertices.shape[0]
                tm_params.nb_triangles = triangles.shape[0]

                # #Friction for ground plane
                # tm_params.dynamic_friction = 0.01
                # tm_params.static_friction = 0.01
                

                if i % num_per_row == 0: 
                    tx = -spacing
                    ty += spacing*2
                
        
                tm_params.transform.p.x = tx
                tm_params.transform.p.y = ty

            
                self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)
                tx += spacing*2
            
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
                        prop_state_pose.p.x = 0.4 
                       
                        prop_state_pose.p.y = 0.0 
                        prop_state_pose.p.z = 0.026+0.014 
                        


                        # Use random positioning?

                        if self.randPos:
                            prop_state_pose.p.x = randrange_float(0.40, 0.80, self.prop_spacing)
                            # propz, propy = 0, p
                            prop_state_pose.p.y = randrange_float(-0.50, 0.50, self.prop_spacing)
                            
                            yaw = random.uniform(0.00000, 6.28318)
            

                        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                        

                        prop_state_pose.r = gymapi.Quat(qx, qy, qz, qw)

                        # choice of prop

                        wall_state_pose = gymapi.Transform()
                        wall_state_pose.p.x = prop_state_pose.p.x+0.10
                        wall_state_pose.p.y = prop_state_pose.p.y
                        wall_state_pose.p.z = 0.04

                        # wall_state_pose.p = gymapi.Vec3(0.5, 0.0, 0.04)

                        yaw=1.57

                        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
                        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
                        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
                        


                        wall_state_pose.r = gymapi.Quat(qx, qy, qz, qw)


                        
                        wall_actor = self.gym.create_actor(env_ptr, wall_asset, wall_state_pose, "wall", i, 2, 0)
                        self.gym.set_actor_dof_properties(env_ptr, wall_actor, wall_dof_props) # may use





                        if xount > len(self.rockList)-1:
                            xount=0
                        
                        
                        if self.randProp == "box":
                            prop_actor = self.gym.create_actor(env_ptr, box_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        elif self.randProp == "cyl":
                            prop_actor = self.gym.create_actor(env_ptr, cyl_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        elif self.randProp == "sphere":
                            prop_actor = self.gym.create_actor(env_ptr, sphere_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        elif self.randProp == "randRock":
                            #Random order
                            # x = chooseProp(len(self.rockList)-1)

                            #Non-random order
                            x = xount 

                            prop_actor = self.gym.create_actor(env_ptr, self.rockList[x], prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        elif self.randProp == "rand":
                            x = chooseProp(3)
                            if x==0: temp_asset=box_asset
                            elif x==1: temp_asset=cyl_asset
                            elif x==2: temp_asset=sphere_asset
                            # elif x==3: temp_asset=rock_asset
                            prop_actor = self.gym.create_actor(env_ptr, temp_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)

                        # xtra_actor = self.gym.create_actor(env_ptr, xtra_asset, prop_state_pose, "xtra{}".format(prop_count), i, 0, 0)
                        # prop_actor = self.gym.create_actor(env_ptr, prop_asset, prop_state_pose, "prop{}".format(prop_count), i, 0, 0)
                        prop_count += 1
                        xount += 1

                        prop_idx = j * props_per_row + k
                        self.default_prop_states.append([prop_state_pose.p.x , prop_state_pose.p.y, prop_state_pose.p.z,
                                                         prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])

                        # self.gym.set_actor_scale(env_ptr, prop_actor, 0.035) 

                        print("Wall pos: ", wall_state_pose.p, wall_state_pose.r)
                                                          
            #Camera setup
            camera_prop = self.camera_prop_setup()
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_prop)
            self.cams.append(camera_handle)
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.2, 0.0, 1.0), gymapi.Vec3(0.6, 0.0, 0.0))

            # wrap camera tensor in a pytorch tensor
            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.cam_tensors.append(torch_cam_tensor)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)
            # self.cabinets.append(cabinet_actor)
            #print("Environment no. ", i+1, " created")
        
        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, franka_actor, "panda_link7")
        #self.drawer_handle = self.gym.find_actor_rigid_body_handle(env_ptr, cabinet_actor, "drawer_top")
        self.prop_handle = self.gym.find_actor_rigid_body_handle(env_ptr, prop_actor, "prop_box")
        self.wall_handle = self.gym.find_actor_rigid_body_handle(env_ptr, wall_actor, "wall") 

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
        #print("Compute reward")
        self.gym.refresh_net_contact_force_tensor(self.sim)
        _force_vec = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_vec = gymtorch.wrap_tensor(_force_vec)
        force_vec = force_vec.view(-1, 12, 3) #Changed from 11 to 12
        _force_vec_franka = force_vec[:, :10]
        _force_vec_left = force_vec.select(1, 8)
        _force_vec_right = force_vec.select(1, 9)

        left_n = torch.nn.functional.normalize(_force_vec_left, dim=-1, p=2)
        right_n = torch.nn.functional.normalize(_force_vec_right, dim=-1, p=2)

        force_ang = torch.acos(torch.diagonal(torch.tensordot(left_n, right_n, dims=([1], [1]))))
    
        grip_tens = torch.where(force_ang > 2.8, 1, 0)
        grip_tens = torch.where(torch.norm(_force_vec_left, p=2, dim=-1) > 5, 
                     torch.where(torch.norm(_force_vec_right, p=2, dim=-1) > 5, grip_tens, 0), 0)

        collision_tens = torch.where(torch.norm(_force_vec_franka, p=2, dim=-1) > 100, 0, 1)
        collision_tens = ~torch.all(collision_tens, 1)

        self.rew_buf[:], self.reset_buf[:], self.success_counter[:], self.fail_counter[:], self.success_timer[:] = compute_franka_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.franka_grasp_pos, self.prop_grasp_pos, self.franka_grasp_rot, self.prop_grasp_rot,
            self.franka_lfinger_pos, self.franka_rfinger_pos,
            self.gripper_forward_axis, self.prop_inward_axis, self.gripper_up_axis, self.prop_up_axis,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.height_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.distX_offset, self.max_episode_length, grip_tens, collision_tens, self.reward_state, self.success_counter, self.fail_counter, self.success_timer
        )

        if(self.global_timer % 500 == 0):
            #Success rate
            total_success = torch.sum(self.success_counter)
            total_failure = torch.sum(self.fail_counter)
            print(f"Total Success rate: {total_success}/{total_success+total_failure} = {total_success/(total_success+total_failure)}")
            env_success = torch.div(self.success_counter, torch.add(self.success_counter, self.fail_counter))
            #print(env_success)


    def compute_observations(self):
        #print ("Compute observation")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
        prop_pos = self.rigid_body_states[:, self.prop_handle][:, 0:3]
        prop_rot = self.rigid_body_states[:, self.prop_handle][:, 3:7]

        # print("Wall: ", self.wall_handle)

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
        

        if self.using_camera:

            if self.global_timer == 1:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                if self.img_save: self.save_images(True)

                img_tensor = torch.stack((self.cam_tensors), 0)  # combine images
                img_tensor[img_tensor < -2] = 0 # remove faraway data

                camera_data = torch.reshape(img_tensor, (self.num_envs, self.cam_pixels))
                self.gym.end_access_image_tensors(self.sim)

            self.obs_buf= torch.cat((camera_data, dof_pos_scaled, self.franka_dof_vel * self.dof_vel_scale, self.franka_grasp_pos[:, 2].unsqueeze(-1)), 1)
            
        else:
            to_target = self.prop_grasp_pos - self.franka_grasp_pos # Distance to target
            self.obs_buf = torch.cat((dof_pos_scaled[:,:8], self.franka_dof_vel[:,:7] * self.dof_vel_scale, to_target, self.prop_grasp_pos[:, 2].unsqueeze(-1)), dim=-1) #self.prop_dof_vel[:, 3].unsqueeze(-1))
            #print(f"Obs_buf shape: {self.obs_buf.shape} test: {dof_pos_scaled[:,:7].shape}")
        return self.obs_buf


    def reset_idx(self, env_ids):
        #print("Running reset_idx")
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + self.randFrankaPos * (torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)


        self.franka_dof_pos[env_ids, :] = pos
        self.franka_dof_vel[env_ids, :] = torch.zeros_like(self.franka_dof_vel[env_ids])
        self.franka_dof_targets[env_ids, :self.num_franka_dofs] = pos
        
        # self.wall_dof_state[env_ids, :] = torch.zeros_like(self.wall_dof_state[env_ids])

        # reset props (Random)
        if self.randPos:

            prop_state_pose = gymapi.Transform()

            prop_state_pose.p.x = random.uniform(0.40, 0.80) 

            prop_state_pose.p.y = random.uniform(-0.50, 0.50)
            prop_state_pose.p.z = 0.026
            
            roll = 0
            pitch = 0
            yaw = random.uniform(0.00000, 6.28318)

            qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
            qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
            qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
            
            prop_state_pose.r = gymapi.Quat(qx, qy, qz, qw)
        
            self.default_prop_states = to_torch([prop_state_pose.p.x , prop_state_pose.p.y, prop_state_pose.p.z,
                                                    prop_state_pose.r.x, prop_state_pose.r.y, prop_state_pose.r.z, prop_state_pose.r.w,
                                                    0, 0, 0, 0, 0, 0], device=self.device, dtype=torch.float).repeat((self.num_envs, 1)).view(self.num_envs ,self.num_props, 13)
            

        if self.num_props > 0:
            prop_indices = self.global_indices[env_ids, 2:].flatten()
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
        
        self.reward_state[env_ids] = 0
        self.success_timer[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        
        # print("Pre_physx")
        # print("Prop_grasp_pos: ", self.prop_grasp_pos)
        self.actions = actions.clone().to(self.device)

        actionGrip = torch.cat((self.actions, self.actions[:,7].unsqueeze(-1)), 1)

        targets = self.franka_dof_targets[:, :self.num_franka_dofs] + self.franka_dof_speed_scales * self.dt * actionGrip * self.action_scale
        self.franka_dof_targets[:, :self.num_franka_dofs] = tensor_clamp(
            targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.franka_dof_targets))


    def post_physics_step(self):
        #print("Post Physics")
        self.progress_buf += 1
        self.global_timer += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)


        #print("        ")
        #print("Look:", env_ids)
        #print("        ")

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


    def set_gripper(self, gripper_action):
        gripper_state = torch.where(gripper_action <= 0.0,-1.0,1.0)


    def save_images(self, save_now):
        frame_no = self.gym.get_frame_count(self.sim)
        for i in range(self.num_envs):
            if frame_no % 100 == 0 or save_now:
                img_dir = "images"
                if not os.path.exists(img_dir):
                    os.mkdir(img_dir)

                cam_img = self.cam_tensors[i].cpu().numpy()
                cam_img[cam_img < -2] = 0
                cam_img= cam_img*100
                fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (frame_no, i))
                imageio.imwrite(fname, cam_img)


    def camera_prop_setup(self):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cam_width
        camera_props.height = self.cam_height
        camera_props.enable_tensors = True
        return camera_props


def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start


def chooseProp(x):
    return np.random.randint(0,x)



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
    finger_dist_reward_scale, action_penalty_scale, distX_offset, max_episode_length, grip, collision, reward_state, success_count, fail_count, success_time
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, float, float, float, float, float, float, float, float, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    # distance from hand to the drawer
    d = torch.norm(franka_grasp_pos - prop_grasp_pos, p=2, dim=-1)
    #Prob height
    prob_height = prop_grasp_pos[:, 2]
    prob_height = prob_height.double()
    # Time penalty
    time_penalty = 0.05
    action_penalty = torch.sum(actions ** 2, dim=-1) * action_penalty_scale

    finger_dist = torch.norm(franka_lfinger_pos - franka_rfinger_pos, p=2, dim=-1)

    #One time rewards
    # reward = torch.where(reward_state == 0, torch.where(d <= 0.40, 1, 0), 0)
    # reward_state = torch.where(reward_state == 0, torch.where(d <= 0.40, 1, reward_state), reward_state)
    
    # reward = torch.where(reward_state == 1, torch.where(d <= 0.1, 7, 0), 0)
    # reward_state = torch.where(reward_state == 1, torch.where(d <= 0.1, 2, reward_state), reward_state)

    # reward = torch.where(reward_state == 2, torch.where(grip_forces > 0.5, 49, 0), 0)
    # reward_state = torch.where(reward_state == 2, torch.where(grip_forces > 0.5, 3, reward_state), reward_state)

    #Continuous reward
    # close_reward = torch.where(d <= 0.3, 1.0, 0.0)
    # dist_reward = torch.where(d <= 0.06, 10, 0)

    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.06, dist_reward*2.0, dist_reward)

    grip_reward = torch.where(finger_dist>0.02, grip*50, 0)

    # height_reward = torch.where(prob_height>0.05, prob_height*prob_height*10000, 0.0)
    # height_reward = torch.where(prob_height>0.10, 100.0, height_reward)
    # height_reward = torch.where(prob_height>0.20, 200.0-prob_height*1000, height_reward)

    height_reward = torch.where(prob_height>0.05, prob_height*prob_height*10000, 0.0)
    height_reward = torch.where(prob_height>0.10, 100.0, height_reward)
    height_reward = torch.where(prob_height>0.30, 400-prob_height*1000, height_reward)

    #reset on Succes
    success_time = torch.where(prob_height>0.10, torch.where(prob_height<0.30, success_time+1, 0), 0)
    success_count = torch.where(success_time>30, success_count+1, success_count)
    reset_buf = torch.where(success_time>30, torch.ones_like(reset_buf), reset_buf)

    end_reward = (max_episode_length-progress_buf)*200
    end_reward = end_reward.double()
    success_reward = torch.where(success_time>30, end_reward, 0.0)
    
    #Reset on failure
    fail_count = torch.where(progress_buf >= max_episode_length - 1, fail_count+1, fail_count)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    #reset if prop somehow falls down
    reset_buf = torch.where(prop_grasp_pos[:, 2]<-0.3, torch.ones_like(reset_buf), reset_buf)

    #Reward
    rewards = dist_reward + grip_reward + height_reward + success_reward

    return rewards, reset_buf, success_count, fail_count, success_time



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
