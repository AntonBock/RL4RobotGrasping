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
import imageio

from isaacgym import gymutil, gymtorch, gymapi
from tasks.base.vec_task import VecTask

class Cartpole(VecTask):


    # def set_viewer(self):
    #     """Create the viewer."""

    #     # todo: read from config
    #     self.enable_viewer_sync = True
    #     self.viewer = None

    #     # if running with a viewer, set up keyboard shortcuts and camera
    #     if self.headless == False:
    #         # subscribe to keyboard shortcuts
    #         self.viewer = self.gym.create_viewer(
    #             self.sim, gymapi.CameraProperties())
    #         self.gym.subscribe_viewer_keyboard_event(
    #             self.viewer, gymapi.KEY_ESCAPE, "QUIT")
    #         self.gym.subscribe_viewer_keyboard_event(
    #             self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    #         # set the camera position based on up axis
    #         sim_params = self.gym.get_sim_params(self.sim)
    #         if sim_params.up_axis == gymapi.UP_AXIS_Z:
    #             cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
    #             cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
    #         else:
    #             cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
    #             cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

    #         self.gym.viewer_camera_look_at(
    #             self.viewer, None, cam_pos, cam_target)




    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = 500
        
        self.cam_width = 64
        self.cam_height = 64
        self.cam_pixels = self.cam_width*self.cam_height

        self.non_cam_observations = 0
        self.cfg["env"]["numObservations"] = self.cam_height*self.cam_width+4
        self.cfg["env"]["numActions"] = 1

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]



    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/cartpole.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(cartpole_asset)

        pose = gymapi.Transform()
        pose.p.z = 2.0
        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.cartpole_handles = []
        self.cams = []
        self.cam_tensors = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.envs.append(env_ptr)

            cartpole_handle = self.gym.create_actor(env_ptr, cartpole_asset, pose, "cartpole", i, 1, 0)
            self.cartpole_handles.append(cartpole_handle)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, cartpole_handle)
            dof_props['driveMode'][0] = gymapi.DOF_MODE_EFFORT
            dof_props['driveMode'][1] = gymapi.DOF_MODE_NONE
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env_ptr, cartpole_handle, dof_props)
            
            #Camera setup
            camera_prop = self.camera_prop_setup()
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_prop)
            self.cams.append(camera_handle)
            self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(1.7, 0, 2), pose.p)

            # wrap camera tensor in a pytorch tensor
            cam_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            torch_cam_tensor = gymtorch.wrap_tensor(cam_tensor)
            self.cam_tensors.append(torch_cam_tensor)
       

    def compute_reward(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        # retrieve environment observations from buffer
        pole_angle = self.dof_pos[env_ids, 1].squeeze()
        pole_vel = self.dof_vel[env_ids, 1].squeeze()
        cart_vel = self.dof_vel[env_ids, 0].squeeze()
        cart_pos = self.dof_pos[env_ids, 0].squeeze()

        self.rew_buf[:], self.reset_buf[:] = compute_cartpole_reward(
            pole_angle, pole_vel, cart_vel, cart_pos,
            self.reset_dist, self.reset_buf, self.progress_buf, self.max_episode_length
        )

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        
        # frame_no = self.gym.get_frame_count(self.sim)
        # for i in env_ids:
        #     if frame_no % 100 == 0:
        #         img_dir = "images"
        #         if not os.path.exists(img_dir):
        #             os.mkdir(img_dir)

        #         cam_img = self.cam_tensors[i].cpu().numpy()
        #         cam_img[cam_img < -2] = 255
        #         fname = os.path.join(img_dir, "cam-%04d-%04d.png" % (frame_no, i))
        #         #self.gym.write_camera_image_to_file(self.sim, self.envs[i], self.cams[i], gymapi.IMAGE_DEPTH, fname)
        #         imageio.imwrite(fname, cam_img)

        #torch.set_printoptions(edgeitems=3)
        #torch.set_printoptions(profile="full")
        
        full_tensor = torch.stack((self.cam_tensors), 0)
        full_tensor[full_tensor < -2] = 0
        #print(f"full tensor shape: {full_tensor.shape}")
        #print(f"full tensor: {full_tensor}")

        #full_img = torch.reshape(full_img, (self.num_envs,self.cam_pixels))
        #print(f"camera image: {full_img}")
        #extra_data = torch.ones(self.num_envs, 1, dtype=torch.float32, device=self.device)*3.2
        #print(extra_data.shape)
        camera_data = torch.reshape(full_tensor, (self.num_envs, self.cam_height*self.cam_width))
        #print(camera_data.shape)
        self.obs_buf = torch.cat((camera_data, self.dof_pos, self.dof_vel), 1)

        self.gym.end_access_image_tensors(self.sim)
        
        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.2 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)
        velocities = 0.5 * (torch.rand((len(env_ids), self.num_dof), device=self.device) - 0.5)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        actions_tensor = torch.zeros(self.num_envs * self.num_dof, device=self.device, dtype=torch.float)
        actions_tensor[::self.num_dof] = actions.to(self.device).squeeze() * self.max_push_effort
        forces = gymtorch.unwrap_tensor(actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

    def camera_prop_setup(self):
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.cam_width
        camera_props.height = self.cam_height
        camera_props.enable_tensors = True
        return camera_props


@torch.jit.script
def compute_cartpole_reward(pole_angle, pole_vel, cart_vel, cart_pos,
                            reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward is combo of angle deviated from upright, velocity of cart, and velocity of pole moving
    reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)

    # adjust reward for reset agents
    reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
    reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

    reset = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return reward, reset


