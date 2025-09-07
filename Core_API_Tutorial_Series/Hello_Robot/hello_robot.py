# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.core.utils.types import ArticulationAction
# from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
import numpy as np
import carb

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        # you configure a new server with /Isaac folder in it
        assets_root_path = "/home/chong/isaac-sim-assets/isaac-sim-assets-robots_and_sensors-5.0.0/Assets/Isaac/5.0"
        if assets_root_path is None:
            # Use carb to log warnings, errors, and infos in your application (shown on terminal)
            carb.log_error("Could not find nucleus server with /Isaac folder")
        asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"

        # This will create a new XFormPrim and point it to the USD file as a reference
        # Similar to how pointers work in memory
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")
        # Wrap the jetbot prim root under a Robot class and add it to the Scene
        # to use high level api to set/ get attributes as well as initializing
        # physics handles needed..etc.
        # Note: this call doesn't create the Jetbot in the stage window, it was already
        # created with the add_reference_to_stage
        jetbot_robot = world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))
        # Note: before a reset is called, we can't access information related to an Articulation
        # because physics handles are not initialized yet. setup_post_load is called after
        # the first reset so we can do so there
        print("Num of degrees of freedom before first reset: " + str(jetbot_robot.num_dof))     # prints None
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("fancy_robot")
        # Print info about the jetbot after the first reset is called
        print("Num of degrees of freedom after first reset: " + str(self._jetbot.num_dof))      # prints 2
        print("Joint Positions after first reset: " + str(self._jetbot.get_joint_positions()))
        # This is an implicit PD controller of the jetbot/ articulation
        # setting PD gains, applying actions, switching control modes..etc.
        # can be done through this controller.
        # Note: should be only called after the first reset happens to the world
        self._jetbot_articulation_controller = self._jetbot.get_articulation_controller()
        # Adding a physics callback to send the actions to apply actions with every
        # physics step executed.
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)
        return

    def send_robot_actions(self, step_size):
        # Every articulation controller has apply_action method
        # which takes in ArticulationAction with joint_positions, joint_efforts and joint_velocities
        # as optional args. It accepts numpy arrays of floats OR lists of floats and None
        # None means that nothing is applied to this dof index in this step
        # ALTERNATIVELY, same method is called from self._jetbot.apply_action(...)
        current_time = self._world.current_time
        if current_time < 5.0:
            self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None,
                                                                                 joint_efforts=None,
                                                                                 joint_velocities=np.array([10.0, -10.0])))
        else:
            self._jetbot_articulation_controller.apply_action(ArticulationAction(joint_positions=None,
                                                                                 joint_efforts=None,
                                                                                 joint_velocities=np.array([0.0, 0.0])))
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
