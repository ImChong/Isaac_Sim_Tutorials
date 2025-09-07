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

# Can be used to create a new cube or to point to an already existing cube in stage.
from isaacsim.core.api.objects import DynamicCuboid
import numpy as np

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()

        fancy_cube = world.scene.add(       # noqa: F841
            DynamicCuboid(
                prim_path="/World/random_cube",     # The prim path of the cube in the USD stage
                name="fancy_cube",                  # The unique name used to retrieve the object from the scene later on
                position=np.array([0, 0, 1.0]),     # Using the current stage units which is in meters by default.
                scale=np.array([0.5015, 0.5015, 0.5015]),     # most arguments accept mainly numpy arrays.
                color=np.array([0, 0, 1.0]),        # RGB channels, going from 0-1
            )
        )

        return

    # Here we assign the class's variables
    # this function is called after load button is pressed
    # regardless starting from an empty stage or not
    # this is called after setup_scene and after
    # one physics time step to propagate appropriate
    # physics handles which are needed to retrieve
    # many physical properties of the different objects
    async def setup_post_load(self):
        self._world = self.get_world()
        self._cube = self._world.scene.get_object("fancy_cube")
        # position, orientation = self._cube.get_world_pose()
        # linear_velocity = self._cube.get_linear_velocity()
        # # will be shown on terminal
        # print("Cube position is : " + str(position))
        # print("Cube's orientation is : " + str(orientation))
        # print("Cube's linear velocity is : " + str(linear_velocity))
        self._world.add_physics_callback("sim_step", callback_fn=self.print_cube_info)  # callback names have to be unique
        return

    # here we define the physics callback to be called before each physics step, all physics callbacks must take
    # step_size as an argument
    def print_cube_info(self, step_size):
        position, orientation = self._cube.get_world_pose()
        linear_velocity = self._cube.get_linear_velocity()
        # will be shown on terminal
        print("Cube position is : " + str(position))
        print("Cube's orientation is : " + str(orientation))
        print("Cube's linear velocity is : " + str(linear_velocity))

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
