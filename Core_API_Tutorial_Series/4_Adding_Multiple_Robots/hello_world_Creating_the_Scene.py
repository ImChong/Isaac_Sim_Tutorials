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
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.api.tasks import BaseTask
import numpy as np

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class RobotsPlaying(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._jetbot_goal_position = np.array([130, 30, 0])
        # Add a subtask of pick and place instead
        # of writing the same task again
        # we just need to add a jetbot and change the positions of the assets and
        # the cube target position
        self._pick_place_task = PickPlace(cube_initial_position=np.array([0.1, 0.3, 0.05]),
                                          target_position=np.array([0.7, -0.3, 0.0515 / 2.0]))
        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._pick_place_task.set_up_scene(scene)
        pick_place_params = self._pick_place_task.get_params()
        self._franka = scene.get_object(pick_place_params["robot_name"]["value"])
        # Changes Franka's default position
        # so that it is set at this position after reset
        self._franka.set_world_pose(position=np.array([1.0, 0, 0]))
        self._franka.set_default_state(position=np.array([1.0, 0, 0]))

        assets_root_path = "/home/chong/isaac-sim-assets/isaac-sim-assets-robots_and_sensors-5.0.0/Assets/Isaac/5.0"
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
        self._jetbot = scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Robot",
                name="fancy_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, 0.3, 0]))
        )
        return

    def get_observations(self):
        current_jetbot_position, current_jetbot_orientation = self._jetbot.get_world_pose()
        # Observations needed to drive the jetbot to push the cube
        observations = {
            self._jetbot.name: {
                "position": current_jetbot_position,
                "orientation": current_jetbot_orientation,
                "goal_position": self._jetbot_goal_position
            }
        }
        return observations

    def get_params(self):
        # To avoid hard coding names..etc.
        pick_place_params = self._pick_place_task.get_params()
        params_representation = pick_place_params
        params_representation["franka_name"] = pick_place_params["robot_name"]
        params_representation["jetbot_name"] = {"value": self._jetbot.name, "modifiable": False}
        return params_representation

    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        return


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        # Add task here
        world.add_task(RobotsPlaying(name="awesome_task"))
        return

    async def setup_post_load(self):
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
