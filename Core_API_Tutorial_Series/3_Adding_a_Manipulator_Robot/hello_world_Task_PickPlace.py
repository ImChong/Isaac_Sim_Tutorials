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
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        # We add the task to the world here
        world.add_task(PickPlace(name="awesome_task"))
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        # The world already called the setup_scene from the task so
        # we can retrieve the task objects
        # Each defined task in the robot extensions
        # has set_params and get_params to allow for changing tasks during
        # simulation, {"task_param_name": "value": [value], "modifiable": [True/ False]}
        task_params = self._world.get_task("awesome_task").get_params()
        self._franka = self._world.scene.get_object(task_params["robot_name"]["value"])
        self._cube_name = task_params["cube_name"]["value"]
        self._controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka,
        )
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        print("=" * 25, "setup_post_reset", "=" * 25)
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Gets all the tasks observations
        current_observations = self._world.get_observations()
        actions = self._controller.forward(
            picking_position=current_observations[self._cube_name]["position"],
            placing_position=current_observations[self._cube_name]["target_position"],
            current_joint_positions=current_observations[self._franka.name]["joint_positions"],
        )
        self._franka.apply_action(actions)
        if self._controller.is_done():
            self._world.pause()
        return

    def world_cleanup(self):
        return
