# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.usd     # noqa: E402
from isaacsim.core.api import World     # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid     # noqa: E402
from isaacsim.core.api.objects.ground_plane import GroundPlane     # noqa: E402
from pxr import Sdf, UsdLux     # noqa: E402

# Add Ground Plane
GroundPlane(prim_path="/World/GroundPlane", z_position=0)

# Add Light Source
stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)

# Add Visual Cubes
visual_cube = VisualCuboid(
    prim_path="/visual_cube",
    name="visual_cube",
    position=np.array([0, 0.5, 1.0]),
    size=0.3,
    color=np.array([255, 255, 0]),
)

visual_cube_static = VisualCuboid(
    prim_path="/visual_cube_static",
    name="visual_cube_static",
    position=np.array([0.5, 0, 0.5]),
    size=0.3,
    color=np.array([0, 255, 0]),
)

# Add Physics Cubes
dynamic_cube = DynamicCuboid(
    prim_path="/dynamic_cube",
    name="dynamic_cube",
    position=np.array([0, -0.5, 1.5]),
    size=0.3,
    color=np.array([0, 255, 255]),
    # mass=1.0,  # Set proper mass
)

# start a world to step simulator
my_world = World(stage_units_in_meters=1.0)

# start the simulator
for i in range(3):
    my_world.reset()
    print("simulator running", i)
    if i == 1:
        print("Adding Physics Properties to the Visual Cube")
        from isaacsim.core.prims import RigidPrim

        rigid_prim = RigidPrim("/visual_cube")
        # Set proper mass and inertia properties
        # rigid_prim.set_masses(np.array([1.0]))
        # Inertia tensor as 3x3 matrix flattened: [Ixx, Ixy, Ixz, Iyx, Iyy, Iyz, Izx, Izy, Izz]
        # rigid_prim.set_inertias(np.array([[0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]]))

    if i == 2:
        print("Adding Collision Properties to the Visual Cube")
        from isaacsim.core.prims import GeometryPrim

        prim = GeometryPrim("/visual_cube")
        prim.apply_collision_apis()

    for j in range(100):
        my_world.step(render=True)  # stepping through the simulation

# shutdown the simulator automatically
simulation_app.close()
