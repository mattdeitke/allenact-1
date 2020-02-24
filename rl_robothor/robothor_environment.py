import typing
from typing import Any, Optional, Dict, List, Union
import logging
import random
import copy
from collections import OrderedDict
import math

import ai2thor
from ai2thor.controller import Controller
from ai2thor.util import metrics
import numpy as np

from utils.experiment_utils import recursive_update

LOGGER = logging.getLogger("embodiedrl")


class RoboThorEnvironment:
    config = dict(
        rotateStepDegrees=30.0,
        visibilityDistance=1.0,
        gridSize=0.25,
        agentType="stochastic",
        continuousMode=True,
        snapToGrid=False,
        agentMode="bot",
        width=640,
        height=480,
    )

    def __init__(self, **kwargs):
        recursive_update(self.config, {**kwargs, "agentMode": "bot"})
        self.controller = Controller(**self.config)
        self.known_good_location = self.agent_state()
        self.grids = {}

    def get_grid(self, reachable_points):
        points = {
            (
                round(p["x"] / self.config["gridSize"]),
                round(p["z"] / self.config["gridSize"]),
            ): p
            for p in reachable_points
        }

        assert len(reachable_points) == len(points)

        xmin, xmax = min([p[0] for p in points]), max([p[0] for p in points])
        zmin, zmax = min([p[1] for p in points]), max([p[1] for p in points])

        return {}, xmin, xmax, zmin, zmax

    def access_grid(self, target):
        if target not in self.grids[self.scene_name][0]:
            xmin, xmax, zmin, zmax = self.grids[self.scene_name][1:5]
            nx = xmax - xmin + 1
            nz = zmax - zmin + 1
            self.grids[self.scene_name][0][target] = -2 * np.ones(
                (nx, nz), dtype=np.float64
            )

        p = self.agent_to_grid()

        if self.grids[self.scene_name][0][target][p[0], p[1]] < -1.5:
            corners = self.path_corners_to_object(target)
            dist = self.path_corners_to_dist(corners)
            if dist == float("inf"):
                dist = -1.0
            self.grids[self.scene_name][0][target][p[0], p[1]] = dist
            return dist

        return self.grids[self.scene_name][0][target][p[0], p[1]]

    def add_valid_grid_points(self):
        if self.scene_name in self.grids:
            return

        self.grids[self.scene_name] = self.get_grid(self.currently_reachable_points)

    def object_reachable(self, object_type):
        # return True
        return self.access_grid(object_type) > -0.5

    def path_corners_to_object(self, object_type):
        pose = self.agent_state()
        position = {k: float(pose[k]) for k in ["x", "y", "z"]}
        rotation = {**pose["rotation"]} if "rotation" in pose else {}
        try:
            path = metrics.get_shortest_path_to_object_type(
                self.controller,
                object_type,
                position,
                rotation if len(rotation) > 0 else None,
            )
        except ValueError:
            path = []
        finally:
            self.controller.step("TeleportFull", **pose)
        return path

    def path_corners_to_dist(self, corners):
        if len(corners) == 0:
            return float("inf")

        sum = 0
        for it in range(1, len(corners)):
            sum += math.sqrt(
                (corners[it]["x"] - corners[it - 1]["x"]) ** 2
                + (corners[it]["z"] - corners[it - 1]["z"]) ** 2
            )
        return sum

    def agent_to_grid(self, xz_subsampling=1, rot_subsampling=1):
        pose = self.agent_state()
        p = {k: float(pose[k]) for k in ["x", "y", "z"]}

        xmin, xmax, zmin, zmax = self.grids[self.scene_name][1:5]
        x = int(np.clip(round(p["x"] / self.config["gridSize"]), xmin, xmax))
        z = int(np.clip(round(p["z"] / self.config["gridSize"]), zmin, zmax))

        rs = self.config["rotateStepDegrees"] * rot_subsampling
        shifted = pose["rotation"]["y"] + rs / 2
        normalized = shifted % 360.0
        r = int(round(normalized / rs))

        return ((x - xmin) // xz_subsampling, (z - zmin) // xz_subsampling, r)

    def dist_to_object(self, object_type):
        # return 0
        return self.access_grid(object_type)

    def agent_state(self):
        agent_meta = self.last_event.metadata["agent"]
        return {
            **{k: float(v) for k, v in agent_meta["position"].items()},
            "rotation": {k: float(v) for k, v in agent_meta["rotation"].items()},
            "horizon": float(agent_meta["cameraHorizon"]),
        }

    def reset(self, scene_name=None):
        if scene_name is not None and scene_name != self.scene_name:
            self.controller.reset(scene_name)
            assert self.last_action_success, "Could not reset to new scene"
            self.known_good_location = self.agent_state()
        else:
            assert (
                self.known_good_location is not None
            ), "Resetting scene without known good location"
            self.controller.step("TeleportFull", **self.known_good_location)
            assert self.last_action_success, "Could not reset to known good location"
        self.add_valid_grid_points()

    def randomize_agent_location(
        self, seed: int = None, partial_position: Optional[Dict[str, float]] = None
    ) -> Dict:
        """Teleports the agent to a random reachable location in the scene."""
        if partial_position is None:
            partial_position = {}
        k = 0
        state: Optional[Dict] = None

        while k == 0 or (not self.last_action_success and k < 10):
            self.reset()
            state = {**self.random_reachable_state(seed=seed), **partial_position}
            self.controller.step("TeleportFull", **state)
            k += 1

        if not self.last_action_success:
            LOGGER.warning(
                (
                    "Randomize agent location in scene {} and current random state {}"
                    " with seed {} and partial position {} failed in "
                    "10 attempts. Forcing the action."
                ).format(self.scene_name, state, seed, partial_position)
            )
            self.controller.step("TeleportFull", **state, force_action=True)  # type: ignore
            assert self.last_action_success, "Force action failed with {}".format(state)

        return self.agent_state()

    def random_reachable_state(self, seed: int = None) -> Dict:
        """Returns a random reachable location in the scene."""
        if seed is not None:
            random.seed(seed)
        xyz = random.choice(self.currently_reachable_points)
        rotation = random.choice(
            np.arange(0.0, 360.0, self.config["rotateStepDegrees"])
        )
        horizon = 0.0  # random.choice([0.0, 30.0, 330.0])
        return {
            **{k: float(v) for k, v in xyz.items()},
            "rotation": {"x": 0.0, "y": float(rotation), "z": 0.0},
            "horizon": float(horizon),
        }

    @property
    def currently_reachable_points(self) -> List[Dict[str, float]]:
        """List of {"x": x, "y": y, "z": z} locations in the scene that are
        currently reachable."""
        self.controller.step(action="GetReachablePositions")
        return self.last_action_return

    @property
    def scene_name(self) -> str:
        """Current ai2thor scene."""
        return self.controller.last_event.metadata["sceneName"]

    @property
    def current_frame(self) -> np.ndarray:
        """Returns rgb image corresponding to the agent's egocentric view."""
        return self.controller.last_event.frame

    @property
    def current_depth(self) -> np.ndarray:
        """Returns depth image corresponding to the agent's egocentric view."""
        return self.controller.last_event.depth_frame

    @property
    def last_event(self) -> ai2thor.server.Event:
        """Last event returned by the controller."""
        return self.controller.last_event

    @property
    def last_action(self) -> str:
        """Last action, as a string, taken by the agent."""
        return self.controller.last_event.metadata["lastAction"]

    @property
    def last_action_success(self) -> bool:
        """Was the last action taken by the agent a success?"""
        return self.controller.last_event.metadata["lastActionSuccess"]

    @property
    def last_action_return(self) -> Any:
        """Get the value returned by the last action (if applicable).

        For an example of an action that returns a value, see
        `"GetReachablePositions"`.
        """
        return self.controller.last_event.metadata["actionReturn"]

    def step(
        self, action_dict: Dict[str, Union[str, int, float]]
    ) -> ai2thor.server.Event:
        """Take a step in the ai2thor environment."""
        return self.controller.step(**action_dict)

    def stop(self):
        """Stops the ai2thor controller."""
        try:
            self.controller.stop()
        except Exception as e:
            LOGGER.warning(str(e))

    def all_objects(self) -> List[Dict[str, Any]]:
        """Return all object metadata."""
        return self.controller.last_event.metadata["objects"]

    def all_objects_with_properties(
        self, properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all objects with the given properties."""
        objects = []
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def visible_objects(self) -> List[Dict[str, Any]]:
        """Return all visible objects."""
        return self.all_objects_with_properties({"visible": True})