import random
import warnings
import typing
from typing import Optional, Dict, List, Any
from collections import OrderedDict
import logging

import gym
from ai2thor.util import metrics

from .tasks import ObjectNavTask
from ..robothor_environment import RoboThorEnvironment
from rl_ai2thor.object_nav.task_samplers import (
    ObjectNavTaskSampler as BaseObjectNavTaskSampler,
)
from rl_base.sensor import Sensor

LOGGER = logging.getLogger("embodiedrl")


class ObjectNavTaskSampler(BaseObjectNavTaskSampler):
    """Samples object navigation tasks in RoboTHOR."""

    def __init__(
        self,
        scenes: List[str],
        object_types: str,
        sensors: List[Sensor],
        max_steps: int,
        env_args: Dict[str, Any],
        action_space: gym.Space,
        rewards_config: Dict,
        scene_period: Optional[int] = None,
        max_tasks: Optional[int] = None,
        seed: Optional[int] = None,
        deterministic_cudnn: bool = False,
        *args,
        **kwargs
    ):
        self.rewards_config = rewards_config
        super().__init__(
            scenes,
            object_types,
            sensors,
            max_steps,
            env_args,
            action_space,
            scene_period,
            max_tasks,
            seed,
            deterministic_cudnn,
            *args,
            **kwargs,
        )

    def next_task(self, force_advance_scene=False) -> Optional[ObjectNavTask]:
        """Samples a new object navigation task where the target object is reachable from the initial location if tasks
         are available."""
        if self.max_tasks is not None and self.max_tasks <= 0:
            return None

        scene = self.sample_scene(force_advance_scene)

        if self.env is not None:
            if scene != self.env.scene_name:
                self.env.reset(scene)
        else:
            self.env = self._create_environment()
            self.env.reset(scene_name=scene)

        object_types_in_scene = set(
            [o["objectType"] for o in self.env.last_event.metadata["objects"]]
        )

        # Assume at least one target will be reachable:
        task_info = OrderedDict()
        for ot in random.sample(self.object_types, len(self.object_types)):
            if ot in object_types_in_scene:
                for attempt in range(5):
                    pose = self.env.randomize_agent_location()
                    if self.env.object_reachable(ot):
                        task_info["object_type"] = ot
                        task_info["start_pose"] = pose
                        break
                    else:
                        LOGGER.warning(
                            "Skipping non-reachable {} from {} in {}".format(
                                ot, pose, scene
                            )
                        )
                        continue
                if len(task_info) > 0:
                    break

        if len(task_info) == 0:
            warnings.warn(
                "Scene {} does not contain any"
                " objects of any of the types {}.".format(scene, self.object_types)
            )

        task_info["actions"] = []

        self._last_sampled_task = ObjectNavTask(
            env=typing.cast(RoboThorEnvironment, self.env),
            sensors=self.sensors,
            task_info=task_info,
            max_steps=self.max_steps,
            action_space=self._action_space,
            reward_configs=self.rewards_config,
        )
        return self._last_sampled_task

    def _create_environment(self):
        """Creates the RoboTHOR environment"""
        env = RoboThorEnvironment(**self.env_args)
        return env
