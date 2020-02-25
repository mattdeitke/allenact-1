from typing import Dict, List, Any, Tuple
import math

from ai2thor.util import metrics

from ..robothor_environment import RoboThorEnvironment
from rl_ai2thor.object_nav.tasks import ObjectNavTask as BaseObjectNavTask
from rl_base.sensor import Sensor
from rl_base.common import RLStepResult
from ..robothor_constants import (
    MOVE_AHEAD,
    MOVE_BACK,
    ROTATE_LEFT,
    ROTATE_RIGHT,
    LOOK_DOWN,
    LOOK_UP,
    END,
)


class ObjectNavTask(BaseObjectNavTask):
    """Defines the object navigation task in RoboTHOR.

    In object navigation an agent is randomly initialized into an AI2-THOR scene and must
    find an object of a given type (e.g. Tomato, Television, etc). An object is considered
    found if the agent takes an `End` action and the object is visible to the agent (see
    [here](https://ai2thor.allenai.org/documentation/concepts) for a definition of visibiliy
    in AI2-THOR).

    The actions available to an agent in this task are:

    1. Move ahead/back
        * Moves agent ahead/back by 0.25 meters.
    1. Rotate left / rotate right
        * Rotates the agent by 90 degrees counter-clockwise / clockwise.
    1. Look down / look up
        * Changes agent view angle by 30 degrees up or down. An agent cannot look more than 30
          degrees above horizontal or less than 60 degrees below horizontal.
    1. End
        * Ends the task and the agent receives a positive reward if the object type is visible to the agent,
        otherwise it receives a negative reward.

    # Attributes

    env : The robothor environment.
    sensor_suite: Collection of sensors formed from the `sensors` argument in the initializer.
    task_info : The task info. Must contain a field "object_type" that specifies, as a string,
        the goal object type.
    max_steps : The maximum number of steps an agent can take an in the task before it is considered failed.
    observation_space: The observation space returned on each step from the sensors.
    """

    _actions = (
        MOVE_AHEAD,
        MOVE_BACK,
        ROTATE_LEFT,
        ROTATE_RIGHT,
        LOOK_DOWN,
        LOOK_UP,
        END,
    )

    def __init__(
        self,
        env: RoboThorEnvironment,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_configs: Dict[str, Any],
        **kwargs
    ) -> None:
        super().__init__(env, sensors, task_info, max_steps)
        self.reward_configs = reward_configs
        self.cur_dist = self.env.dist_to_object(self.task_info["object_type"])
        self.visited = set(
            [self.env.quantized_agent_state(xz_subsampling=4, rot_subsampling=3)]
        )  # squares of 4 * 0.25 m2, sectors of 3 * 30 deg

    def shaping(self) -> float:
        if self.reward_configs["shaping_weight"] == 0.0:
            return 0.0

        # Reward getting closer to the target...
        new_dist = self.env.dist_to_object(self.task_info["object_type"])
        if self.cur_dist > -0.5 and new_dist > -0.5:
            rew = self.cur_dist - new_dist
        else:
            rew = 0.0
        self.cur_dist = new_dist

        # ...and also exploring! We won't be able to hit the optimal path in test
        old_visited = len(self.visited)
        self.visited.add(
            self.env.quantized_agent_state(xz_subsampling=4, rot_subsampling=3)
        )  # squares of 4 * 0.25 m2, sectors of 3 * 30 deg
        rew += self.reward_configs["exploration_shaping_weight"] * (
            len(self.visited) - old_visited
        )

        return rew * self.reward_configs["shaping_weight"]

    def judge(self) -> float:
        """ Judge the last event. """
        reward = self.reward_configs["step_penalty"]

        reward += self.shaping()

        if not self.last_action_success:
            reward += self.reward_configs["unsuccessful_action_penalty"]

        if self._took_end_action:
            reward += (
                self.reward_configs["goal_success_reward"]
                if self._success
                else self.reward_configs["failed_stop_reward"]
            )

        return float(reward)

    def _step(self, action: int) -> RLStepResult:
        action_str = self.action_names()[action]

        if action_str == END:
            self._took_end_action = True
            self._success = self._is_goal_object_visible()
            self.last_action_success = self._success
        else:
            self.env.step({"action": action_str})
            self.last_action_success = self.env.last_action_success

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success},
        )
        return step_result

    @classmethod
    def action_names(cls) -> Tuple[str, ...]:
        return cls._actions
