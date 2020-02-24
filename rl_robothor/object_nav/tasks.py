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
        self.is_robot = False
        self.cur_dist = self.env.dist_to_object(self.task_info["object_type"])
        self.visited = set(
            [self.env.agent_to_grid(xz_subsampling=4, rot_subsampling=3)]
        )  # squares of 1 m2, sectors of 90 deg

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
            self.env.agent_to_grid(xz_subsampling=4, rot_subsampling=3)
        )  # squares of 1 m2, sectors of 90 deg
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

    def scan(
        self, postaction, criterion, vert=False, default_action=END
    ) -> Tuple[int, bool]:
        # TODO move to environment
        ROT_ANGLE = 30
        GRID_SIZE = 0.25

        # rots = (
        #     [[]]
        #     + [[ROTATE_LEFT]] * (180 // ROT_ANGLE)
        #     + [[ROTATE_RIGHT]] * (180 // ROT_ANGLE - 1)
        # )
        rots = [[ROTATE_LEFT] * it for it in range((180 // ROT_ANGLE) + 1)] + [
            [ROTATE_RIGHT] * it for it in range(1, (180 // ROT_ANGLE))
        ][::-1]
        verts = [[], [LOOK_UP], [LOOK_DOWN]] if vert else [[]]

        old_dist = self.env.dist_to_object(self.task_info["object_type"])

        astate = self.env.agent_state()

        paths = {}
        for rit in range(360 // ROT_ANGLE):
            rpath = rots[rit]
            cstate = self.env.agent_state()
            for v in verts:
                vpath = rpath + v
                if len(v) > 0:
                    self.env.step({"action": v[0]})

                vpath.append(postaction)
                self.env.step({"action": postaction})

                new_dist = self.env.dist_to_object(self.task_info["object_type"])
                valid = criterion(old_dist, new_dist, GRID_SIZE)
                if valid:
                    if new_dist in paths:
                        paths[new_dist].append(vpath)
                    else:
                        paths[new_dist] = [vpath]

                self.env.step({"action": "TeleportFull", **cstate})
            self.env.step({"action": ROTATE_LEFT})

        self.env.step({"action": "TeleportFull", **astate})

        dists = paths.keys()
        if len(dists) == 0:
            return default_action, False

        dists = sorted(dists)
        cands = paths[dists[0]]
        cands = sorted(cands, key=lambda x: len(x))
        assert len(cands[0]) > 0
        return cands[0][0], True

    def scan_success(self, new_dist, old_dist, GRID_SIZE, eps=1e-6):
        if new_dist > -0.5:
            if old_dist > -0.5 and new_dist < old_dist:
                return (
                    self.last_action_success
                    and old_dist - new_dist < 2 * GRID_SIZE + eps
                )  # maybe we make an illegal move?
            else:
                return self.last_action_success
        else:
            return False

    def query_expert(self) -> Tuple[int, bool]:
        action2id = {action: it for it, action in enumerate(self._actions)}

        if self._is_goal_object_visible():
            act, valid = END, True
        else:
            if self.env.dist_to_object(self.task_info["object_type"]) == 0.0:
                # our snapped to grid expert thinks we're already there,
                # so try look up/down in different angles, else finish with False
                act, valid = self.scan(
                    lambda x: END,
                    lambda x, y, z: self._is_goal_object_visible(),
                    vert=True,
                    default_action=END,
                )
            else:
                act, valid = self.scan(
                    lambda x: MOVE_AHEAD,
                    self.scan_success,
                    vert=False,
                    default_action=END,
                )  # this might break the causality assumption, but at some point the agent will likely have observed the target

        return action2id[act], valid
