import json
from math import ceil
from typing import Dict, Any, List, Optional

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from models.resnet_tensor_object_nav_models import ResnetTensorObjectNavActorCritic
from onpolicy_sync.losses import PPO
from onpolicy_sync.losses.ppo import PPOConfig
from rl_ai2thor.ai2thor_sensors import RGBSensorThor, GoalObjectTypeThorSensor
from rl_base.experiment_config import ExperimentConfig
from rl_base.preprocessor import ObservationSet
from rl_base.task import TaskSampler
from rl_habitat.habitat_preprocessors import ResnetPreProcessorHabitat
from rl_robothor.robothor_task_samplers import ObjectNavTaskSampler
from rl_robothor.robothor_tasks import ObjectNavTask
from utils.experiment_utils import Builder, PipelineStage, TrainingPipeline, LinearDecay
from utils.viz_utils import (
    SimpleViz,
    TrajectoryViz,
    ActorViz,
    AgentViewViz,
    TensorViz1D,
    TensorViz2D,
)
from rl_robothor.robothor_viz import ThorViz


class CorelVizExperimentConfig(ExperimentConfig):
    """An Object Navigation experiment configuration in RoboThor."""

    TRAIN_SCENES = [
        "FloorPlan_Train%d_%d" % (wall + 1, furniture + 1)
        for wall in range(12)
        for furniture in range(5)
    ]

    VALID_SCENES = [
        "FloorPlan_Val%d_%d" % (wall + 1, furniture + 1)
        for wall in range(3)
        for furniture in range(5)
    ]

    # TEST_SCENES = [
    #     "FloorPlan_test-dev%d_%d" % (wall + 1, furniture + 1)
    #     for wall in range(2)
    #     for furniture in range(2)
    # ]
    # For font size settings:
    # TEST_SCENES = "rl_robothor/data/minival.json"
    TEST_SCENES = "rl_robothor/data/val.json"
    ep_ids = ["Val_3_3_House Plant_32", "Val_1_2_Alarm Clock_25", "Val_3_2_Laptop_31"]
    # ep_ids = [
    #     "Val_3_1_House Plant_30",
    #     "Val_1_5_BasketBall_23",
    #     "Val_3_2_Alarm Clock_36",
    #     "Val_3_4_Television_31",
    #     "Val_1_3_Garbage Can_19",
    #     "Val_2_4_Television_20",
    #     "Val_2_5_Spray Bottle_22",
    #     "Val_3_5_Garbage Can_35",
    #     "Val_1_4_House Plant_23",
    #     "Val_1_2_Apple_26",
    #     "Val_1_4_Television_20",
    #     "Val_1_5_Spray Bottle_24",
    #     "Val_1_2_Alarm Clock_25",
    #     "Val_3_3_House Plant_32",
    #     "Val_2_3_Laptop_22",
    #     "Val_3_2_Laptop_31",
    #     "Val_2_1_Baseball Bat_27",
    #     "Val_3_3_Alarm Clock_23",
    #     "Val_3_1_Garbage Can_33",
    #     "Val_3_3_Garbage Can_29",
    #     "Val_3_2_Television_29",
    #     "Val_3_4_Baseball Bat_25",
    #     "Val_3_1_Bowl_33",
    #     "Val_1_4_Apple_20",
    #     "Val_3_3_Apple_34",
    #     "Val_3_4_BasketBall_30",
    #     "Val_1_1_Garbage Can_26",
    #     "Val_2_3_Vase_23",
    #     "Val_2_3_House Plant_25",
    #     "Val_3_3_Bowl_36",
    #     "Val_3_5_Television_33",
    #     "Val_1_1_Laptop_30",
    #     "Val_1_1_BasketBall_25",
    #     "Val_2_4_Garbage Can_25",
    #     "Val_3_4_House Plant_20",
    #     "Val_1_2_Spray Bottle_28",
    #     "Val_2_5_Television_22",
    #     "Val_3_3_Laptop_39",
    #     "Val_1_2_House Plant_30",
    #     "Val_2_1_BasketBall_22",
    #     "Val_2_2_BasketBall_25",
    #     "Val_3_2_Baseball Bat_35",
    #     "Val_3_1_Alarm Clock_29",
    #     "Val_1_3_House Plant_22",
    #     "Val_1_5_Apple_26",
    #     "Val_3_4_Spray Bottle_34",
    #     "Val_2_5_House Plant_24",
    #     "Val_1_5_Garbage Can_21",
    #     "Val_3_2_Vase_32",
    #     "Val_2_1_House Plant_32",
    #     "Val_2_1_Bowl_30",
    #     "Val_1_4_Vase_26",
    #     "Val_3_1_Apple_26",
    #     "Val_1_5_House Plant_20",
    #     "Val_2_1_Apple_34",
    #     "Val_2_5_BasketBall_28",
    #     "Val_3_5_Apple_34",
    #     "Val_3_4_Garbage Can_29",
    #     "Val_2_1_Garbage Can_31",
    #     "Val_2_2_Television_28",
    #     "Val_2_1_Television_25",
    #     "Val_3_5_Mug_24",
    #     "Val_1_5_Laptop_23",
    #     "Val_1_4_BasketBall_28",
    #     "Val_1_4_Baseball Bat_23",
    #     "Val_3_5_Bowl_38",
    #     "Val_1_4_Bowl_18",
    #     "Val_1_5_Alarm Clock_18",
    #     "Val_3_1_Television_37",
    #     "Val_2_3_Baseball Bat_20",
    #     "Val_3_3_Baseball Bat_31",
    #     "Val_3_5_Laptop_33",
    #     "Val_2_3_Mug_22",
    #     "Val_2_3_BasketBall_25",
    #     "Val_1_4_Mug_23",
    #     "Val_1_3_Apple_29",
    #     "Val_1_5_Bowl_26",
    #     "Val_1_1_House Plant_21",
    #     "Val_2_5_Mug_29",
    #     "Val_2_5_Garbage Can_18",
    #     "Val_2_4_Television_14",
    #     "Val_1_5_BasketBall_16",
    #     "Val_3_2_House Plant_12",
    #     "Val_3_4_Television_19",
    #     "Val_3_5_Garbage Can_18",
    #     "Val_1_4_Apple_15",
    #     "Val_3_2_Television_10",
    #     "Val_3_4_House Plant_12",
    #     "Val_3_2_Alarm Clock_26",
    #     "Val_3_2_BasketBall_18",
    #     "Val_2_5_Spray Bottle_16",
    #     "Val_2_1_Apple_17",
    #     "Val_2_2_Television_12",
    #     "Val_1_3_Garbage Can_9",
    #     "Val_3_4_BasketBall_21",
    #     "Val_2_1_Baseball Bat_13",
    #     "Val_1_5_Apple_14",
    #     "Val_2_1_Garbage Can_15",
    #     "Val_3_3_Television_20",
    #     "Val_2_3_Laptop_12",
    #     "Val_1_4_House Plant_19",
    #     "Val_3_3_Baseball Bat_20",
    #     "Val_3_1_Garbage Can_17",
    #     "Val_3_1_BasketBall_17",
    #     "Val_3_1_Apple_14",
    #     "Val_3_1_Television_20",
    #     "Val_1_3_BasketBall_10",
    #     "Val_3_3_Alarm Clock_15",
    #     "Val_1_4_Vase_13",
    #     "Val_3_2_Laptop_24",
    #     "Val_1_2_Spray Bottle_20",
    #     "Val_2_3_BasketBall_14",
    #     "Val_3_2_Baseball Bat_21",
    #     "Val_1_2_Apple_15",
    #     "Val_3_3_Apple_20",
    #     "Val_3_3_Garbage Can_21",
    #     "Val_3_5_Television_16",
    #     "Val_1_1_Garbage Can_21",
    #     "Val_1_5_Spray Bottle_13",
    #     "Val_2_1_BasketBall_19",
    #     "Val_1_2_Alarm Clock_15",
    #     "Val_3_4_Spray Bottle_22",
    #     "Val_1_5_House Plant_14",
    #     "Val_1_1_Laptop_11",
    #     "Val_3_1_Bowl_18",
    #     "Val_3_3_House Plant_15",
    #     "Val_2_5_House Plant_19",
    #     "Val_2_3_Television_12",
    # ]
    video_ids = None
    # # Produces error for two testers:
    NUM_TEST_SCENES = 6116  # 6116

    CAMERA_WIDTH = 400
    CAMERA_HEIGHT = 300

    SCREEN_SIZE = 224

    MAX_STEPS = 500
    ADVANCE_SCENE_ROLLOUT_PERIOD = 10000000000000  # if more than 1 scene per worker

    VALIDATION_SAMPLES_PER_SCENE = 1

    NUM_PROCESSES = 56  # TODO 2 for debugging

    TARGET_TYPES = sorted(
        [
            "AlarmClock",
            "Apple",
            "BaseballBat",
            "BasketBall",
            "Bowl",
            "GarbageCan",
            "HousePlant",
            "Laptop",
            "Mug",
            "Remote",  # now it's called RemoteControl, so all epsiodes for this object will be random
            "SprayBottle",
            "Television",
            "Vase",
            # 'AlarmClock',
            # 'Apple',
            # 'BasketBall',
            # 'Mug',
            # 'Television',
        ]
    )

    SENSORS = [
        RGBSensorThor(
            height=SCREEN_SIZE,
            width=SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
        ),
        GoalObjectTypeThorSensor(object_types=TARGET_TYPES,),
    ]

    PREPROCESSORS = [
        Builder(
            ResnetPreProcessorHabitat,
            dict(
                input_height=SCREEN_SIZE,
                input_width=SCREEN_SIZE,
                output_width=7,
                output_height=7,
                output_dims=512,
                pool=False,
                torchvision_resnet_model=models.resnet18,
                input_uuids=["rgb_lowres"],
                output_uuid="rgb_resnet",
                parallel=False,  # TODO False for debugging
            ),
        ),
    ]

    OBSERVATIONS = [
        "rgb_resnet",
        "goal_object_type_ind",
    ]

    ENV_ARGS = dict(
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        continuousMode=True,
        applyActionNoise=True,
        # agentType="stochastic",
        rotateStepDegrees=45.0,
        visibilityDistance=1.5,
        gridSize=0.25,
        snapToGrid=False,
        agentMode="bot",
        # include_private_scenes=True,
    )

    @classmethod
    def tag(cls):
        return "ObjectNavigationResnet"

    @classmethod
    def training_pipeline(cls, **kwargs):
        ppo_steps = int(3e8)
        lr = 3e-4
        num_mini_batch = 1
        update_repeats = 3
        num_steps = 30
        save_interval = 200000
        log_interval = 1
        gamma = 0.99
        use_gae = True
        gae_lambda = 0.95
        max_grad_norm = 0.5
        return TrainingPipeline(
            save_interval=save_interval,
            metric_accumulate_interval=log_interval,
            optimizer_builder=Builder(optim.Adam, dict(lr=lr)),
            num_mini_batch=num_mini_batch,
            update_repeats=update_repeats,
            max_grad_norm=max_grad_norm,
            num_steps=num_steps,
            named_losses={"ppo_loss": Builder(PPO, kwargs={}, default=PPOConfig,)},
            gamma=gamma,
            use_gae=use_gae,
            gae_lambda=gae_lambda,
            advance_scene_rollout_period=cls.ADVANCE_SCENE_ROLLOUT_PERIOD,
            pipeline_stages=[
                PipelineStage(loss_names=["ppo_loss"], max_stage_steps=ppo_steps)
            ],
            lr_scheduler_builder=Builder(
                LambdaLR, {"lr_lambda": LinearDecay(steps=ppo_steps)}
            ),
        )

    # def machine_params(self, mode="train", **kwargs):
    #     if mode == "train":
    #         nprocesses = 1 if not torch.cuda.is_available() else self.NUM_PROCESSES  # TODO default 2 for debugging
    #         sampler_devices = [1, 2, 3, 4, 5, 6]  # TODO vs4 only has 7 gpus
    #         gpu_ids = [] if not torch.cuda.is_available() else [0]
    #         render_video = False
    #     elif mode == "valid":
    #         nprocesses = 1  # TODO debugging (0)
    #         if not torch.cuda.is_available():
    #             gpu_ids = []
    #         else:
    #             gpu_ids = [0]
    #         render_video = False
    #     elif mode == "test":
    #         nprocesses = 1
    #         if not torch.cuda.is_available():
    #             gpu_ids = []
    #         else:
    #             gpu_ids = [0]
    #         render_video = True
    #     else:
    #         raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")
    #
    #     # Disable parallelization for validation process
    #     prep_args = {}
    #     if mode == "valid":
    #         prep_args["parallel"] = False
    #     observation_set = ObservationSet(
    #         self.OBSERVATIONS, [prep(config=prep_args) for prep in self.PREPROCESSORS], self.SENSORS
    #     ) if nprocesses > 0 else None
    #
    #     return {
    #         "nprocesses": nprocesses,
    #         "gpu_ids": gpu_ids,
    #         "sampler_devices": sampler_devices if mode == "train" else gpu_ids,
    #         "observation_set": observation_set,
    #         "render_video": render_video,
    #     }

    def split_num_processes(self, ndevices, nprocesses=None):
        if nprocesses is None:
            nprocesses = self.NUM_PROCESSES
        assert nprocesses >= ndevices, "NUM_PROCESSES {} < ndevices {}".format(
            nprocesses, ndevices
        )
        res = [0] * ndevices
        for it in range(nprocesses):
            res[it % ndevices] += 1
        return res

    def machine_params(self, mode="train", **kwargs):
        if mode == "train":
            # gpu_ids = [] if not torch.cuda.is_available() else [0]
            # nprocesses = 1 if not torch.cuda.is_available() else self.NUM_PROCESSES
            # sampler_devices = [1]
            # render_video = False
            workers_per_device = 1
            gpu_ids = (
                []
                if not torch.cuda.is_available()
                else [0, 1, 2, 3, 4, 5, 6] * workers_per_device
            )  # TODO vs4 only has 7 gpus
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else self.split_num_processes(len(gpu_ids))
            )
            render_video = False
            visualizer = None
        elif mode == "valid":
            nprocesses = 1
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [0]
            render_video = False
            visualizer = None
        elif mode == "test":
            if not torch.cuda.is_available():
                gpu_ids = []
            else:
                gpu_ids = [0, 1, 2, 3, 4, 5, 6]  # TODO vs4 only has 7 gpus
                # gpu_ids = [0]  # TODO vs4 only has 7 gpus

            # nprocesses = 2  # To replicate error with two testers
            nprocesses = (
                1
                if not torch.cuda.is_available()
                else self.split_num_processes(
                    len(gpu_ids), self.NUM_PROCESSES
                )  # total=4
            )

            # render_video = False
            if len(self.ep_ids) == 128:
                # self.video_ids = [[ep] for ep in self.ep_ids]
                self.ep_ids = [self.ep_ids[i : i + 8] for i in range(0, 128, 8)]
                self.video_ids = self.ep_ids
            if self.video_ids is None:
                self.video_ids = [[ep] for ep in self.ep_ids]
            # if self.ep_ids is None:
            #     with open(self.TEST_SCENES, "r") as f:
            #         all_eps = json.load(f)[
            #             self.TEST_SHIFT : self.TEST_SHIFT + self.NUM_TEST_SCENES
            #         ]  # TODO take a small number of samples from shifted starting point
            #         self.ep_ids = [
            #             ep["id"] for ep in all_eps[: self.NUM_TEST_SCENES // 2]
            #         ]  # TODO keep only first half for first group
            #         self.ep_ids = [
            #             self.ep_ids,
            #             [ep["id"] for ep in all_eps[self.NUM_TEST_SCENES // 2 :]],
            #         ]  # TODO keep only second half for second group
            #         # self.video_ids = [ep["id"] for ep in all_eps[-1:]]
            #         self.video_ids = [[ep["id"]] for ep in all_eps]

            # self.video_ids = ["Val_2_1_Garbage Can_0"]

            # print(self.video_ids)

            visualizer = Builder(
                SimpleViz,
                dict(
                    episode_ids=self.ep_ids,
                    mode="test",
                    # v1=Builder(TrajectoryViz, dict()),
                    v3=Builder(ActorViz, dict(figsize=(3.25, 10), fontsize=(18))),
                    # v4=Builder(TensorViz1D, dict()),
                    # v5=Builder(TensorViz1D, dict(rollout_source=("masks"))),
                    # v6=Builder(TensorViz2D, dict()),
                    v7=Builder(
                        ThorViz, dict(figsize=(16, 8), viz_rows_cols=(448, 448))
                    ),
                    v2=Builder(
                        AgentViewViz,
                        dict(max_video_length=100, episode_ids=self.video_ids),
                    ),
                ),
            )
            # visualizer = None
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        # Disable preprocessor naive parallelization for eval
        if mode in ["valid", "test"]:
            for prep in self.PREPROCESSORS:
                prep.kwargs["parallel"] = False

        observation_set = (
            Builder(
                ObservationSet,
                kwargs=dict(
                    source_ids=self.OBSERVATIONS,
                    all_preprocessors=self.PREPROCESSORS,
                    all_sensors=self.SENSORS,
                ),
            )
            if (isinstance(nprocesses, int) and nprocesses > 0)
            or (isinstance(nprocesses, List) and max(nprocesses) > 0)
            else None
        )

        return {
            "nprocesses": nprocesses,
            "gpu_ids": gpu_ids,
            "observation_set": observation_set,
            "visualizer": visualizer,
        }

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        return ResnetTensorObjectNavActorCritic(
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            observation_space=kwargs["observation_set"].observation_spaces,
            goal_sensor_uuid="goal_object_type_ind",
            resnet_preprocessor_uuid="rgb_resnet",
            rnn_hidden_size=512,
            goal_dims=32,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

    @staticmethod
    def _partition_inds(n: int, num_parts: int):
        return np.round(np.linspace(0, n, num_parts + 1, endpoint=True)).astype(
            np.int32
        )

    def _get_sampler_args_for_scene_split(
        self,
        scenes: List[str],
        process_ind: int,
        total_processes: int,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if total_processes > len(scenes):  # oversample some scenes -> bias
            if total_processes % len(scenes) != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisible by the number of scenes"
                )
            scenes = scenes * int(ceil(total_processes / len(scenes)))
            scenes = scenes[: total_processes * (len(scenes) // total_processes)]
        else:
            if len(scenes) % total_processes != 0:
                print(
                    "Warning: oversampling some of the scenes to feed all processes."
                    " You can avoid this by setting a number of workers divisor of the number of scenes"
                )
        inds = self._partition_inds(len(scenes), total_processes)

        return {
            "scenes": scenes[inds[process_ind] : inds[process_ind + 1]],
            "object_types": self.TARGET_TYPES,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
            "rewards_config": {
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "failed_stop_reward": 0.0,
                "shaping_weight": 1.0,  # applied to the decrease in distance to target
            },
        }

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.TRAIN_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = "manual"
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        res["allow_flipping"] = True
        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        res = self._get_sampler_args_for_scene_split(
            self.VALID_SCENES,
            process_ind,
            total_processes,
            seeds=seeds,
            deterministic_cudnn=deterministic_cudnn,
        )
        res["scene_period"] = self.VALIDATION_SAMPLES_PER_SCENE
        res["max_tasks"] = self.VALIDATION_SAMPLES_PER_SCENE * len(res["scenes"])
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        res["env_args"]["x_display"] = (
            ("0.%d" % devices[process_ind % len(devices)])
            if devices is not None and len(devices) > 0
            else None
        )
        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        from utils.system import get_logger

        get_logger().info(
            "sampler_args process {} total {}".format(process_ind, total_processes)
        )
        inds = self._partition_inds(self.NUM_TEST_SCENES, total_processes)
        res = dict(
            scenes=self.TEST_SCENES,  # special case: dataset file name (triggered by dataset_first, dataset_last >=0)
            object_types=self.TARGET_TYPES,
            max_steps=100,  # TODO self.MAX_STEPS,
            sensors=self.SENSORS,
            action_space=gym.spaces.Discrete(len(ObjectNavTask.class_action_names())),
            seed=seeds[process_ind] if seeds is not None else None,
            deterministic_cudnn=deterministic_cudnn,
            dataset_first=inds[process_ind],
            # + self.TEST_SHIFT,  # TODO sample other episodes
            dataset_last=inds[process_ind + 1] - 1,
            # + self.TEST_SHIFT,  # TODO sample other episodes
            rewards_config={
                "step_penalty": -0.01,
                "goal_success_reward": 10.0,
                "failed_stop_reward": 0.0,
                "shaping_weight": 0.0,  # applied to the decrease in distance to target
            },
        )
        res["env_args"] = {}
        res["env_args"].update(self.ENV_ARGS)
        if isinstance(devices[0], int):
            res["env_args"]["x_display"] = (
                ("0.%d" % devices[process_ind % len(devices)])
                if devices is not None and len(devices) > 0
                else None
            )
        else:
            print("Got devices {}".format(devices))
        return res
