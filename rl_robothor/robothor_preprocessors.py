from typing import Dict, Any, Callable

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import gym

from rl_base.preprocessor import Preprocessor


class ResNetEmbedder(nn.Module):
    """Wrapper to freeze torchvision ResNet module. By default it skips the last average pooling layer."""

    def __init__(self, resnet, pool=True):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            if not self.pool:
                return x
            else:
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
                return x


class ResnetPreProcessorThor(Preprocessor):
    """Preprocess RGB image using a frozen torchvision ResNet model. By default it won't use the last pooling layer."""

    def __init__(self, config: Dict[str, Any], *args: Any, **kwargs: Any):
        def f(x: Dict[str, Any], k: str) -> Any:
            assert k in x, "{} must be set in ResnetPreProcessorThor".format(k)
            return x[k]

        def optf(x: Dict[str, Any], k: str, default: Any) -> Dict[str, Any]:
            return x[k] if k in x else default

        self.input_height: int = f(config, "input_height")
        self.input_width: int = f(config, "input_width")
        self.output_height: int = f(config, "output_height")
        self.output_width: int = f(config, "output_width")
        self.output_dims: int = f(config, "output_dims")
        self.make_model: Callable[..., models.ResNet] = optf(
            config, "torchvision_resnet_model", models.resnet18
        )
        self.pool: bool = optf(config, "pool", False)
        self.device: torch.device = optf(config, "device", "cpu")

        self.resnet = ResNetEmbedder(
            self.make_model(pretrained=True).to(self.device), pool=self.pool
        )

        low = -np.inf
        high = np.inf
        shape = (self.output_dims, self.output_height, self.output_width)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape)

        assert (
            len(config["input_uuids"]) == 1
        ), "resnet preprocessor can only consume one observation type"

        f(config, "output_uuid")

        super().__init__(config, *args, **kwargs)

    def to(self, device: torch.device) -> "ResnetPreProcessorThor":
        """Transfer ResNet to given device."""
        self.resnet = self.resnet.to(device)
        self.device = device
        return self

    def process(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        """Obtain preprocessed output from observations."""
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)
        return self.resnet(x.to(self.device))
