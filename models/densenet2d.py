import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict

from typing import Type, Any, Callable, Union, List, Optional, Dict

import torch
import re
import torch.nn as nn
import torch.nn.functional as F
#from utils import WeightsEnum, Weights
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url  # noqa: 401
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url  # noqa: 401

__all__ = [
    "DenseNet",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]

model_urls = {
    "densenet121":"https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet161":"https://download.pytorch.org/models/densenet161-8d451a50.pth",
    "densenet169":"https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "densenet201":"https://download.pytorch.org/models/densenet201-c1103571.pth"
}

def _log_api_usage_once(obj: Any) -> None:

    """
    Logs API usage(module and name) within an organization.
    In a large ecosystem, it's often useful to track the PyTorch and
    TorchVision APIs usage. This API provides the similar functionality to the
    logging module in the Python stdlib. It can be used for debugging purpose
    to log which methods are used and by default it is inactive, unless the user
    manually subscribes a logger via the `SetAPIUsageLogger method <https://github.com/pytorch/pytorch/blob/eb3b9fe719b21fae13c7a7cf3253f970290a573e/c10/util/Logging.cpp#L114>`_.
    Please note it is triggered only once for the same API call within a process.
    It does not collect any data from open-source users since it is no-op by default.
    For more information, please refer to
    * PyTorch note: https://pytorch.org/docs/stable/notes/large_scale_deployments.html#api-usage-logging;
    * Logging policy: https://github.com/pytorch/vision/issues/5052;
    Args:
        obj (class instance or method): an object to extract info from.
    """
    if not obj.__module__.startswith("torchvision"):
        return
    name = obj.__class__.__name__
    if isinstance(obj, FunctionType):
        name = obj.__name__
    torch._C._log_api_usage_once(f"{obj.__module__}.{name}")



def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self,
                 num_input_features,
                 growth_rate,
                 bn_size,
                 drop_rate,
                 memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.Module):
    def __init__(self,
                 num_layers,
                 num_input_features,
                 bn_size,
                 growth_rate,
                 drop_rate,
                 memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""2D-DenseNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes (if 'classifier' mode)
        in_channels (int) - number of input channels (1 for sMRI)
        mode (str) - specify in which mode DenseNet is trained on -- must be "encoder" or "classifier"
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self,
                 growth_rate=32,
                 block_config=(3, 12, 24, 16),
                 num_init_features=64,
                 bn_size=4,
                 drop_rate=0,
                 num_classes=1000,
                 #in_channels=1,    #in_channels=2 pour avoir le mask du foie
                 mode="encoder",
                 z_dim=128,
                 pretrained=False,
                 memory_efficient=False):

        super(DenseNet, self).__init__()
        _log_api_usage_once(self)

        if pretrained:
            in_channels = 3
        else:
            in_channels = 1

        assert mode in {'encoder', 'classifier'}, "Unknown mode selected: %s"%mode

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ('conv0', nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ('norm0', nn.BatchNorm2d(num_init_features)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )                                                                                # première convolution : conv+BN+relu+maxpool
        self.mode = mode
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):                                      # boucle pour construire le modèle : ajout de DenseBlocks (BN+relu+conv+BN+relu+conv)
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)                      # ajout de chaque DenseBlock
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features

        ### OUR PART ###
        if self.mode == "classifier":
            # Final batch norm
            self.features.add_module('last_bn', nn.BatchNorm2d(num_features))
            # Linear layer
            self.classifier_ = nn.Linear(num_features, num_classes)
        elif self.mode == "encoder":
            self.hidden_representation = nn.Linear(num_features, 512)
            self.head_projection = nn.Linear(512, z_dim)
        ### END ###

        # Init. with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode

        ## Eventually keep the input images for visualization
        self.input_imgs = x.detach().cpu().numpy()
        features = self.features(x)                                                     # forward pass une image dans DenseNet
        #if self.mode == "classifier":                                                  # Classifier Mode : à la fin de DenseNet pré-entraîné,
        if mode == "classifier":
            out = F.relu(features, inplace=True)                                        # ajout de ReLu+Pool+flatten+BN+LinearLayer
            out = F.adaptive_avg_pool2d(out, 1)
            out = torch.flatten(out, 1)
            out = self.classifier_(out)                                                  # output : un vecteur de taille num_classes
        elif mode == "encoder":                                                         # Encoder Mode : à la fin de DenseNet,
            out = F.relu(features, inplace=True)                                        # Relu+Pool+flatten+hidden_repr+head_predictor
            out = F.adaptive_avg_pool2d(out, 1)
            out = torch.flatten(out, 1)
            out = self.hidden_representation(out)
            out = F.relu(out, inplace=True)
            out = self.head_projection(out)                                             # output : un vecteur de taille 128 ==> vecteur de représentation de l'image en input
        elif mode == "representation":
            out = F.relu(features, inplace=True)                                        # Relu+Pool+flatten+hidden_repr+head_predictor
            out = F.adaptive_avg_pool2d(out, 1)
            out = torch.flatten(out, 1)

        return out.squeeze(dim=1)

    def get_current_visuals(self):
        return self.input_imgs



def _densenet(arch,
      growth_rate,
      block_config,
      num_init_features,
      pretrained,
      progress,
      **kwargs
) -> DenseNet:

    model = DenseNet(growth_rate, block_config, num_init_features, pretrained=pretrained, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)

    return model


def densenet121(pretrained: bool = False, progress: bool = True, **kwargs: Any):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs)

def densenet161(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    return _densenet('densenet161', 48, (6, 12, 36, 24), 96, pretrained, progress, **kwargs)


def densenet169(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet('densenet169', 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs)


def densenet201(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    return _densenet('densenet201', 32, (6, 12, 48, 32), 64, pretrained, progress, **kwargs)

