import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet
from pytorchcv.model_provider import get_model


class Head(nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(in_f, 512)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(512, out_f)
        self.b1 = nn.BatchNorm1d(in_f)
        self.b2 = nn.BatchNorm1d(512)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)

        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)

        out = self.o(x)
        return out


class FCN(nn.Module):
    def __init__(self, base, in_f, out_f):
        super(FCN, self).__init__()
        self.base = base
        self.h1 = Head(in_f, out_f)

    def forward(self, x):
        x = self.base(x)
        return self.h1(x)


class BaseFCN(nn.Module):
    def __init__(self, n_classes: int):
        super(BaseFCN, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(625, 256)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.f(x)
        x = self.l(x)
        x = self.d(x)
        out = self.o(x)
        return out

    def get_trainable_parameters_cooccur(self):
        return self.parameters()


class BaseFCNHigh(nn.Module):
    def __init__(self, n_classes: int):
        super(BaseFCNHigh, self).__init__()

        self.f = nn.Flatten()
        self.l = nn.Linear(625, 512)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.f(x)
        x = self.l(x)
        x = self.d(x)
        out = self.o(x)
        return out

    def get_trainable_parameters_cooccur(self):
        return self.parameters()


class BaseFCN4(nn.Module):
    def __init__(self, n_classes: int):
        super(BaseFCN4, self).__init__()

        self.f = nn.Flatten()
        self.l1 = nn.Linear(625, 512)
        self.l2 = nn.Linear(512, 384)
        self.l3 = nn.Linear(384, 256)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.f(x)
        x = self.l1(x)
        x = self.d(x)
        x = self.l2(x)
        x = self.d(x)
        x = self.l3(x)
        x = self.d(x)
        out = self.o(x)
        return out

    def get_trainable_parameters_cooccur(self):
        return self.parameters()


class BaseFCNBnR(nn.Module):
    def __init__(self, n_classes: int):
        super(BaseFCNBnR, self).__init__()

        self.f = nn.Flatten()
        self.b1 = nn.BatchNorm1d(625)
        self.b2 = nn.BatchNorm1d(256)
        self.l = nn.Linear(625, 256)
        self.d = nn.Dropout(0.5)
        self.o = nn.Linear(256, n_classes)
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.f(x)
        x = self.b1(x)
        x = self.d(x)
        x = self.l(x)
        x = self.r(x)
        x = self.b2(x)
        x = self.d(x)
        out = self.o(x)
        return out

    def get_trainable_parameters_cooccur(self):
        return self.parameters()


def forward_resnet_conv(net, x, upto: int = 4):
    """
    Forward ResNet only in its convolutional part
    :param net:
    :param x:
    :param upto:
    :return:
    """
    x = net.conv1(x)  # N / 2
    x = net.bn1(x)
    x = net.relu(x)
    x = net.maxpool(x)  # N / 4

    if upto >= 1:
        x = net.layer1(x)  # N / 4
    if upto >= 2:
        x = net.layer2(x)  # N / 8
    if upto >= 3:
        x = net.layer3(x)  # N / 16
    if upto >= 4:
        x = net.layer4(x)  # N / 32
    return x


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class FeatureExtractorGray(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.479], std=[0.226])


class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str, n_classes: int, pretrained: bool):
        super(EfficientNetGen, self).__init__()

        if pretrained:
            self.efficientnet = EfficientNet.from_pretrained(model)
        else:
            self.efficientnet = EfficientNet.from_name(model)
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, n_classes)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.efficientnet._dropout(x)
        x = self.classifier(x)
        # x = F.softmax(x, dim=-1)
        return x


class EfficientNetB0(EfficientNetGen):
    def __init__(self, n_classes: int, pretrained: bool):
        super(EfficientNetB0, self).__init__(model='efficientnet-b0', n_classes=n_classes, pretrained=pretrained)


class EfficientNetB4(EfficientNetGen):
    def __init__(self, n_classes: int, pretrained: bool):
        super(EfficientNetB4, self).__init__(model='efficientnet-b4', n_classes=n_classes, pretrained=pretrained)


class ResNet50(FeatureExtractor):
    def __init__(self, n_classes: int, pretrained: bool):
        super(ResNet50, self).__init__()
        self.resnet = resnet.resnet50(pretrained=pretrained)
        self.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)
        del self.resnet.fc

    def features(self, x):
        x = forward_resnet_conv(self.resnet, x)
        x = self.resnet.avgpool(x).flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


"""
Xception from Kaggle
"""


class XceptionWeiHao(FeatureExtractor):

    def __init__(self, n_classes: int, pretrained: bool):
        super(XceptionWeiHao, self).__init__()

        self.model = get_model("xception", pretrained=pretrained)
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove original output layer
        self.model[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)))
        self.model = FCN(self.model, 2048, n_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.base(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.model.h1(x)



