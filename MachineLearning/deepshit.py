import os
import platform
import shutil
import warnings
from typing import Optional, List, Callable, Tuple

import cv2
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as func
from torchvision.datasets import ImageFolder
from torchvision.models import InceptionOutputs
from torchvision.models.inception import BasicConv2d, InceptionB, InceptionD, \
                    InceptionAux, InceptionA, InceptionC, InceptionE
from torchvision.transforms import transforms
import smtplib
import ssl


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 7, (5, 5))
        self.conv3 = nn.Conv2d(7, 10, (5, 5))
        self.conv4 = nn.Conv2d(10, 10, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(40, 12)
        self.fc2 = nn.Linear(12, 5)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv4(x)))
        x = self.pool(func.relu(self.conv4(x)))
        x = self.pool(func.relu(self.conv4(x))).view(-1, 40)
        x = self.fc2(func.relu((self.fc1(x))))
        return x


class ConvNetNew(nn.Module):
    def __init__(self):
        super(ConvNetNew, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (10, 10))
        self.conv2 = nn.Conv2d(4, 8, (10, 10))
        self.conv3 = nn.Conv2d(8, 8, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x))).view(-1, 512)
        x = self.fc2(func.relu((self.fc1(x))))
        return x


class ConvNet3(nn.Module):
    def __init__(self):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, (5, 5))
        self.conv2 = nn.Conv2d(4, 8, (5, 5))
        self.conv3 = nn.Conv2d(8, 8, (5, 5))
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.LazyLinear(32, 4)

    def forward(self, x):
        x = self.pool(func.relu(self.conv1(x)))
        x = self.pool(func.relu(self.conv2(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv3(x))).view(-1, 32)
        # print(x.shape)
        x = self.fc1(x)
        return x

class ProConvNet(nn.Module):
    def __init__(self):
        super(ProConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 8, 3)
        self.conv3 = nn.Conv2d(8, 16, 2)
        self.conv4 = nn.Conv2d(16, 16, 2)
        
        self.fc1 = nn.LazyLinear(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        
        x = func.max_pool2d(func.relu(self.conv1(x)), 2)
        x = func.max_pool2d(func.relu(self.conv2(x)), 2)
        x = func.max_pool2d(func.relu(self.conv3(x)), 2)
        x = func.max_pool2d(func.relu(self.conv4(x)), 2)
        x = func.max_pool2d(func.relu(self.conv4(x)), 2)
        
        
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc2(x))
        x = func.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.classifier(x)
        return x


class AlexMulti(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6 + 1, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, 40),
            nn.ReLU(inplace=True),
            nn.Linear(40, num_classes)
        )


    def forward(self, x: torch.Tensor, chi: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        chi = chi.view(chi.shape[0],1)
        x = torch.cat((x, chi), 1)
        x = self.fc(x)
        return x


def load_model(model):
    ans = ""
    while True:
        while ans not in ("y", "Y", "N", "n"):
            ans = input("Load previous model? (y/n): ")
        if ans == "n" or ans == "N":
            break
        if ans == "y" or ans == "Y":
            try:
                path = "Models/" + input("enter path: ")
                model.load_state_dict(torch.load(path))
                break
            except FileNotFoundError as e:
                print(e)
                ans = ""
            except IsADirectoryError:
                print("You must enter a file name goddamnit!")
                ans = ""



def test_network(loader, model_, criterion_, device, print_results=False):
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for images, params in loader:
            images = images.to(device)
            params = params.to(device)
            output = model_(images)
            loss_ = criterion_(output, params)
            total_loss += loss_
            n_batches += 1
            if print_results:
                for i, param in enumerate(params):
                    niceoutput = [round(n, 4) for n in output[i].tolist()]
                    niceparam = [round(n, 4) for n in param.tolist()]
                    print(f"{f'Correct: {niceparam}' : <50} {f'Output: {niceoutput}' : ^50}")
        return total_loss / n_batches


def print_images(images, params):
    for i, img in enumerate(images):
        image = images[i].numpy().reshape(images[i].shape[1], images[i].shape[2], 1)
        image = cv2.resize(image, (400, 400))
        image = cv2.putText(image, f' dist: {str(params[i][0].item())} ein: {str(params[i][1].item())} sigma: {str(params[i][2].item())} x: {str(params[i][3].item())} y: {str(params[i][4].item())}', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("img", image)
        cv2.waitKey(0)


class Inception3(nn.Module):
    def __init__(
            self,
            num_outputs: int = 5,
            aux_logits: bool = True,
            transform_input: bool = False,
            inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
            init_weights: Optional[bool] = None,
            dropout: float = 0.5,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of inception_v3 will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(inception_blocks) != 7:
            raise ValueError(f"lenght of inception_blocks should be 7 instead of {len(inception_blocks)}")
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(1, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_outputs)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(2048, num_outputs)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        # x = self._transform_input(x)
        x, aux = self._forward(x)
        return x
        # aux_defined = self.training and self.aux_logits
        # if torch.jit.is_scripting():
        #     if not aux_defined:
        #         warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
        #     return InceptionOutputs(x, aux)
        # else:
        #     return self.eager_outputs(x, aux)

