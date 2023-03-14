import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from coda_config import weight_saving_dir
import io
import requests


class pytorch_network:
    def __init__(self, **kwargs):
        # block user access to the pytorch network itself
        # network defined as private variable,
        # so w/b parameters cannot be manually manipulated by user.
        self.__XRALEFKDO = XRALEFKDO(**kwargs)
        self.n_epoch = 1
        self.n_batch = 1

        if torch.cuda.is_available():
            self.device = "cuda"
            self.__XRALEFKDO.to("cuda")
        else:
            self.device = "cpu"

    def __call__(self, *args):  # overriding forward method
        # foward pass
        return self.__XRALEFKDO(*args)

    def parameters(self):
        return self.__XRALEFKDO.parameters()

    def save(self):
        buffer = io.BytesIO()
        # only files in XRALEFKDO directory will sent back to the platform
        torch.save(self.__XRALEFKDO.state_dict(), buffer)
        buffer.seek(0)
        host_message_receiver_url = f"http://{os.environ.get('RESULT_BRIDGE_IP')}/training/{os.environ.get('TRAINING_ID')}/result/XRALEFKDO_EPOCH_{self.n_epoch}"

        requests.post(
            host_message_receiver_url,
            files={"file": buffer.read()},
        )

    def load(self, path_weight):
        self.__XRALEFKDO.load_state_dict(
            torch.load(path_weight, map_location=self.device)
        )

    def train(self):
        self.__XRALEFKDO.train()

    def eval(self):
        self.__XRALEFKDO.eval()

    def n_epoch_increment(self):
        self.n_epoch += 1

    def n_batch_increment(self):
        self.n_batch += 1


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class XRALEFKDO(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9):
        super(XRALEFKDO, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, batch_inputs):
        return self.model(batch_inputs)
