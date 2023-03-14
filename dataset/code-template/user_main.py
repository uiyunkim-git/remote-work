from tulon_network_module_A import pytorch_network as NETWORK_A  # template - read only
from tulon_loss_function_A import pytorch_loss as LOSS_A  # template - read only

# User Define Code Starts Here -----
from scipy import io as sio
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
from torch import optim


class ct_low_high_dataset(Dataset):
    def __init__(self, dir_dataset):
        self.input_paths = [dir_dataset + "/" + x for x in os.listdir(dir_dataset)]

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        input_, label_ = self.__preprocess(input_path)
        return {"x": input_, "y": label_}

    def __preprocess(self, input_path):
        input_ = sio.loadmat(input_path)["imdb"]["low"][0][0]
        input_ = Image.fromarray(self.__rescale(input_))

        label_ = sio.loadmat(input_path)["imdb"]["high"][0][0]
        label_ = Image.fromarray(self.__rescale(label_))

        crop = transforms.RandomCrop(size=128)
        tf = transforms.Compose([crop, transforms.ToTensor()])

        seed = np.random.randint(1234)  # make a seed with numpy generator
        random.seed(seed)
        torch.manual_seed(seed)

        input_ = tf(input_)
        label_ = tf(label_)
        return input_, label_

    def __rescale(self, ct):
        ct[ct < -1024.0] = -1024.0
        ct /= 4000
        return ct


full_dataset = ct_low_high_dataset("/datasets/ct_denoising/train")

validation_size = 0.1
train_size = int(1 - validation_size * len(full_dataset))
validation_size = len(full_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, validation_size]
)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)

network = NETWORK_A()
loss_fn = LOSS_A()
network.save()
optimizer = optim.Adam(network.parameters(), lr=0.0001)


for n_epoch in range(3):
    network.train()
    loss_fn.train()  # set loss tag as TRAIN so logger

    for n_batch, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        x = batch["x"]
        y = batch["y"]

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        print(n_batch)
        net_y = network(x)
        loss = loss_fn(net_y, y)

        loss.backward()  # This automatically logs f
        optimizer.step()

        loss_fn.n_batch_increment()
        network.n_batch_increment()

    loss_fn.n_epoch_increment()
    network.n_epoch_increment()

    network.eval()
    loss_fn.eval()

    for n_batch, batch in enumerate(validation_dataloader):
        x = batch["x"]
        y = batch["y"]

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        net_y = network(x)
        loss = loss_fn(net_y, y)
    network.save()

