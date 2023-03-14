import torch
from torch.utils.data import DataLoader
from os import makedirs
from os.path import join, isdir
from dataloader import QSMDataset
from models import ResolutionQSM

dataset_train = QSMDataset('train')
# dataset_test = QSMDataset('test')

dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
# dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
# dataloader_test.flist_P = dataset_test.flist

model = ResolutionQSM()

model.train(dataloader_train)