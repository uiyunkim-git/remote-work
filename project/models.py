import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import makedirs
from os.path import join, isdir, isfile
from scipy import io as sio
from math import ceil, floor
from networks import Unet_AdaIN, forward_model
from losses import GRADLoss, TVLoss
from utils import init_net, get_patch_weight
from tqdm import tqdm


class ResolutionQSM:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.nY = 64
        self.nX = 64
        self.nZ = 64
        self.lambda_cycle = 1
        self.lambda_grad = 0.05
        self.lambda_tv = 0.01
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.n_epochs = 100
        self.n_epochs_decay = 0
        self.continue_epoch = 0
        self.test_epoch = 100
        self.lr = 1e-4
        self.init_type = 'xavier'
        self.init_gain = 1
        self.save_epoch = 10
        self.save_path = '../results'
        self.experiment_name = 'QSM_test'
        self.stride = 32
        self.ckpt_dir = join(self.save_path, self.experiment_name, 'ckpt_dir')
        if not isdir(self.ckpt_dir):
            makedirs(self.ckpt_dir)

        self.cycle_loss = nn.L1Loss()
        self.grad_loss = GRADLoss()
        
        self.net = Unet_AdaIN(1, 1, 32).to(self.device)
        self.forward_model = forward_model()

        self.optim = torch.optim.Adam(self.net.parameters(), self.lr, betas=(self.beta1, self.beta2))

    def train(self, dataloader):
        if isfile(join(self.ckpt_dir, str(self.continue_epoch) + '.pth')):
            checkpoint = torch.load(join(self.ckpt_dir, str(self.continue_epoch) + '.pth'))
            self.net.load_state_dict(checkpoint['state_dict'])
            self.optim.load_state_dict(checkpoint['optim_state_dict'])
            trained_epoch = checkpoint['epoch']
            print('Start from saved model - ' + str(trained_epoch))
        else:
            init_net(self.net, self.init_type, self.init_gain)
            trained_epoch = 0
            print('Start initially')
        
        dataset_len = len(dataloader)

        for epoch in tqdm(range(trained_epoch, self.n_epochs + self.n_epochs_decay), desc='Epoch', total=self.n_epochs + self.n_epochs_decay, initial=trained_epoch):
            for step, (phase, DK, mask, voxel) in enumerate(tqdm(dataloader)):
                phase = phase.to(self.device)
                DK = DK.to(self.device)
                mask = mask.to(self.device)
                voxel = voxel.to(self.device)

                QSM = self.net(phase, voxel) * mask
                phase_recon = self.forward_model(QSM, DK) * mask

                cycle_loss = self.cycle_loss(phase_recon, phase)
                grad_loss = self.grad_loss(phase_recon, phase)
                tv_loss = TVLoss(QSM)
                total_loss = self.lambda_cycle * cycle_loss + self.lambda_grad * grad_loss + self.lambda_tv * tv_loss

                self.optim.zero_grad()
                total_loss.backward()
                self.optim.step()

            if (epoch + 1) % self.save_epoch == 0:
                torch.save({'epoch': epoch + 1, 'state_dict': self.net.state_dict(),
                            'optim_state_dict': self.optim.state_dict()},
                           join(self.ckpt_dir, '{}'.format(epoch + 1) + '.pth'))
        
        def test(self, dataloader):
            with torch.no_grad():
                checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
                self.net.load_state_dict(checkpoint['state_dict'])
                self.net.eval()

                save_path = join(self.save_path, self.experiment_name, 'test_{}_{}'.format(self.test_epoch, self.stride))
                if not isdir(save_path):
                    makedirs(save_path)

                print('Start test.')
                for step, [phase, _, mask, voxel] in enumerate(tqdm(dataloader)):
                    overlap_y = self.nY - self.stride
                    overlap_x = self.nX - self.stride
                    overlap_z = self.nZ - self.stride

                    pad_y = (ceil((phase.size(2) + overlap_y - self.nY) / self.stride) * self.stride + self.nY) - (phase.size(2) + overlap_y)
                    pad_x = (ceil((phase.size(3) + overlap_x - self.nX) / self.stride) * self.stride + self.nX) - (phase.size(3) + overlap_x)
                    pad_z = (ceil((phase.size(4) + overlap_z - self.nZ) / self.stride) * self.stride + self.nZ) - (phase.size(4) + overlap_z)

                    pad_y1 = overlap_y + floor(pad_y / 2)
                    pad_y2 = overlap_y + (pad_y - floor(pad_y / 2))
                    pad_x1 = overlap_x + floor(pad_x / 2)
                    pad_x2 = overlap_x + (pad_x - floor(pad_x / 2))
                    pad_z1 = overlap_z + floor(pad_z / 2)
                    pad_z2 = overlap_z + (pad_z - floor(pad_z / 2))

                    phase_pad = F.pad(phase, [pad_z1, pad_z2, pad_x1, pad_x2, pad_y1, pad_y2], mode='constant')
                    mask_pad = F.pad(mask, [pad_z1, pad_z2, pad_x1, pad_x2, pad_y1, pad_y2], mode='constant')
                    voxel = voxel.to(self.device)
                    tmp_output = np.zeros((phase_pad.size(2), phase_pad.size(3), phase_pad.size(4)), dtype=np.float32)

                    # Patch-wise inference (reduce patch artifact)
                    patch_weight = get_patch_weight((self.nY, self.nX, self.nZ), self.stride)
                    for i in range(ceil((phase_pad.size(2) - self.nY) / self.stride) + 1):
                        for j in range(ceil((phase_pad.size(3) - self.nX) / self.stride) + 1):
                            for k in range(ceil((phase_pad.size(4) - self.nZ) / self.stride) + 1):
                                ys = self.stride * i
                                xs = self.stride * j
                                zs = self.stride * k

                                ys = min(ys, phase_pad.size(2) - self.nY)
                                xs = min(xs, phase_pad.size(3) - self.nX)
                                zs = min(zs, phase_pad.size(4) - self.nZ)

                                patch = phase_pad[:, :, ys:ys + self.nY, xs:xs + self.nX, zs:zs + self.nZ].to(self.device)
                                patch_mask = mask_pad[:, :, ys:ys + self.nY, xs:xs + self.nX, zs:zs + self.nZ].to(self.device)

                                patch_output = np.squeeze((self.net(patch, voxel) * patch_mask).to('cpu:0').detach().numpy())
                                tmp_output[ys:ys + self.nY, xs:xs + self.nX, zs:zs + self.nZ] += patch_output * patch_weight

                    QSM = tmp_output[pad_y1:-pad_y2, pad_x1:-pad_x2, pad_z1:-pad_z2]

                    # subpath = dataloader.flist_P[step].split('test/')[1]
                    # subname = subpath.split('/')[0]
                    # fname = subpath.split('/')[1]

                    # test_output = {'QSM': QSM}

                    # sub_save_path = join(save_path, subname)
                    # if not isdir(sub_save_path):
                    #     makedirs(sub_save_path)

                    # sio.savemat(join(sub_save_path, fname), test_output)
