import numpy as np
import torch
from torch.nn import init
from torch.optim import lr_scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def fft3(data):
    data = torch.fft.fftshift(torch.fft.fftn(data))
    return data


def ifft3(data):
    data = torch.fft.ifftn(torch.fft.ifftshift(data))
    return data


# Function for patch-wise inference
def get_patch_weight(size, stride):
    patch_weight_y = np.ones(size[0])
    overlap_size_y = size[0] - stride
    patch_weight_y[:overlap_size_y] = patch_weight_y[:overlap_size_y] * (np.cos(np.linspace(-np.pi, 0, overlap_size_y)) + 1) / 2
    patch_weight_y[-overlap_size_y:] = patch_weight_y[-overlap_size_y:] * (np.cos(np.linspace(0, np.pi, overlap_size_y)) + 1) / 2
    patch_weight_y = np.tile(patch_weight_y[:, np.newaxis, np.newaxis], [1, size[0], size[0]])

    patch_weight_x = np.ones(size[1])
    overlap_size_x = size[1] - stride
    patch_weight_x[:overlap_size_x] = patch_weight_x[:overlap_size_x] * (np.cos(np.linspace(-np.pi, 0, overlap_size_x)) + 1) / 2
    patch_weight_x[-overlap_size_x:] = patch_weight_x[-overlap_size_x:] * (np.cos(np.linspace(0, np.pi, overlap_size_x)) + 1) / 2
    patch_weight_x = np.tile(patch_weight_x[np.newaxis, :, np.newaxis], [size[1], 1, size[1]])

    patch_weight_z = np.ones(size[2])
    overlap_size_z = size[2] - stride
    patch_weight_z[:overlap_size_z] = patch_weight_z[:overlap_size_z] * (np.cos(np.linspace(-np.pi, 0, overlap_size_z)) + 1) / 2
    patch_weight_z[-overlap_size_z:] = patch_weight_z[-overlap_size_z:] * (np.cos(np.linspace(0, np.pi, overlap_size_z)) + 1) / 2
    patch_weight_z = np.tile(patch_weight_z[np.newaxis, np.newaxis, :], [size[2], size[2], 1])

    patch_weight = patch_weight_y * patch_weight_x * patch_weight_z
    return patch_weight
