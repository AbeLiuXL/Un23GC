from __future__ import division
import shutil
import numpy as np
import torch
import torch.nn as nn
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0,1,low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0,max_value,resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:,i]) for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


# def tensor2array(tensor, max_value=None, colormap='rainbow'):
#
#     tensor = tensor.detach().cpu()
#     if max_value is None:
#         max_value = tensor.max().item()
#     if tensor.ndimension() == 2 or tensor.size(0) == 1:
#         norm_array = tensor.squeeze().numpy()/max_value
#         array = COLORMAPS[colormap](norm_array).astype(np.float32)
#         array = array.transpose(2, 0, 1)
#
#     elif tensor.ndimension() == 3:
#         assert(tensor.size(0) == 3)
#         array = 0.5 + tensor.numpy()*0.5
#     return array
def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array[:,:,:3]
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        if (tensor.size(0) == 3):
            array = 0.5 + tensor.numpy()*0.5
        elif (tensor.size(0) == 2):
            array = tensor.numpy()

    return array

def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))

def save_checkpoint_DPF(save_path, dispnet_state, exp_pose_state,flownet_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose','flownet']
    states = [dispnet_state, exp_pose_state,flownet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))
def save_checkpoint_DP(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))
    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))
def save_checkpoint_F(save_path,flownet_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['flownet']
    states = [flownet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix,filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix,filename), save_path/'{}_model_best.pth.tar'.format(prefix))
class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, targets, *inputs):
        outputs = self.model(*inputs)
        loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss, 0), outputs


def DataParallel_withLoss(model, loss, **kwargs):
    model = FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids = kwargs['device_ids']
    else:
        device_ids = None
    if 'output_device' in kwargs.keys():
        output_device = kwargs['output_device']
    else:
        output_device = None
    if 'cuda' in kwargs.keys():
        cudaID = kwargs['cuda']
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
    return model