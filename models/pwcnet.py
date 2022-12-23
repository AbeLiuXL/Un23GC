import torch
import torch.nn as nn
import numpy as np
# from correlation_package.correlation import Correlation
# from spatial_correlation_sampler import spatial_correlation_sample
from torch.nn.init import xavier_uniform_, zeros_
import torch.nn.functional as F


# __all__ = [
#     'pwc_dc_net', 'pwc_dc_net_old'
#     ]
# python模块中的__all__，用于模块导入时限制，如：from module import *
# 此时被导入模块若定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入；
# 若没定义，则导入模块内的所有公有属性，方法和类。

# def conv(in_planes, out_planes, batch_norm=False, kernel_size  = 3, stride = 1, dilation = 1,padding=1):
#     if batch_norm:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
#             nn.BatchNorm2d(out_planes),
#             nn.LeakyReLU(0.1,inplace=True)
#         )
#     else:
#         return nn.Sequential(
#             nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=True),
#             nn.LeakyReLU(0.1,inplace=True)
#         )
# def correlate(input1, input2):
#     out_corr = spatial_correlation_sample(input1,
#                                           input2,
#                                           kernel_size=1,
#                                           patch_size=9,
#                                           stride=1,
#                                           padding=0,
#                                           dilation_patch=2)
#     # collate dimensions 1 and 2 in order to be treated as a
#     # regular 4D tensor
#     b, ph, pw, h, w = out_corr.size()
#     out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
#     return out_corr
def corr_naive(input1, input2, d=4):
    # naive pytorch implementation of the correlation layer.
    assert (input1.shape == input2.shape)
    batch_size, feature_num, H, W = input1.shape[0:4]
    input2 = F.pad(input2, (d, d, d, d), value=0)
    cv = []
    for i in range(2 * d + 1):
        for j in range(2 * d + 1):
            cv.append((input1 * input2[:, :, i:(i + H), j:(j + W)]).mean(1).unsqueeze(1))
    return torch.cat(cv, 1)

def fc(c_in, c_out, activation=False):
    if activation:
        return nn.Sequential(nn.Linear(c_in, c_out), nn.ReLU(inplace=True))
    else:
        return nn.Linear(c_in, c_out)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class PWCDCNet(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections
    """

    def __init__(self, md=4,is_full=True):
        """
         input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet, self).__init__()
        self.is_full = is_full
        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv6aa = conv(128, 196, kernel_size=3, stride=2)
        self.conv6a = conv(196, 196, kernel_size=3, stride=1)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)
        # self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        # self.corr = correlate
        self.corr = corr_naive

        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2  # Correlation output ch
        dd = np.cumsum([128, 128, 96, 64, 32])  # np.cumsum([1, 2, 3, 4, 5])= array([1, 3, 6, 10, 15])
        # densenet connections
        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 128 + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 96 + 4
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 64 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 32 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        # context network
        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        :param x:  [B, C, H, W] (im2)
        :param flo: [B, 2, H, W] flow
        :return: im1*
        """
        B, C, H, W = x.size()
        # mesh grid

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.to(x.device)
        vgrid = grid + flo

        # scale grid to [-1 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid,align_corners=False)
        mask = torch.ones(x.size()).to(x.device)
        mask = nn.functional.grid_sample(mask, vgrid,align_corners=False)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, im1, im2):
        # im1 = x[:,:3,:,:]
        # im2 = x[:,3:,:,:]
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)

        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)

        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)

        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)

        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)

        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)

        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        flows = [flow2, flow3, flow4, flow5, flow6]
        if self.is_full:
            flows = [torch.nn.functional.interpolate(flow * 4, scale_factor=4,
                                                     mode='bilinear', align_corners=True) for flow in flows]
            return flows
        else:
            return flows



class PWCDCNet_old(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections

    """

    def __init__(self, md=4):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet_old, self).__init__()

        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        self.conv6a = conv(128, 196, kernel_size=3, stride=2)
        self.conv6b = conv(196, 196, kernel_size=3, stride=1)

        # self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.corr = correlate
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow6 = predict_flow(od + dd[4])
        self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 128 + 4
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 96 + 4
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 64 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        od = nd + 32 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        # vgrid = torch.tensor(grid) + flo
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.ones(x.size()).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, im1, im2):
        # im1 = x[:, :3, :, :]
        # im2 = x[:, 3:, :, :]

        c11 = self.conv1b(self.conv1a(im1))
        c21 = self.conv1b(self.conv1a(im2))
        c12 = self.conv2b(self.conv2a(c11))
        c22 = self.conv2b(self.conv2a(c21))
        c13 = self.conv3b(self.conv3a(c12))
        c23 = self.conv3b(self.conv3a(c22))
        c14 = self.conv4b(self.conv4a(c13))
        c24 = self.conv4b(self.conv4a(c23))
        c15 = self.conv5b(self.conv5a(c14))
        c25 = self.conv5b(self.conv5a(c24))
        c16 = self.conv6b(self.conv6a(c15))
        c26 = self.conv6b(self.conv6a(c25))

        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((corr6, self.conv6_0(corr6)), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((x, self.conv6_2(x)), 1)
        x = torch.cat((x, self.conv6_3(x)), 1)
        x = torch.cat((x, self.conv6_4(x)), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((x, self.conv5_0(x)), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((x, self.conv5_2(x)), 1)
        x = torch.cat((x, self.conv5_3(x)), 1)
        x = torch.cat((x, self.conv5_4(x)), 1)

        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((x, self.conv4_0(x)), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((x, self.conv4_2(x)), 1)
        x = torch.cat((x, self.conv4_3(x)), 1)
        x = torch.cat((x, self.conv4_4(x)), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((x, self.conv3_0(x)), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((x, self.conv3_2(x)), 1)
        x = torch.cat((x, self.conv3_3(x)), 1)
        x = torch.cat((x, self.conv3_4(x)), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((x, self.conv2_0(x)), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((x, self.conv2_2(x)), 1)
        x = torch.cat((x, self.conv2_3(x)), 1)
        x = torch.cat((x, self.conv2_4(x)), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        if self.training:
            return flow2, flow3, flow4, flow5, flow6
        else:
            return flow2


class PWCDCNet_pose(nn.Module):
    """
    PWC-DC net. add dilation convolution and densenet connections
    """

    def __init__(self, md=4):
        """
         input: md --- maximum displacement (for correlation. default: 4), after warpping

        """
        super(PWCDCNet_pose, self).__init__()
        self.conv1a = conv(3, 16, kernel_size=3, stride=2)
        self.conv1aa = conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = conv(32, 32, kernel_size=3, stride=1)
        self.conv2b = conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = conv(64, 64, kernel_size=3, stride=1)
        self.conv4a = conv(64, 96, kernel_size=3, stride=2)
        self.conv4aa = conv(96, 96, kernel_size=3, stride=1)
        self.conv4b = conv(96, 96, kernel_size=3, stride=1)
        self.conv5a = conv(96, 128, kernel_size=3, stride=2)
        self.conv5aa = conv(128, 128, kernel_size=3, stride=1)
        self.conv5b = conv(128, 128, kernel_size=3, stride=1)
        # self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        # self.corr = correlate
        self.corr=corr_naive
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2  # Correlation output ch
        dd = np.cumsum([128, 128, 96, 64, 32])  # np.cumsum([1, 2, 3, 4, 5])= array([1, 3, 6, 10, 15])

        # densenet connections
        od = nd
        self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow5 = predict_flow(od + dd[4])
        self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        self.pose_conv5_0 = conv(od + dd[4], 128, kernel_size=3, stride=1)

        # od = nd + 128 + 4
        # self.conv5_0 = conv(od, 128, kernel_size=3, stride=1)
        # self.conv5_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        # self.conv5_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        # self.conv5_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        # self.conv5_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        # self.predict_flow5 = predict_flow(od + dd[4])
        # self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        # self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        # self.pose_conv5_0 = conv(od + dd[4], 128, kernel_size=3, stride=2)

        od = nd + 96 + 4
        self.conv4_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv4_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow4 = predict_flow(od + dd[4])
        self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        self.pose_conv4_0 = conv(od + dd[4], 128, kernel_size=3, stride=2)
        # self.pose_conv4_1 = conv(128, 128, kernel_size=3, stride=2)

        od = nd + 64 + 4
        self.conv3_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv3_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

        self.pose_conv3_0 = conv(od + dd[4], 128, kernel_size=3, stride=2)
        self.pose_conv3_1 = conv(128, 128, kernel_size=3, stride=2)
        # self.pose_conv3_2 = conv(128, 128, kernel_size=3, stride=2)

        od = nd + 32 + 4
        self.conv2_0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv2_3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv2_4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow2 = predict_flow(od + dd[4])
        self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        self.pose_conv2_0 = conv(od + dd[4], 128, kernel_size=3, stride=2)
        self.pose_conv2_1 = conv(128, 128, kernel_size=3, stride=2)
        self.pose_conv2_2 = conv(128, 128, kernel_size=3, stride=2)
        # self.pose_conv2_3 = conv(128, 128, kernel_size=3, stride=2)

        # self.pose_pred = nn.Conv2d(512, 6, kernel_size=1, padding=0)
        # self.pose_A = nn.AdaptiveAvgPool2d(1)
        self.pose_fc_1 = fc(512 * 4 * 13, 128, activation=True)
        self.pose_fc_2 = fc(128, 6, activation=False)

        # context network
        self.dc_conv1 = conv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        :param x:  [B, C, H, W] (im2)
        :param flo: [B, 2, H, W] flow
        :return: im1*
        """
        B, C, H, W = x.size()
        # mesh grid

        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1 1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.ones(x.size()).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, im1, im2):
        # im1 = x[:,:3,:,:]
        # im2 = x[:,3:,:,:]
        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        # c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        # c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        # corr6 = self.corr(c16, c26)
        # corr6 = self.leakyRELU(corr6)
        corr5 = self.corr(c15, c25)
        corr5 = self.leakyRELU(corr5)

        x = torch.cat((self.conv5_0(corr5), corr5), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        feat5 = x
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        # warp5 = self.warp(c25, up_flow6 * 0.625)
        # corr5 = self.corr(c15, warp5)
        # corr5 = self.leakyRELU(corr5)
        # x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        # x = torch.cat((self.conv5_0(x), x), 1)
        # x = torch.cat((self.conv5_1(x), x), 1)
        # x = torch.cat((self.conv5_2(x), x), 1)
        # x = torch.cat((self.conv5_3(x), x), 1)
        # x = torch.cat((self.conv5_4(x), x), 1)
        # feat5 = x
        # flow5 = self.predict_flow5(x)
        # up_flow5 = self.deconv5(flow5)
        # up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        feat4 = x
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        feat3 = x
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        feat2 = x
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        # feat6 = self.pose_conv6_0(feat6)
        feat5 = self.pose_conv5_0(feat5)
        feat4 = self.pose_conv4_0(feat4)
        feat3 = self.pose_conv3_0(feat3)
        feat3 = self.pose_conv3_1(feat3)
        feat2 = self.pose_conv2_0(feat2)
        feat2 = self.pose_conv2_1(feat2)
        feat2 = self.pose_conv2_2(feat2)
        pose = torch.cat((feat5, feat4, feat3, feat2), 1)
        pose = pose.view(pose.size(0), -1)
        pose = self.pose_fc_1(pose)
        pose = self.pose_fc_2(pose)

        # pose = self.pose_pred(pose)
        # pose = self.pose_A(pose)
        # pose = 0.01 * pose.view(pose.size(0), 6)
        # TODO:Add SE
        # print(feat2.shape,feat3.shape,feat4.shape,feat5.shape,feat6.shape,pose.shape)
        # if self.training:
        #     return flow2, flow3, flow4, flow5, flow6
        # else:
        #     return flow2
        return pose


def pwc_dc_net(path=None):
    model = PWCDCNet()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model


def pwc_dc_net_old(path=None):
    model = PWCDCNet_old()
    if path is not None:
        data = torch.load(path)
        if 'state_dict' in data.keys():
            model.load_state_dict(data['state_dict'])
        else:
            model.load_state_dict(data)
    return model


if __name__ == '__main__':
    model = PWCDCNet().cuda()
    # print(model)
    __tmp = torch.ones(4, 3, 256, 832).cuda()
    __tmp2 = torch.zeros(4, 3, 256, 832).cuda()
    # model.eval()
    out = model(__tmp, __tmp2)
    print(len(out))
    for i  in range(len(out)):
        print(out[i].shape)