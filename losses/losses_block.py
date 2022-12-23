import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2

def EPE(x, y):
    return torch.norm(x - y, 2, 1, keepdim=True)

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self,md=3):
        super(SSIM, self).__init__()
        patch_size = 2 * md + 1
        self.mu_x_pool = nn.AvgPool2d(patch_size, 1,padding=0)
        self.mu_y_pool = nn.AvgPool2d(patch_size, 1,padding=0)
        self.sig_x_pool = nn.AvgPool2d(patch_size, 1,padding=0)
        self.sig_y_pool = nn.AvgPool2d(patch_size, 1,padding=0)
        self.sig_xy_pool = nn.AvgPool2d(patch_size, 1,padding=0)

        self.refl = nn.ReflectionPad2d(md)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# def disp_smooth_loss(img, disp):
#     # img: [b,3,h,w] depth: [b,1,h,w]
#     """Computes the smoothness loss for a disparity image
#     The color image is used for edge-aware smoothness
#     """
#     grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
#     grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
#
#     grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
#     grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
#
#     grad_disp_x *= torch.exp(-grad_img_x)
#     grad_disp_y *= torch.exp(-grad_img_y)
#
#     return grad_disp_x.mean() + grad_disp_y.mean()

def disp_smooth_loss(disp, img):
    """
    Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """

    # normalize
    mean_disp = disp.mean(2, True).mean(3, True)
    norm_disp = disp / (mean_disp + 1e-7)
    disp = norm_disp

    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def charbonnier(dist):
    return (dist ** 2 + 1e-6) ** 0.5 - 1e-3


def smooth_grad_1st(flo, image, alpha, mask, do_charb):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)

    loss_x = weights_x * dx.abs() / 2.
    loss_y = weights_y * dy.abs() / 2

    if mask is not None:
        loss_x *= mask[:, :, :, 1:]
        loss_y *= mask[:, :, 1:, :]

    if do_charb:
        return charbonnier(loss_x).mean() / 2. + charbonnier(loss_y).mean() / 2.
    else:
        return loss_x.mean() / 2. + loss_y.mean() / 2.


def smooth_grad_2nd(flo, image, alpha=1, mask=None, do_charb=False):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    if mask is not None:
        loss_x *= mask[:, :, :, 1:-1]
        loss_y *= mask[:, :, 1:-1, :]

    if do_charb:
        return charbonnier(loss_x).mean() / 2. + charbonnier(loss_y).mean() / 2.
    else:
        return loss_x.mean() / 2. + loss_y.mean() / 2.

def gaussianblur_pt(x_batch, kernel_sz, sigma):
    B = x_batch.size(0)
    dtype = x_batch.type()
    x_batch = np.split(x_batch.detach().cpu().numpy(), B, axis=0)
    x_out = []
    for x in x_batch:
        x_out.append(cv2.GaussianBlur(x[0][0], kernel_sz, sigma))
    x_out = torch.tensor(np.stack(x_out)).type(dtype).unsqueeze(1)
    return x_out

def percentile_pt(x_batch, th=85):
    B = x_batch.size(0)
    dtype = x_batch.type()
    x_batch = np.split(x_batch.detach().cpu().numpy(), B, axis=0)
    x_out = []
    for x in x_batch:
        x_out.append(np.percentile(x, th))
    x_out = torch.tensor(x_out).type(dtype)
    return x_out


def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 100:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(diff.device)
    return mean_value

def check_sizes(input,input_name,expected):
    condition = [input.ndimension() == len(expected)] # ndimension =  len of dim
    for i,size in enumerate(expected):
        if size.isdigit():
            # Python isdigit() 判断数字维度对应
            condition.append(input.size(i)==int(size))
            assert (all(condition)),"wrong size for {}, expected {}, got {}".format(input_name,'x'.join(expected),list(input.size()))
def edge_aware_smoothness_loss(pred_disp, img, max_scales):
    def gradient_x(img):
      gx = img[:,:,:-1,:] - img[:,:,1:,:]
      return gx

    def gradient_y(img):
      gy = img[:,:,:,:-1] - img[:,:,:,1:]
      return gy

    def get_edge_smoothness(img, pred):
      pred_gradients_x = gradient_x(pred)
      pred_gradients_y = gradient_y(pred)

      image_gradients_x = gradient_x(img)
      image_gradients_y = gradient_y(img)

      weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
      weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

      smoothness_x = torch.abs(pred_gradients_x) * weights_x
      smoothness_y = torch.abs(pred_gradients_y) * weights_y
      return torch.mean(smoothness_x) + torch.mean(smoothness_y)

    loss = 0
    weight = 1.
    if type(pred_disp) not in [tuple, list]:
        pred_disp = [pred_disp]
    s = 0
    for scaled_disp in pred_disp:
        s += 1
        if s > max_scales:
            break

        b, _, h, w = scaled_disp.size()
        scaled_img = nn.functional.adaptive_avg_pool2d(img, (h, w))
        loss += get_edge_smoothness(scaled_img, scaled_disp) * weight
        weight /= 2.3 # 2sqrt(2) 4

    return loss
def inverse_poses_loss(pose, pose_inv):
        b, _,_ = pose.size()
        b4 = torch.tensor([0, 0, 0, 1]).expand(b, 1, 4).type_as(pose).to(pose.device)
        pose_RT = torch.cat((pose, b4), dim=-2)
        pose_inv_RT = torch.cat((pose_inv, b4), dim=-2)
        # E = torch.eye(4).expand(b, 4, 4).type_as(pose).to(pose.device)
        # pose_E = pose_RT @ pose_inv_RT
        # pose_E_inv = pose_inv_RT @ pose_RT
        # inverse_loss = charbonnier((pose_E - E)) + charbonnier((pose_E_inv - E))
        inverse_loss = ((pose_RT.inverse() - pose_inv_RT)).abs() + ((pose_inv_RT.inverse() - pose_RT)).abs()
        return inverse_loss.sum()

def flow2pi(flow,pixel_coords):
    """
        Inverse warp a source image to the target image plane.

        Args:
            img: the source image (where to sample pixels) -- [B, 3, H, W]
            flow: flow map of the target image -- [B, 2, H, W]
             padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        Returns:
            Source image warped to the target image plane
        """
    check_sizes(flow, 'flow', 'B2HW')
    b, _,h,w = flow.size()

    X = pixel_coords[:,0,:,:] + flow[:,0,:,:]
    Y = pixel_coords[:,1,:,:] + flow[:,1,:,:]

    X_n = 2. * (X / (w - 1.0) - 0.5)
    Y_n = 2. * (Y / (h - 1.0) - 0.5)
    grid_tf = torch.stack((X_n, Y_n), dim=3)
    #img_tf = torch.nn.functional.grid_sample(img, grid_tf, padding_mode=padding_mode)
    valid_points = grid_tf.abs().max(dim=-1)[0] <= 1
    valid_mask = valid_points.unsqueeze(1).float()
    #valid_mask = 1 - (img_tf == 0).prod(1, keepdim=True).type_as(img_tf)
    return torch.stack((X.view(b,h*w),Y.view(b,h*w),pixel_coords[:,2,:,:].view(b,h*w)),dim=-2),valid_mask

def flow2img(flow12, flow13, intrinsics):
    check_sizes(flow12, 'flow', 'B2HW')
    b, _, h, w = flow12.size()
    # print(intrinsics.shape)
    # M1 = intrinsics.inverse() @ pixel_coords
    grid_x = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(flow12).expand_as(flow12[:,0,:,:])  # [b, H, W]
    grid_y = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(flow12).expand_as(flow12[:,1,:,:])  # [b, H, W]
    Z = torch.ones(b, h, w).type_as(flow12)
    pixel_coords = torch.stack((grid_x,grid_y,Z),dim=1)#(b,3,H,W)

    M1 = pixel_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    M2,mask2 = flow2pi(flow12,pixel_coords)
    M3,mask3 = flow2pi(flow13,pixel_coords)
    intrinsics_inv = intrinsics.inverse()
    M1 = (intrinsics_inv @ M1).reshape(b, h, w, 3)
    M2 = (intrinsics_inv @ M2).reshape(b, h, w, 3)
    M3 = (intrinsics_inv @ M3).reshape(b, h, w, 3)
    #  M1 = anti_matrix(M1)
    #  M2_anti = anti_matrix(M2)
    #  M3_anti = anti_matrix(M3)
    return M1, M2, M3, mask2, mask3

def  anti_matrix(M):
    b,h,w,_ = M.size()
    A_M = torch.zeros((b,h,w,3,3)).to(M.device)
    A_M[:,:,:, 0 ,1] =-M[:,:,:,2]
    A_M[:, :, :, 0, 2] = M[:, :, :, 1]
    A_M[:, :, :, 1, 0] = M[:, :, :, 2]
    A_M[:, :, :, 1, 2] = -M[:, :, :, 0]
    A_M[:, :, :, 2, 0] = -M[:, :, :, 1]
    A_M[:, :, :, 2, 1] = M[:, :, :, 0]
    return A_M

def compute_trifocal_flow_loss(T21,T31,flow21,flow31,intrinsics,rigid_mask):
    b, _, h, w = flow21.size()
    R21,t21 = T21[:,:,:3],T21[:,:,-1:]
    R31, t31 = T31[:, :, :3], T31[:, :, -1:]
    M1, M2, M3, mask2, mask3 = flow2img(flow21, flow31, intrinsics)
    M2_anti = anti_matrix(M2)
    M3_anti = anti_matrix(M3)
    trifocal_loss = ((M2_anti.view(b * h * w, 3, 3) @ R21.expand(h * w, b, 3, 3).reshape(b * h * w, 3, 3) @ M1.view(
        b * h * w, 3, 1) @ t31.transpose(1, 2).expand(h * w, b, 1, 3).reshape(b * h * w, 1, 3) @ M3_anti.view(b * h * w,
                                                                                                              3, 3)) \
                    - (M2_anti.view(b * h * w, 3, 3) @ t21.expand(h * w, b, 3, 1).reshape(b * h * w, 3, 1) @ M1.view(
        b * h * w, 1, 3) @ R31.expand(h * w, b, 3, 3).reshape(b * h * w, 3, 3).transpose(1, 2) @ M3_anti.view(b * h * w,
                                                                                                        3, 3)))**2
    mask_all = mask2 * mask3 * rigid_mask
    trifocal_loss = trifocal_loss.view(b, h, w, 9).sum(-1)
    trifocal_loss = mean_on_mask(trifocal_loss, mask_all.squeeze(1))
    return trifocal_loss

@torch.no_grad()
def compute_errors(gt, pred):
    # pred : b c h w
    # gt: b h w

    abs_diff = abs_rel = sq_rel = log10 = rmse = rmse_log = a1 = a2 = a3 = 0.0

    batch_size, h, w = gt.size()

    if pred.nelement() != gt.nelement():
        pred = F.interpolate(
            pred, [h, w], mode='bilinear', align_corners=False)

    pred = pred.view(batch_size, h, w)

    crop_mask = gt[0] != gt[0]
    y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
    x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
    crop_mask[y1:y2, x1:x2] = 1
    max_depth = 80

    min_depth = 0.1
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        # align scale
        valid_pred = valid_pred * \
            torch.median(valid_gt)/torch.median(valid_pred)

        valid_pred = valid_pred.clamp(min_depth, max_depth)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = valid_gt - valid_pred
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / valid_gt)
        sq_rel += torch.mean(((diff_i)**2) / valid_gt)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(valid_gt) -
                               torch.log(valid_pred)) ** 2))
        log10 += torch.mean(torch.abs((torch.log10(valid_gt) -
                            torch.log10(valid_pred))))

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3]]
