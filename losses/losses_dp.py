from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.losses_block import SSIM,mean_on_mask,disp_smooth_loss,inverse_poses_loss
from losses.inverse_warp import inverse_warp

class DepthPoseLosses(nn.Module):
    def __init__(self,args):
        super(DepthPoseLosses, self).__init__()
        self.args = args
        self.ssim = SSIM()

    def dp_losses(self,tgt_img_scaled,ref_img_scaled,\
                 tgt_depth,ref_depth,\
                 current_pose,current_pose_inv,
                 intrinsic_scaled,s):
        ################################## warped ##############################
        dp_ref_img_warped, dp_valid_mask, projected_depth_r, computed_depth, dp_flow \
            = inverse_warp(ref_img_scaled, tgt_depth, ref_depth, current_pose, intrinsic_scaled)

        p_consistancy = inverse_poses_loss(current_pose, current_pose_inv)

        d_consistancy_r = ((computed_depth - projected_depth_r).abs() / (
                    computed_depth + projected_depth_r).abs())

        ################################## loss_photomatric ##################################
        diff_img_abs = (tgt_img_scaled-dp_ref_img_warped).abs()
        diff_img_ssim  = self.ssim(tgt_img_scaled,dp_ref_img_warped)

        ################################## mask ##################################

        d_occ_weight_mask = (1 - d_consistancy_r).detach()
        if self.args.auto_mask == 1:
            auto_mask = (diff_img_abs.mean(dim=1, keepdim=True) < (tgt_img_scaled - ref_img_scaled).abs().mean(dim=1,
                                                                                             keepdim=True)).float()
            dp_valid_mask = auto_mask * dp_valid_mask

        # if self.cfg.d_hard_occu_th > 0:
        #     d_occ_weight_mask = (d_occ_weight_mask < self.cfg.d_hard_occu_th).float()

        ##################################### loss_smooth ####################################
        d_smooth = disp_smooth_loss(tgt_depth,tgt_img_scaled)

        ##################################### loss_mean ####################################
        dp_photomatric = 0.85 * diff_img_ssim + 0.15 * diff_img_abs.clamp(0, 1)
        dp_photomatric = torch.mean(dp_photomatric, dim=1, keepdim=True)
        d_consistancy = mean_on_mask(d_consistancy_r , dp_valid_mask)
        dp_photomatric = mean_on_mask(dp_photomatric * d_occ_weight_mask, dp_valid_mask)

        return dp_photomatric, p_consistancy, d_consistancy, d_smooth, d_occ_weight_mask

    def forward(self,imgs,intrinsics,depths,poses, poses_inv):
        """
        :param imgs:[1,2,3]
        :param intrinsics:[4x4]
        :param depths:[1,2,3]
        :param poses:[12,13,23]
        :param poses_inv:[21,31,32]
        :param num_scales:1 or 4
        :return:
        """
        DP_photomatric_loss = 0
        D_consistancy_loss = 0
        P_consistancy_loss = 0
        D_smooth_loss = 0
        imgpart = [[0, 1], [0, 2], [1, 2]]
        num_scales = min(self.args.num_scales,len(depths[0]))
        for i, ip in enumerate(imgpart):
            tgt_depth = depths[ip[0]]
            ref_depth = depths[ip[1]]
            tgt_img = imgs[ip[0]]
            ref_img = imgs[ip[1]]
            current_pose = poses[i]
            current_pose_inv = poses_inv[i]
            for s in range(num_scales):
                b, _, h, w = tgt_depth[s].size()
                if tgt_img.size(2)!= h:
                    downscale = tgt_img.size(2) / h
                    tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='bilinear',align_corners=False)
                    ref_img_scaled = F.interpolate(ref_img, (h, w), mode='bilinear',align_corners=False)
                    intrinsic_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)
                else:
                    tgt_img_scaled = tgt_img
                    ref_img_scaled = ref_img
                    intrinsic_scaled = intrinsics


                dp_photomatric1, p_consistancy1, d_consistancy1, d_smooth1, d_occ_weight_mask1 = self.dp_losses(tgt_img_scaled,ref_img_scaled,\
                 tgt_depth[s],ref_depth[s],\
                 current_pose,current_pose_inv,\
                 intrinsic_scaled,s)

                dp_photomatric2, p_consistancy2, d_consistancy2, d_smooth2, d_occ_weight_mask2 = self.dp_losses(ref_img_scaled,tgt_img_scaled , \
                                                                 ref_depth[s],tgt_depth[s], \
                                                                current_pose_inv,current_pose,  \
                                                                intrinsic_scaled, s)

                DP_photomatric_loss += (dp_photomatric1 + dp_photomatric2)
                D_consistancy_loss += (d_consistancy1+d_consistancy2)
                P_consistancy_loss += (p_consistancy1+p_consistancy2)
                D_smooth_loss += (d_smooth1+d_smooth2)

        return DP_photomatric_loss/3.0, D_consistancy_loss/3.0,P_consistancy_loss/3.0, D_smooth_loss/3.0
