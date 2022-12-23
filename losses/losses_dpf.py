from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.losses_block import SSIM,mean_on_mask,disp_smooth_loss,inverse_poses_loss,EPE,smooth_grad_2nd,compute_trifocal_flow_loss
from losses.inverse_warp import inverse_warp,flow_warp

class DepthPoseFlowLosses(nn.Module):
    def __init__(self,args):
        super(DepthPoseFlowLosses, self).__init__()
        self.args = args
        self.ssim = SSIM()
    def dpf_loss(self, tgt_img_scaled, ref_img_scaled, \
                 tgt_depth, ref_depth, \
                 current_pose, current_pose_inv, \
                 current_flow, current_flow_inv, \
                 intrinsic_scaled, s, h, w):
        dp_ref_img_warped, dp_valid_mask, projected_depth_r, computed_depth, dp_flow \
            = inverse_warp(ref_img_scaled, tgt_depth, ref_depth, current_pose, intrinsic_scaled)
        if self.args.df_occ:
            f_ref_img_warped,f_fb_bw_warped,projected_depth, f_valid_mask, f_occu_mask = flow_warp(ref_img_scaled, current_flow, current_flow_inv,ref_depth)
        else:
            f_ref_img_warped,f_fb_bw_warped, f_valid_mask, f_occu_mask = flow_warp(ref_img_scaled, current_flow, current_flow_inv)
            projected_depth = projected_depth_r
        if self.args.rigid_mask:
            rigid_mask = 1 - (EPE(current_flow, dp_flow)**0.5 / (h**2+w**2)**0.5)
            #rigid_mask = (rigid_mask - torch.min(rigid_mask.detach())) / (torch.max(rigid_mask.detach())-torch.min(rigid_mask.detach()))
            rigid_mask = rigid_mask.detach()
        if self.args.hard_rigid_mask:
            # rigid_mask = (rigid_mask + (f_occu_mask > 0.2).float()).clamp(0., 1.)
            rigid_mask = (rigid_mask.detach() > (torch.mean(rigid_mask.detach()) * 1000 // 1 / 1000)) & (
                    tgt_depth.detach() < (0.5 * torch.max(tgt_depth.detach())))
            rigid_mask = rigid_mask.detach()
        if self.args.flow_occ>0:
            f_occu_mask = (f_occu_mask > self.args.flow_occ).float()

        diff_img_abs = (tgt_img_scaled - dp_ref_img_warped).abs()
        diff_img_ssim = self.ssim(tgt_img_scaled, dp_ref_img_warped)
        if self.args.auto_mask:
            auto_mask = (diff_img_abs.mean(dim=1, keepdim=True) < (tgt_img_scaled - ref_img_scaled).abs().mean(dim=1,
                                                                                                               keepdim=True)).float()
        else:
            auto_mask = torch.tensor(1.0)
        ################################## loss_consistancy ##################################
        # dpf_consistancy = charbonnier(dpf_consistancy)
        f_consistancy = (current_flow + f_fb_bw_warped).abs()  # 前后一致
        p_consistancy = inverse_poses_loss(current_pose, current_pose_inv)  # 前后一致

        # projected_depth = torch.mean(computed_depth * f_valid_mask) / torch.mean(
        #     projected_depth * f_valid_mask) * projected_depth
        # projected_depth_r = torch.mean(computed_depth * dp_valid_mask) / torch.mean(
        #     projected_depth_r * dp_valid_mask) * projected_depth_r
        d_consistancy_o = (
                    (computed_depth - projected_depth).abs() / (computed_depth + projected_depth).abs()).clamp(0,
                                                                                                               1)  # * rigid_mask.float()  # 前后一致
        d_consistancy_r = (
                    (computed_depth - projected_depth_r).abs() / (computed_depth + projected_depth_r).abs()).clamp(
            0, 1)  # * rigid_mask.float()

        ################################## mask ##################################
        if self.args.depth_mask:
            d_occ_weight_mask =( (1 - d_consistancy_r.detach())*f_valid_mask + (1 - d_consistancy_o.detach())*dp_valid_mask)/2.0
            d_occ_weight_mask=d_occ_weight_mask.detach()
        else:
            d_occ_weight_mask = torch.tensor(1.0)

        f_photomatric = 0.85 * self.ssim(tgt_img_scaled, f_ref_img_warped) + 0.15 * (tgt_img_scaled - f_ref_img_warped).abs().clamp(0, 1)
        f_photomatric = mean_on_mask(f_photomatric, f_valid_mask*f_occu_mask)

        dpf_consistancy = (current_flow - dp_flow).abs().mean(1, keepdim=True)
        dpf_consistancy_o = (current_flow - dp_flow.detach()).abs().mean(1, keepdim=True)

        f_smooth = smooth_grad_2nd( current_flow / h, tgt_img_scaled, 10., None, None)

        # d_smooth = smooth_grad_2nd(tgt_depth / (tgt_depth.mean(2, True).mean(3, True) + 1e-7), tgt_img_scaled, 1, None,
        #                            None) / 2.3 ** s
        d_smooth = disp_smooth_loss(tgt_depth, tgt_img_scaled)
        dp_photomatric = 0.85 * diff_img_ssim + 0.15 * diff_img_abs.clamp(0, 1)
        dp_photomatric = torch.mean(dp_photomatric, dim=1, keepdim=True)
        if self.args.rigid_mask and self.args.hard_rigid_mask:
            dp_photomatric = mean_on_mask(dp_photomatric  * d_occ_weight_mask,
                                          dp_valid_mask * auto_mask *rigid_mask.float())
            dpf_consistancy = mean_on_mask(dpf_consistancy , f_occu_mask*f_valid_mask * rigid_mask.float()) + mean_on_mask(
                dpf_consistancy_o, (1 - f_occu_mask*f_valid_mask*dp_valid_mask))
        elif self.args.rigid_mask:
            dp_photomatric = mean_on_mask(dp_photomatric *d_occ_weight_mask*rigid_mask.float(), dp_valid_mask*auto_mask)
            dpf_consistancy = mean_on_mask(dpf_consistancy* rigid_mask.float(),f_occu_mask*f_valid_mask )+ mean_on_mask(dpf_consistancy_o,(1-f_occu_mask*f_valid_mask*dp_valid_mask))
        else:
            dp_photomatric = mean_on_mask(dp_photomatric * d_occ_weight_mask,
                                          dp_valid_mask * auto_mask)
            dpf_consistancy = mean_on_mask(dpf_consistancy,
                                           f_occu_mask*f_valid_mask) + mean_on_mask(
                dpf_consistancy_o, (1 - f_occu_mask*f_valid_mask))
        f_consistancy = mean_on_mask(f_consistancy , f_valid_mask * f_occu_mask)
        d_consistancy_o = mean_on_mask(d_consistancy_o , f_valid_mask)
        d_consistancy_r = mean_on_mask(d_consistancy_r , dp_valid_mask)
        d_consistancy = (d_consistancy_o + d_consistancy_r)/2.0

        if not self.args.hard_rigid_mask:
            rigid_mask = (rigid_mask.detach() > (torch.mean(rigid_mask.detach())*1000//1/1000)) & (tgt_depth.detach() < (0.5 * torch.max(tgt_depth.detach())))
        return dp_photomatric,f_photomatric,\
               dpf_consistancy,f_consistancy,p_consistancy,d_consistancy, \
               f_smooth,d_smooth, \
               rigid_mask

    def forward(self, imgs, intrinsics, depths, poses, poses_inv, flows, flows_inv):
        """
        :param imgs: [i-1,i,i+1],3*[B3HW]
        :param intrinsics:
        :param depths: n * [B1hw]
        :param poses: [pose_net(imgs[i-1dpf_consistancy], imgs[i]), pose_net(imgs[i-1], imgs[i+1]), pose_net(imgs[i], imgs[i+1])]
        :param poses_inv: [flow_net(imgs[i-1], imgs[i]), flow_net(imgs[i-1], imgs[i+1]), flow_net(imgs[i], imgs[i+1])]
        :param flows:B2HW
        :param flows_inv:
        :return:
        """
        DP_photomatric_loss = 0
        F_photomatric_loss = 0

        D_consistancy_loss = 0
        P_consistancy_loss = 0
        F_consistancy_loss = 0
        DPF_consistancy_loss = 0

        D_smooth_loss = 0
        F_smooth_loss = 0

        PF_trifocal_loss = 0
        rigid_loss = 0
        pyramid_mask = []

        imgpart = [[0, 1], [0, 2], [1, 2]]
        num_scales = min(self.args.num_scales, len(depths[0]), len(flows[0]))
        for i, ip in enumerate(imgpart):
            tgt_depth = depths[ip[0]]
            ref_depth = depths[ip[1]]
            tgt_img = imgs[ip[0]]
            ref_img = imgs[ip[1]]

            current_pose = poses[i]
            current_pose_inv = poses_inv[i]
            current_flow = flows[i]
            current_flow_inv = flows_inv[i]
            pyramid_mask_i = []
            for s in range(num_scales):
                b, _, h, w = tgt_depth[s].size()
                _, _, h_f, w_f = current_flow[s].size()
                downscale = tgt_img.size(2) / h
                tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='bilinear',align_corners=False)
                ref_img_scaled = F.interpolate(ref_img, (h, w), mode='bilinear',align_corners=False)
                intrinsic_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)

                dp_photomatric1, f_photomatric1, \
                dpf_consistancy1, f_consistancy1, p_consistancy1, d_consistancy1, \
                f_smooth1, d_smooth1, \
                rigid_mask1 = self.dpf_loss(tgt_img_scaled, ref_img_scaled, \
                                            tgt_depth[s], ref_depth[s], \
                                            current_pose, current_pose_inv, \
                                            current_flow[s], current_flow_inv[s], \
                                            intrinsic_scaled, s, h_f, w_f)

                dp_photomatric2, f_photomatric2, \
                dpf_consistancy2, f_consistancy2, p_consistancy2, d_consistancy2, \
                f_smooth2, d_smooth2, \
                rigid_mask2 = self.dpf_loss(ref_img_scaled, tgt_img_scaled, \
                                            ref_depth[s], tgt_depth[s], \
                                            current_pose_inv, current_pose, \
                                            current_flow_inv[s], current_flow[s], \
                                            intrinsic_scaled, s, h_f, w_f)

                DP_photomatric_loss += (dp_photomatric1 + dp_photomatric2)
                F_photomatric_loss += (f_photomatric1 + f_photomatric2)

                D_consistancy_loss += (d_consistancy1 + d_consistancy2)
                P_consistancy_loss += (p_consistancy1 + p_consistancy2)
                F_consistancy_loss += (f_consistancy1 + f_consistancy2)
                DPF_consistancy_loss += (dpf_consistancy1 + dpf_consistancy2)

                D_smooth_loss += (d_smooth1 + d_smooth2)/2.0
                F_smooth_loss += (f_smooth1 + f_smooth2)/2.0
                pyramid_mask_i.append([rigid_mask1, rigid_mask2])
            pyramid_mask.append(pyramid_mask_i)

        for s in range(num_scales):
            _, _, h_f, w_f = flows[0][s].size()
            downscale = tgt_img.size(2) / h_f
            intrinsic_scaled = torch.cat((intrinsics[:, 0:2] / downscale, intrinsics[:, 2:]), dim=1)

            flow21 = flows[0][s]
            flow31 = flows[1][s]

            flow12 = flows_inv[0][s]
            flow32 = flows[2][s]

            flow13 = flows_inv[1][s]
            flow23 = flows_inv[2][s]

            if self.args.rigid_mask:
                rigid_mask1 = pyramid_mask[0][s][0] & pyramid_mask[1][s][0]
                rigid_mask2 = pyramid_mask[0][s][1] & pyramid_mask[2][s][0]
                rigid_mask3 = pyramid_mask[1][s][1] & pyramid_mask[2][s][1]
                rigid_mask1 = rigid_mask1.float()
                rigid_mask2 = rigid_mask2.float()
                rigid_mask3 = rigid_mask3.float()
            else:
                rigid_mask1 = 1
                rigid_mask2 = 1
                rigid_mask3 = 1

            # rigid_mask = pyramid_mask[0][s] & pyramid_mask[1][s] & pyramid_mask[2][s]

            trifocal_loss1 = compute_trifocal_flow_loss(poses[0], poses[1], flow21, flow31, intrinsic_scaled,
                                                         rigid_mask1)
            trifocal_loss2 = compute_trifocal_flow_loss(poses_inv[0], poses[2], flow12, flow32, intrinsic_scaled,
                                                         rigid_mask2)
            trifocal_loss3 = compute_trifocal_flow_loss(poses_inv[1], poses_inv[2], flow13, flow23, intrinsic_scaled,
                                                         rigid_mask3)
            PF_trifocal_loss += trifocal_loss1 + trifocal_loss2 + trifocal_loss3

        return DP_photomatric_loss/(3.0), F_photomatric_loss/(3.0), \
               D_consistancy_loss/(3.0), P_consistancy_loss/(3.0), F_consistancy_loss/(3.0), DPF_consistancy_loss/(3.0), \
               D_smooth_loss/(3.0), F_smooth_loss/(3.0), PF_trifocal_loss/(3.0)