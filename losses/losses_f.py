from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.inverse_warp import flow_warp
from losses.losses_block import mean_on_mask,SSIM,smooth_grad_2nd
class FlowLosses(nn.Module):
    def __init__(self,args):
        super(FlowLosses, self).__init__()
        self.args = args
        self.ssim = SSIM(md=7)

    def flowlosses(self,tgt_img,ref_img,cur_flow,cur_inv_flow,h,s):
        warped_img, warped_flow, valid_mask, occu_mask = flow_warp(ref_img,cur_flow,cur_inv_flow)

        if self.args.flow_occ > 0:
            occu_mask = (occu_mask > self.args.flow_occ).float()
        flow_smooth = smooth_grad_2nd(cur_flow/h, tgt_img,10)
        # flow_smooth = smooth_grad_1st(cur_flow, tgt_img, 1, None,False)
        flow_consistancy = (cur_flow + warped_flow).abs()*occu_mask
        flow_consistancy = mean_on_mask(flow_consistancy, valid_mask)
        # flow_consistancy = flow_consistancy.mean((1,2,3))

        flow_photomatric = 0.85*self.ssim(tgt_img,warped_img)+0.15*(tgt_img-warped_img).abs()
        flow_photomatric = mean_on_mask(flow_photomatric*occu_mask, valid_mask)
        # flow_photomatric=flow_photomatric.mean((1,2,3))
        return flow_photomatric,flow_consistancy,flow_smooth

    def forward(self,tgt_img,ref_img,flows,inv_flows):
        """
        :param tgt_img,tgt_img: B3HW
        :param flows: PWCNet , RAFT
        :param inv_flows: PWCNet , RAFT
        :return:
        """
        Flow_Photomatric = 0
        Flow_Consistancy = 0
        Flow_Smooth = 0
        num_scales = min(len(flows), self.args.num_scales)
        for s in range(num_scales):
            current_flow, current_inv_flow=flows[s],inv_flows[s]
            _, _, h_f, w_f = current_flow.size()
            tgt_img = torch.nn.functional.interpolate(tgt_img, (h_f, w_f),mode='bilinear',align_corners=False)
            ref_img = torch.nn.functional.interpolate(ref_img, (h_f, w_f),mode='bilinear',align_corners=False)

            flow_photomatric_t, flow_consistancy_t, flow_smooth_t = self.flowlosses(tgt_img,
                                                                                    ref_img,
                                                                                    current_flow,
                                                                                    current_inv_flow,h_f,s)
            flow_photomatric_r, flow_consistancy_r, flow_smooth_r = self.flowlosses(ref_img,
                                                                                    tgt_img,
                                                                                    current_inv_flow,
                                                                                    current_flow,h_f,s)
            Flow_Photomatric = (flow_photomatric_t + flow_photomatric_r)
            Flow_Consistancy = (flow_consistancy_t + flow_consistancy_r)
            Flow_Smooth = (flow_smooth_t + flow_smooth_r)
        return Flow_Photomatric,Flow_Consistancy,Flow_Smooth
