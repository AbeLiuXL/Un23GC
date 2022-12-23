from __future__ import division
import argparse
from tqdm import tqdm
import numpy as np
from path import Path
from torch import nn
import torch
import models
from datasets.kitti_flow import KittiFlow2012,KittiFlow2015
from utils.load_model import load_pre_model
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='Evaluate optical flow on KITTI',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir',default="../Data/Kitti/kitti_flow2015", metavar='PATH')
parser.add_argument('--flownet', type=str, default='PWCDCNet', choices=['Back2Future', 'FlowNetC6','PWCDCNet'],help='flow network architecture.')
parser.add_argument('--pre_path',default="/media/lxl/Data/Home/Projects/UnFlowVO2022/checkpoints/1221_f_e200_b4es500x4_fpw10fcw0.01fsw100_s4/epoch=196-step=98500-V_loss=1.190-T_loss=1.560.ckpt", metavar='PATH', help='path to pre-trained flownet model')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="checkpoints/results/flow/test/", help='path to output directory')
parser.add_argument("--output_name", default='test', type=str)
parser.add_argument('--dataset', default='kitti2015',choices=['kitti2015','kitti2012'])
args = parser.parse_args()

@torch.no_grad()
def main():
    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)/args.output_name
        args.output_dir.makedirs_p()

        image_dir = args.output_dir / 'images'
        gt_dir = args.output_dir / 'gt'
        mask_dir = args.output_dir / 'mask'
        viz_dir = args.output_dir / 'viz'

        image_dir.makedirs_p()
        gt_dir.makedirs_p()
        mask_dir.makedirs_p()
        viz_dir.makedirs_p()
    test_transform = transforms.Compose([
        transforms.Resize(size=[256, 832]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    # test_transform = T.Compose([
    #                             T.RescaleTo(),
    #                             T.ArrayToTensor(),
    #                             T.Normalize()])

    if args.dataset == "kitti2015":
        test_flow_set = KittiFlow2015(root=args.data_dir,transform=test_transform)
        Num=200
    else:
        test_flow_set = KittiFlow2012(root=args.data_dir,transform=test_transform)
        Num = 194
    test_loader = torch.utils.data.DataLoader(test_flow_set, batch_size=1, shuffle=False,
                                             num_workers=4, pin_memory=True, drop_last=False)
    flow_net = getattr(models, args.flownet)().cuda()
    if args.pre_path:
        print("=> using pre-trained weights for FlowNet")
        flow_net=load_pre_model(flow_net,args.pre_path,str_h='flow_net.')
    else:
        print('pre_path=None')
    flow_net.eval()
    error_all = 0
    error_noc = 0
    error_occ = 0
    error_move = 0
    error_static = 0
    F1 = 0
    error_names = ['epe_all', 'epe_occ', 'epe_noc', 'Fl']
    for i, (tgt_img, ref_img, intrinsics, intrinsics_inv, flow_gt_occ, noc_mask, obj_map_gt) in enumerate(tqdm(test_loader)):
        tgt_img = tgt_img.cuda()
        ref_img = ref_img.cuda()
        flow_gt_occ = flow_gt_occ.cuda()
        noc_mask = noc_mask.float().cuda()
        # obj_map_gt = obj_map_gt.cuda()
        flow_fwd = flow_net(tgt_img, ref_img)[0]
        # obj_map_gt = obj_map_gt.unsqueeze(1).type_as(flow_fwd)
        # flow_fwd_img = flow_to_image(flow_fwd[0].detach().cpu().numpy()).transpose(1, 2, 0)
        # imsave('/img_joint/{:0>6d}.png'.format(i), flow_fwd_img)
        error_all += compute_epe(flow_gt_occ[:, 0:2, :, :], flow_fwd, flow_gt_occ[:, 2, :, :])
        F1 += outlier_err(flow_gt_occ, flow_fwd)
        error_occ += compute_epe(flow_gt_occ[:, 0:2, :, :], flow_fwd, flow_gt_occ[:, 2, :, :] - noc_mask)
        error_noc += compute_epe(flow_gt_occ[:, 0:2, :, :], flow_fwd, noc_mask)

    print("Results")
    print("Name   \t {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("Errors \t {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(error_all / Num, error_occ / Num, error_noc / Num,
                                                                    F1 / Num))


epsilon = 1e-8
def compute_epe(gt, pred,valid):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()

    u_gt, v_gt = gt[:,0,:,:], gt[:,1,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear',align_corners=True)
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid
    avg_epe = epe.sum() / (valid.sum() + epsilon)
    # if nc == 3:
    #     valid = gt[:,2,:,:]
    #     epe = epe * valid
    #     avg_epe = epe.sum()/(valid.sum() + epsilon)
    # else:
    #     avg_epe = epe.sum()/(bs*h_gt*w_gt)

    # if type(avg_epe) == Variable: avg_epe = avg_epe.data
    return avg_epe.item()

def outlier_err(gt, pred, tau=[3,0.05]):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt.size()
    u_gt, v_gt, valid_gt = gt[:,0,:,:], gt[:,1,:,:], gt[:,2,:,:]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear',align_corners=True)
    u_pred = pred[:,0,:,:] * (w_gt/w_pred)
    v_pred = pred[:,1,:,:] * (h_gt/h_pred)

    epe = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2))
    epe = epe * valid_gt

    F_mag = torch.sqrt(torch.pow(u_gt, 2)+ torch.pow(v_gt, 2))
    E_0 = (epe > tau[0]).type_as(epe)
    E_1 = ((epe / (F_mag+epsilon)) > tau[1]).type_as(epe)
    n_err = E_0 * E_1 * valid_gt
    #n_err   = length(find(F_val & E>tau(1) & E./F_mag>tau(2)));
    #n_total = length(find(F_val));
    f_err = n_err.sum()/(valid_gt.sum() + epsilon);
    #if type(f_err) == Variable: f_err = f_err.data
    return f_err.item()
def calculate_error_rate(gt_flow,pred, mask):
    _, _, h_pred, w_pred = pred.size()
    bs, nc, h_gt, w_gt = gt_flow.size()
    u_gt, v_gt, valid_gt = gt_flow[:, 0, :, :], gt_flow[:, 1, :, :], gt_flow[:, 2, :, :]
    pred = nn.functional.interpolate(pred, size=(h_gt, w_gt), mode='bilinear', align_corners=True)
    u_pred = pred[:, 0, :, :] * (w_gt / w_pred)
    v_pred = pred[:, 1, :, :] * (h_gt / h_pred)
    mask = mask.detach().cpu().numpy()
    epe_map = torch.sqrt(torch.pow((u_gt - u_pred), 2) + torch.pow((v_gt - v_pred), 2)).detach().cpu().numpy()
    bad_pixels = np.logical_and(
        epe_map * mask > 3,
        epe_map * mask / np.maximum(
            np.sqrt(np.sum(np.square(gt_flow.detach().cpu().numpy()), axis=1)), 1e-10) > 0.05)
    return bad_pixels.sum() / mask.sum()
if __name__ == '__main__':
    main()