import torch
from skimage.transform import resize as imresize
from imageio import imread
from path import Path
import argparse
import models
from utils.load_model import load_pre_model
from tqdm import trange
import numpy as np
from utils.kitti_eval.depth_evaluation_utils import *
parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pre_path",default="checkpoints/1216_dp2_dppw1_dcw0.1_dsw0.1_pcw0.01-pcse3_dpse3_es250x4/last.ckpt", type=str, help="pretrained DispNet path")
parser.add_argument("--output_dir", default='checkpoints/results/depth/', type=str,
                    help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--output_name", default='test', type=str)

parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--data_dir", default="../Data/Kitti/sync/kitti_raw_sync_zip/", type=str, help="Dataset directory")
parser.add_argument("--data_list", default="utils/kitti_eval/test_files_eigen.txt", type=str, help="Dataset list file")

parser.add_argument('--dispnet', dest='dispnet', type=str,default='DispResNet',help='depth network architecture.')
parser.add_argument("--num_layers", type=int, default=18, choices=[18, 50, 101])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# def load_tensor_image(filename, args):
#     img = imread(filename).astype(np.float32)
#     h, w, _ = img.shape
#     if (not args.no_resize) and (h != args.img_height or w != args.img_width):
#         img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
#     img = np.transpose(img, (2, 0, 1))
#     tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
#     return tensor_img

def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    #tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
    return tensor_img
args = parser.parse_args()
@torch.no_grad()
def main():
    disp_net = getattr(models, args.dispnet)(num_layers=args.num_layers).to(device)
    if args.pre_path:
        print("=> using pre-trained weights for PoseNet")
        disp_net = load_pre_model(disp_net,args.pre_path,'disp_net.')
    else:
        print('pre_path=None')
    disp_net.eval()

    dataset_dir = Path(args.data_dir)
    with open(args.data_list, 'r') as f:
        test_files = list(f.read().splitlines())
    # print('{} files to test'.format(len(test_files)))

    output_dir = Path(args.output_dir)/args.output_name
    output_dir.makedirs_p()

    for j in trange(len(test_files)):
        tgt_img = load_tensor_image(dataset_dir + test_files[j], args)
        pred_disp = disp_net(tgt_img)[0].cpu().numpy()[0, 0]
        # pred_disp = disp_net(tgt_img)
        # pred_disp = F.interpolate(pred_disp, (64, 208), mode='area').cpu().numpy()[0,0]
        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        predictions[j] = 1 / pred_disp

    np.save(output_dir / 'predictions.npy', predictions)

def eval():
    pred_file = Path(args.output_dir)/args.output_name /'predictions.npy'
    pred_depths = np.load(pred_file)
    test_files = read_text_lines(args.data_list)
    gt_files, gt_calib, im_sizes, im_files, cams = \
        read_file_data(test_files, args.data_dir)
    num_test = len(im_files)
    gt_depths = []
    pred_depths_resized = []
    for t_id in trange(num_test):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        pred_depths_resized.append(
            cv2.resize(pred_depths[t_id],
                       (im_sizes[t_id][1], im_sizes[t_id][0]),
                       interpolation=cv2.INTER_LINEAR))
        depth = generate_depth_map(gt_calib[t_id],
                                   gt_files[t_id],
                                   im_sizes[t_id],
                                   camera_id,
                                   False,
                                   True)
        gt_depths.append(depth.astype(np.float32))
    pred_depths = pred_depths_resized

    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)
    scalors = []
    for i in trange(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > args.min_depth,
                              gt_depth < args.max_depth)
        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
        # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,
                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        # Scale matching
        scalor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        scalors.append(scalor)
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])
    print('scale_factor:',np.mean(np.array(scalors)))
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))


if __name__ == '__main__':
    main()
    eval()