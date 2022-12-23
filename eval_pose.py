from utils.kitti_eval.kitti_evo import calculate_sequence_error, calculate_ave_errors
import numpy as np
import argparse
import torch
import models
from path import Path
from imageio import imread
from utils.load_model import load_pre_model
from tqdm import trange
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
parser = argparse.ArgumentParser(
    description='Unsupervised Scale-consistent Depth and Ego-motion Learning from Monocular Video (KITTI and CityScapes)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir',  type=str ,default='../Data/Kitti/sync/kitti_vo_256/',
                    help='path to dataset')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="checkpoints/results/pose/", help='path to output directory')
parser.add_argument("--output_name", default='test', type=str)
parser.add_argument('--pre_path', default='/media/lxl/Data/Home/Projects/UnFlowVO2022/checkpoints/1220_dp_adamw_dppw1_dcw0.1_dsw0.1_pcw0.01_se3_es250x4/last.ckpt', metavar='PATH',
                    help='path to pre-trained Pose net model')
parser.add_argument('--posenet', dest='posenet', type=str, default='PoseResNet',
                    help='depth network architecture.')
parser.add_argument("--num_layers", type=int, default=18, choices=[18, 50, 101])
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")

args = parser.parse_args()


@torch.no_grad()
def validate_pose_with_gt(pose_net, seq):

    pose_net.eval()

    rot_error = []
    tra_error = []
    def load_tensor_image(filename):
        img = imread(filename).astype(np.float32)
        h, w, _ = img.shape
        img = np.transpose(img, (2, 0, 1))
        tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.45) / 0.225).to(device)
        #tensor_img = ((torch.from_numpy(img).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
        return tensor_img

    # logger.valid_bar.update(0)

    for s in seq:
        image_dir = Path(args.data_dir + '{:0>2d}_2'.format(s))
        test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
        test_files.sort()
        # print('{} files to test'.format(len(test_files)))
        global_pose = np.identity(4)
        poses = [global_pose[0:3, :].reshape(1, 12)]
        n = len(test_files)
        tensor_img1 = load_tensor_image(test_files[0])

        for iter in trange(n - 1):
            tensor_img2 = load_tensor_image(test_files[iter + 1])
            pose = pose_net(tensor_img1, tensor_img2)
            #print(pose)
            #pose[:, :3] = pose[:, :3]*100
            #pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
            pose_mat = (pose).squeeze(0).cpu().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @ np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :].reshape(1, 12))
            # update
            tensor_img1 = tensor_img2

            # logger.valid_bar.update(i)
        poses_gt = np.loadtxt('utils/kitti_eval/poses/{:0>2d}.txt'.format(s))
        poses = np.array(poses)
        scale_factor = np.sum(poses_gt.reshape(-1, 3, 4)[:, :, -1] * poses.reshape(-1, 3, 4)[:, :, -1]) / np.sum(
            poses.reshape(-1, 3, 4)[:, :, -1] ** 2)
        print("scale_factor:",scale_factor)
        np.savetxt(args.output_dir/'{:0>2d}.txt'.format(s), poses.reshape(-1, 12))
        poses.reshape(-1, 3, 4)[:, :, -1] = poses.reshape(-1, 3, 4)[:, :, -1] * scale_factor
        poses = poses.reshape(-1, 12)
        #np.savetxt('pose_result/{:0>2d}.txt'.format(s),poses)
        errors = calculate_sequence_error(poses_gt, poses)
        rot, tra = calculate_ave_errors(errors)
        rot = np.array(rot).mean() * 180 / np.pi * 100
        tra = np.array(tra).mean() * 100
        rot_error.append(rot)
        tra_error.append(tra)
    return rot_error, tra_error


if __name__ == '__main__':
    print( args.posenet)
    pose_net = getattr(models, args.posenet)(num_layers=args.num_layers).to(device)
    if args.pre_path:
        print("=> using pre-trained weights for PoseNet")
        pose_net = load_pre_model(pose_net,args.pre_path,'pose_net.')
    else:
        print('pre_path=None')

    if args.output_dir is not None:
        args.output_dir = Path(args.output_dir)/args.output_name
        args.output_dir.makedirs_p()

    rot_error, tra_error = validate_pose_with_gt(pose_net,[9])
    print(rot_error, tra_error)
