
#Un23GC
## Requirements
* torch >= 1.0.1
* torchvision
* pytorch-lightning
* opencv-python
* imageio
* matplotlib
* scipy
* path.py
* evo
* pypng

## Datasets
* Kitti Raw/Sync : ([Kitti Raw/Sync](https://www.cvlibs.net/datasets/kitti/raw_data.php))
* Kitti Visual Odometry : ([Kitti Visual Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php))
* Kitti Flow 2012 : ([Kitti Flow 2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow))
* Kitti Flow 2015 : ([Kitti Flow 2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow))

## Pre-trained models
[Pre-trained models](https://pan.baidu.com/s/1cYnXcSYiQ7-yU2RgMHdgLw) (Verification Code：c7af)

## Training
> export OMP_NUM_THREADS=4
> 
> export CUDA_VISIBLE_DEVICES=0,1,2,3
### Flow Net
>python train_pl_f.py --data_dir DATA_DIR --gpu_id 0,1,2,3  --save_name SAVE_NAME 
### Depth-Pose Net
>python train_pl_dp.py --data_dir DATA_DIR --gpu_id 0,1,2,3 --save_name SAVE_NAME 
### Depth-Pose-Flow Net
>python train_pl_dpf.py --data_dir DATA_DIR --gpu_id 0,1,2,3 --save_name SAVE_NAME --pre_dp checkpoints/... --pre_flow checkpoints/... 

## Evaluation
### Flow
> python eval_flow.py --data_dir DATA_DIR --pre_path PRE_PATH 
### Depth
> python eval_depth --data_dir DATA_DIR --pre_path PRE_PATH 
### Pose
> python eval_pose --data_dir DATA_DIR --pre_path PRE_PATH 
