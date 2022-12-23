import pytorch_lightning as pl
import torch
import utils.custom_transforms as T
from datasets.kitti_seq import Kitti_Seq
from datasets.validation_depth import ValidationSet
from utils.lr_scheduler import LR_Scheduler
from models.dispresnet import DispResNet
from models.poseresnet import PoseResNet
from losses.losses_dp import DepthPoseLosses
from losses.losses_block import compute_errors
import numpy as np


class DP_Module(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.args = args
        self.disp_net=DispResNet(num_layers = args.num_layers)
        self.pose_net=PoseResNet(num_layers = args.num_layers)
        self.loss_functions = DepthPoseLosses(args)
        self.normalize = T.Normalize()

    def compute_depth(self,disp_net, imgs):
        depths = []
        for img in imgs:
            depth = [1 / disp for disp in disp_net(img)]
            depths.append(depth)
        return depths

    def compute_pose(self, pose_net, imgs):
        poses = [pose_net(imgs[0], imgs[1]), pose_net(imgs[0], imgs[2]), pose_net(imgs[1], imgs[2])]
        poses_inv = [pose_net(imgs[1], imgs[0]), pose_net(imgs[2], imgs[0]), pose_net(imgs[2], imgs[1])]
        return poses, poses_inv

    def forward(self, batch):
        imgs,intrinsics=batch
        depths = self.compute_depth(self.disp_net, imgs)
        poses, poses_inv = self.compute_pose(self.pose_net, imgs)
        DP_photomatric_loss, D_consistancy_loss,P_consistancy_loss, D_smooth_loss=self.loss_functions(imgs,intrinsics,depths,poses, poses_inv)
        loss = self.args.dppw*DP_photomatric_loss\
        +self.args.dcw*D_consistancy_loss \
        + self.args.pcw * P_consistancy_loss \
        +self.args.dsw*D_smooth_loss
        return loss,DP_photomatric_loss, D_consistancy_loss,P_consistancy_loss, D_smooth_loss

    def training_step(self, batch, batch_nb):
        loss, DP_photomatric_loss, D_consistancy_loss, P_consistancy_loss, D_smooth_loss = self.forward(batch)
        self.log("T_DP_photomatric_loss", DP_photomatric_loss.item())
        self.log("T_D_consistancy_loss", D_consistancy_loss.item())
        self.log("T_P_consistancy_loss", P_consistancy_loss.item())
        self.log("T_D_smooth_loss", D_smooth_loss.item())
        self.log("T_loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):

        if self.args.val_type == 'depth':
            tgt_img, gt_depth = batch
            tgt_depth = 1.0/self.disp_net(tgt_img)[0]
            errs = compute_errors(gt_depth, tgt_depth)
            errs = {'abs_diff': errs[0], 'abs_rel': errs[1],
                    'a1': errs[6], 'a2': errs[7], 'a3': errs[8]}
            self.log('val_abs_diff', errs['abs_diff'],sync_dist=True,on_epoch=True)
            self.log('val_abs_rel', errs['abs_rel'],sync_dist=True,on_epoch=True)
            self.log('val_a1', errs['a1'], on_epoch=True,sync_dist=True)
            self.log('val_a2', errs['a2'], on_epoch=True,sync_dist=True)
            self.log('val_a3', errs['a3'], on_epoch=True,sync_dist=True)
            self.log("V_loss", errs['abs_rel'], sync_dist=True, prog_bar=True, on_epoch=True)
        else:
            loss,DP_photomatric_loss, D_consistancy_loss,P_consistancy_loss, D_smooth_loss = self.forward(batch)
            self.log("V_DP_photomatric_loss", DP_photomatric_loss.item(),sync_dist=True,prog_bar=True,on_epoch=True)
            self.log("V_D_consistancy_loss", D_consistancy_loss.item(),sync_dist=True,on_epoch=True)
            self.log("V_P_consistancy_loss", P_consistancy_loss.item(),sync_dist=True,on_epoch=True)
            self.log("V_D_smooth_loss", D_smooth_loss.item(),sync_dist=True,on_epoch=True)
            self.log("V_loss", loss,sync_dist=True,prog_bar=True,on_epoch=True)

    def configure_optimizers(self):
        optim_params = [
            {'params': self.pose_net.parameters(), 'lr': self.args.lr},
            {'params': self.disp_net.parameters(), 'lr': self.args.lr}
        ]

        self.optimizer = torch.optim.AdamW(optim_params,
                                           betas=(self.args.momentum, self.args.beta),
                                           weight_decay=self.args.weight_decay)

        self.scheduler = LR_Scheduler('cos',
                                      base_lr=self.args.lr,
                                      num_epochs=self.args.epochs,
                                      iters_per_epoch=len(self.train_dataloader()),
                                      warmup_epochs=self.args.warmup)
        return self.optimizer

    def train_dataloader(self):
        train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomScaleCrop(),
            T.ArrayToTensor(),
            self.normalize
        ])
        train_set = Kitti_Seq(
            self.args.data_dir,
            transform=train_transform,
            train=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)
        return train_loader

    def val_dataloader(self):
        valid_transform = T.Compose([T.ArrayToTensor(),
                                     self.normalize])
        if self.args.val_type=='depth':
            val_set =ValidationSet(self.args.data_dir,
            transform=valid_transform)
        else:
            val_set = Kitti_Seq(
                self.args.data_dir,
                transform=valid_transform,
                train=False,
            )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)
        return val_loader

    def test_dataloader(self):
        return self.val_dataloader()

