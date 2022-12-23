import pytorch_lightning as pl
import torch
from models.pwcnet import PWCDCNet
import utils.custom_transforms as T
from datasets.kitti_seq import Kitti_Seq
from utils.lr_scheduler import LR_Scheduler
from losses.losses_f import FlowLosses
class F_Module(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.flow_net=PWCDCNet()
        self.loss_functions=FlowLosses(args)
        self.normalize = T.Normalize()

    def compute_flow(self,flow_net, imgs):
        flows = [flow_net(imgs[0], imgs[1]), flow_net(imgs[0], imgs[2]), flow_net(imgs[1], imgs[2])]
        flows_inv = [flow_net(imgs[1], imgs[0]), flow_net(imgs[2], imgs[0]), flow_net(imgs[2], imgs[1])]
        return flows,flows_inv

    def forward(self, batch):
        imgs,intrinsics=batch
        Flow_Photomatric, Flow_Consistancy, Flow_Smooth=0,0,0
        imgpart = [[0, 1], [0, 2], [1, 2]]
        for i, ip in enumerate(imgpart):
            tgt_img = imgs[ip[0]]
            ref_img = imgs[ip[1]]
            flows = self.flow_net(tgt_img,ref_img)
            flows_inv = self.flow_net(ref_img,tgt_img)
            flow_photomatric, flow_consistancy, flow_smooth=self.loss_functions(tgt_img,ref_img,flows,flows_inv)
            Flow_Photomatric+=flow_photomatric
            Flow_Consistancy+=flow_consistancy
            Flow_Smooth+=flow_smooth
        loss = Flow_Photomatric/3. * self.args.fpw \
               + Flow_Consistancy/3. * self.args.fcw \
               + Flow_Smooth/3. * self.args.fsw
        return loss, Flow_Photomatric, Flow_Consistancy, Flow_Smooth

    def training_step(self, batch, batch_nb):
        # self.scheduler(self.optimizer, batch_nb, self.current_epoch, best_pred=0)
        loss, Flow_Photomatric, Flow_Consistancy, Flow_Smooth = self.forward(batch)
        self.log("T_F_photomatric_loss", Flow_Photomatric.item())
        self.log("T_F_consistancy_loss", Flow_Consistancy.item())
        self.log("T_F_smooth_loss", Flow_Smooth.item())
        self.log("T_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_nb):
        loss, Flow_Photomatric, Flow_Consistancy, Flow_Smooth = self.forward(batch)
        self.log("V_F_photomatric_loss", Flow_Photomatric.item(),sync_dist=True)
        self.log("V_F_consistancy_loss", Flow_Consistancy.item(),sync_dist=True)
        self.log("V_F_smooth_loss", Flow_Smooth.item(),sync_dist=True)
        self.log("V_loss", loss.item(),sync_dist=True)

    def configure_optimizers(self):
        optim_params = [
            {'params': self.flow_net.parameters(), 'lr': self.args.lr}
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

