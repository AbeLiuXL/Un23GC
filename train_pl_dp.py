import os
import numpy as np
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from modules.module_dp import DP_Module
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.strategies import DDPStrategy,DDPShardedStrategy

def main(args):
    seed_everything(0)
    checkpoint_dir = os.path.join('checkpoints', args.save_name)
    logger = TensorBoardLogger(checkpoint_dir)
    logger.log_hyperparams(args)
    checkpoint = ModelCheckpoint(monitor="V_loss", mode="min", save_top_k=1, save_last=True,
                                 dirpath=checkpoint_dir,
                                 filename='{epoch}-{step}-{V_loss:.3f}-{T_loss:.3f}')
    module = DP_Module(args=args)
    # if args.epoch_size is not None:
    #     g_len=len(args.gpu_id.split(','))
    #     limit_train_batches=args.epoch_size // g_len
    # else:
    #     limit_train_batches=None
    trainer = Trainer(
        # fast_dev_run=bool(args.dev),
        logger=logger,
        gpus=-1,
        # deterministic=True,
        log_every_n_steps=1,
        max_epochs=args.epochs,
        callbacks=[checkpoint],
        # precision=args.precision,
        # strategy=DDPPlugin(find_unused_parameters=True),
        # strategy=DDPStrategy(find_unused_parameters=True),
        strategy='ddp',
        # amp_backend=args.amp_backend,
        # amp_level = args.amp_level,
        limit_train_batches=args.epoch_size,
        # reload_dataloaders_every_n_epochs=1,
        # limit_val_batches=16,
        # limit_test_batches=16,
        # plugins='ddp_sharded'
        # plugins = DDPPlugin(find_unused_parameters=False),
        # plugins="deepspeed_stage_2",
        # stochastic_weight_avg = False,
        # num_nodes=2,
        # replace_sampler_ddp = replace_sampler_ddp
        benchmark=True
    )
    if args.pretrained!=None:
        # state_dict = os.path.join(args.data_type,
        #     "models", "state_dicts", args.classifier + ".ckpt"
        # )
        print('pretrained')
        state_dict = args.pretrained
        # model.model.load_state_dict(torch.load(state_dict))
        module = module.load_from_checkpoint(state_dict, args=args)
    trainer.fit(module)

if __name__ == '__main__':
    parser = ArgumentParser()

    ######Dataset#######
    parser.add_argument("--data_dir", type=str, default="../Data/Kitti/sync/kitti_256/")
    parser.add_argument("--save_name", type=str, default="dp_test")
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        metavar='N', help='mini-batch size')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument("--val_type", type=str, default="depth")
    ######Optimizer#######
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight_decay', default=0, type=float,
                        metavar='N', help='weight decay')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epoch_size', default=1000, type=int, metavar='N',
                        help='manual epoch size (will match dataset size if not set)')
    parser.add_argument("--warmup", type=int, default=5)
    ######Losses#######
    parser.add_argument('--dppw', type=float, metavar='W', default=1)
    parser.add_argument('--dcw', type=float, metavar='W', default=0.1)
    parser.add_argument('--pcw', type=float, metavar='W', default=0.01)
    parser.add_argument('--dsw', type=float, metavar='W', default=0.1)
    parser.add_argument("--auto_mask", type=int, default=1, choices=[0, 1])
    parser.add_argument("--num_scales", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--num_layers", type=int, default=18, choices=[18, 50, 101])

    ##########Trainer########
    parser.add_argument("--pretrained", type=str, default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    main(args)