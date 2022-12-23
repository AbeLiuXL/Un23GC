import torch.utils.data as data
import numpy as np
from path import Path
import random
# from PIL import Image
from imageio import imread
class Kitti_Seq(data.Dataset):
    def __init__(self, root, seed=0, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders(sequence_length)

    def crawl_folders(self, sequence_length):
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length, demi_length + 1))
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length):
                sample = {'intrinsics': intrinsics, 'imgs': []}
                for j in shifts:
                    sample['imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.array(imread(img).astype(np.float32)) for img in sample['imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform(imgs, np.copy(sample['intrinsics']))
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return imgs, intrinsics

    def __len__(self):
        return len(self.samples)

if __name__ == '__main__':
    import utils.custom_transforms as custom_transforms
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()
    ])
    root_dir ="../../Data/Kitti/sync/kitti_256/"
    data_set=Kitti_Seq(
        root_dir,
        transform=train_transform,
        train=True
    )

