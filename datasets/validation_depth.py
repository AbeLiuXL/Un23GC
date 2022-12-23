import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import torch
def crawl_folders(folders_list):
    imgs = []
    depth = []
    for folder in folders_list:
        current_imgs = sorted(folder.files('*.jpg'))
        current_depth = []
        for img in current_imgs:
            d = img.dirname() / (img.name[:-4] + '.npy')
            assert (d.isfile()), "depth file {} not found".format(str(d))
            depth.append(d)
        imgs.extend(current_imgs)
        depth.extend(current_depth)
    return imgs, depth


class ValidationSet(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1]
                       for folder in open(scene_list_path)]
        self.transform = transform
        self.imgs, self.depth = crawl_folders(self.scenes)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)
        depth = np.load(self.depth[index]).astype(np.float32)
        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]
        return img, depth

    def __len__(self):
        return len(self.imgs)
if __name__ == '__main__':
    import utils.custom_transforms as custom_transforms

    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()
    ])
    root_dir = "../../Data/Kitti/sync/kitti_256/"
    data_set = ValidationSet(
        root_dir,
        transform=train_transform)