import os
import numpy as np
import PIL.Image
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from datatransforms import *
import os
from torch.utils import data

class depthDataset(Dataset):
    """
    load data in a folder
    """
    def __init__(self, root, transform=None):

        self.root = root
        self.transform = transform

        img_root = os.path.join(self.root, 'image')
        lbl_root = os.path.join(self.root, 'depth')
        focal_root = os.path.join(self.root, 'focal')

        file_names = os.listdir(img_root)
        self.img_names = []
        self.lbl_names = []
        self.focal_names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.lbl_names.append(
                os.path.join(lbl_root, name[:-4]+'.png')
            )
            self.img_names.append(
                os.path.join(img_root, name)
            )
            self.focal_names.append(
                os.path.join(focal_root, name[:-4]+'.mat')
            )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # load image
        img_file = self.img_names[index]
        img = PIL.Image.open(img_file)
        image = img.resize((256, 256))
        lbl_file = self.lbl_names[index]
        lbl = PIL.Image.open(lbl_file)
        depth = lbl.resize((256,256))
        focal_file = self.focal_names[index]
        focal = sio.loadmat(focal_file)
        focal = focal['img']
        sample = {'image': image, 'depth': depth,'focal':focal}

        if self.transform:
            sample = self.transform(sample)
        return sample


class testdepthDataset(Dataset):
        """
        load data in a folder
        """
        def __init__(self, root, transform=None):

            self.root = root
            self.transform = transform

            img_root = os.path.join(self.root, 'image')
            lbl_root = os.path.join(self.root, 'depth')
            focal_root = os.path.join(self.root, 'focal')

            file_names = os.listdir(img_root)
            self.img_names = []
            self.lbl_names = []
            self.names = []
            self.focal_names = []
            for i, name in enumerate(file_names):
                #if not name.endswith('.png'):
                if not name.endswith('.jpg'):
                    continue
                self.lbl_names.append(
                    os.path.join(lbl_root, name[:-4] + '.png')
                )
                self.img_names.append(
                    os.path.join(img_root, name)
                )
                self.focal_names.append(
                    os.path.join(focal_root, name[:-4] + '.mat')
                )
                self.names.append(name[:-4])

        def __len__(self):
            return len(self.img_names)

        def __getitem__(self, index):
            # load image
            img_file = self.img_names[index]
            img = PIL.Image.open(img_file)
            image = img.resize((256, 256))
            img_size = img.size
            lbl_file = self.lbl_names[index]
            lbl = PIL.Image.open(lbl_file)
            depth = lbl.resize((256, 256))
            focal_file = self.focal_names[index]
            focal = sio.loadmat(focal_file)
            focal = focal['img']
            sample = {'image': image, 'depth': depth, 'focal': focal}
            if self.transform:
                sample = self.transform(sample)
            return sample,self.names[index],img_size

def getTrainingData(train_dataRoot,batch_size=64):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140],[-0.5836, -0.6948, 0.4203]]),
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transformed_training = depthDataset(train_dataRoot,
                                        transform=transforms.Compose([
                                            RandomRotate(5),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std'])
                                        ]))
    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=0, pin_memory=False)
    return dataloader_training


def getTestingData(test_dataRoot,batch_size=64):
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    transformed_testing = testdepthDataset(test_dataRoot,
                                       transform=transforms.Compose([
                                           ToTensor(is_test=True),
                                           Normalize(__imagenet_stats['mean'],
                                                     __imagenet_stats['std'])
                                       ]))
    dataloader_testing= DataLoader(transformed_testing, batch_size=1,
                                     shuffle=False, num_workers=0, pin_memory=False)
    return dataloader_testing