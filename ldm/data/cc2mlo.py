import webdataset as wds
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
import pytorch_lightning as pl
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
import os

class CC2MLODataset(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, seed=0, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def train_dataloader(self):
        dataset=CustomDataset(train_val_test='train')
        sampler = DistributedSampler(dataset, seed=self.seed)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        dataset=CustomDataset(train_val_test='test')
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return wds.WebLoader(CustomDataset(train_val_test='test'),\
                          batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class CustomDataset(Dataset):
    def __init__(self,img_dir='/memory/a100/Datasets/zero123',train_val_test='train'):
        self.cc_img_root = os.path.join(img_dir,'pair_images/CC',train_val_test)
        self.images=os.listdir(self.cc_img_root)
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 保持 128x128 大小
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        cc_img_path = os.path.join(self.cc_img_root, self.images[idx])
        mlo_img_path = cc_img_path.replace('CC','MLO')
        cc_image= self.image_transform(Image.open(cc_img_path).convert("RGB"))
        mlo_image= self.image_transform(Image.open(mlo_img_path).convert("RGB"))
        data = {}
        data["target_image"] = torch.stack([cc_image,mlo_image], 0)
        return data
