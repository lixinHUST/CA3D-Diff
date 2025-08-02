import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave

from ldm.models.diffusion.main_models import SyncDDIMSampler
from ldm.util import instantiate_from_config, prepare_inputs
from torchvision import transforms
import os
from PIL import Image
from einops import rearrange

def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/syncdreamer-step80k.ckpt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)

    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=2)
    parser.add_argument('--seed', type=int, default=6033)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--sampler', type=str, default='ddim')
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--cc2mlo', action='store_true')
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)
    
    model = load_model(flags.cfg, flags.ckpt, strict=True)
    model = model.to(flags.device)
    print(f"Actual type: {type(model)}")  # 查看真实类型

    # prepare data
    names=os.listdir(flags.input)
    print('test data:',len(names))
    save_dir=flags.ckpt.replace('a100','lixin').replace('checkpoint','test_results').replace('.ckpt',f'_scale{flags.cfg_scale}')
    
    image_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # 保持 128x128 大小
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        ])
    
    if flags.sampler=='ddim':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    else:
        raise NotImplementedError
    B=flags.batch_size
    if flags.cc2mlo:
        save_dir+='_cc2mlo'
        target_index = torch.zeros((B, 1), device=flags.device).long()
    else:
        save_dir+='_mlo2cc'
        target_index = torch.ones((B, 1), device=flags.device).long()

    print('save_dir:',save_dir)
    os.makedirs(save_dir, exist_ok=True)
    n=len(names)
    for i in range(0,n,B):
        target_images=[]
        for j in range(i,min(i+B,n)):
            cc_img_path = os.path.join(flags.input, names[j])
            cc_image= image_transform(Image.open(cc_img_path).convert("RGB")).to(flags.device)
            mlo_img_path = cc_img_path.replace('CC','MLO')
            mlo_image= image_transform(Image.open(mlo_img_path).convert("RGB")).to(flags.device)
            target_image=torch.stack([cc_image,mlo_image], 0)
            target_images.append(target_image)
        
        data= {"target_image":  torch.stack(target_images, dim=0)}
        if data['target_image'].shape[0] != B:
            target_index=target_index[:data['target_image'].shape[0]]
        
        # import time
        # st_time=time.time()
        x_sample,_ = model.sample(sampler, data, flags.cfg_scale, flags.batch_view_num ,target_index)
        # memory=torch.cuda.max_memory_allocated() / (1024.0 ** 3)
        # print(f"{time.time() - st_time} s, {memory} GB")

        x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
        x_sample = x_sample.permute(0,2,3,1).cpu().numpy() * 255
        x_sample = x_sample.astype(np.uint8)
        for k in range(x_sample.shape[0]):
            name = names[i+k]
            pred_mlo= Image.fromarray(x_sample[k]).convert('L')
            if flags.cc2mlo:
                pred_mlo.save(os.path.join(save_dir, name.replace('CC', 'MLO')))
            else:
                pred_mlo.save(os.path.join(save_dir, name))

if __name__=="__main__":
    main()

