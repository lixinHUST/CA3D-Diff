from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.io import imsave
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ldm.base_utils import read_pickle, concat_images_list
from ldm.models.diffusion.sync_dreamer_network import NoisyTargetViewEncoder, SpatialTime3DNet, FrustumTV3DNet, FrustumTV2DNet
from ldm.modules.diffusionmodules.util import make_ddim_timesteps, timestep_embedding
from ldm.util import instantiate_from_config
from torchvision.utils import make_grid
from einops import rearrange, repeat

def project_cc(coords_3d):
    return coords_3d[..., :2]

def project_mlo(coords_3d):
    theta = np.radians(45)
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ], dtype=torch.float32, device=coords_3d.device)
    
    coords_rotated = torch.einsum('...i,ji->...j', coords_3d, rotation_matrix)
    
    return coords_rotated[..., :2]

def normalize_coords(coords_2d):
    max_val = coords_2d.max(dim=1, keepdim=True).values
    min_val = coords_2d.min(dim=1, keepdim=True).values
    return (coords_2d - min_val) / (max_val - min_val + 1e-5) * 2 - 1

def get_2d_coords(volume_3d):
    B, _, V, _, _ = volume_3d.shape
    coords_3d = volume_3d.permute(0, 2, 3, 4, 1).reshape(B, -1, 3)
    coords_cc = normalize_coords(project_cc(coords_3d))
    coords_mlo = normalize_coords(project_mlo(coords_3d))
    return coords_cc.view(B, V**3, 1, 2), coords_mlo.view(B, V**3, 1, 2)

import math

def project_volume_to_plane(spatial_volume, target_index):
    B, C, V, _, _ = spatial_volume.shape
    device = spatial_volume.device

    result = []

    for b in range(B):
        vol = spatial_volume[b]  # [C, V, V, V]
        view = target_index[b, 0].item()

        if view == 1:  
            proj = vol.mean(dim=3)  # [C, V, V]

        elif view == 0:  
            theta = torch.tensor(45.0 * 3.1415926 / 180.0, device=device)
            rot_matrix = torch.tensor([
                [1, 0, 0],
                [0, torch.cos(theta), -torch.sin(theta)],
                [0, torch.sin(theta),  torch.cos(theta)],
            ], device=device)  # [3,3]

            lin_coords = torch.linspace(-1, 1, V, device=device)
            grid_z, grid_y, grid_x = torch.meshgrid(lin_coords, lin_coords, lin_coords, indexing='ij')  # [V,V,V]
            coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # [V, V, V, 3]

            coords_flat = coords.reshape(-1, 3) @ rot_matrix.T  # [V^3, 3]
            coords_rotated = coords_flat.reshape(V, V, V, 3)  # [V, V, V, 3]

            # Grid sample 的输入
            coords_rotated = coords_rotated.unsqueeze(0)  # [1, V, V, V, 3]
            vol_input = vol.unsqueeze(0)  # [1, C, V, V, V]
            rotated_vol = F.grid_sample(vol_input, coords_rotated, mode='bilinear', align_corners=True)  # [1,C,V,V,V]
            rotated_vol = rotated_vol[0]  # [C,V,V,V]

            proj = rotated_vol.mean(dim=3)  # [C, V, V]

        else:
            raise ValueError("target_index must be 0 (MLO) or 1 (CC)")

        result.append(proj)

    return torch.stack(result, dim=0)  # [B, C, V, V]


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def disable_training_module(module: nn.Module):
    module = module.eval()
    module.train = disabled_train
    for para in module.parameters():
        para.requires_grad = False
    return module

def repeat_to_batch(tensor, B, VN):
    t_shape = tensor.shape
    ones = [1 for _ in range(len(t_shape)-1)]
    tensor_new = tensor.view(B,1,*t_shape[1:]).repeat(1,VN,*ones).view(B*VN,*t_shape[1:])
    return tensor_new

class UNetWrapper(nn.Module):
    def __init__(self, diff_model_config, drop_conditions=False, drop_scheme='default', use_zero_123=True):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.drop_conditions = drop_conditions
        self.drop_scheme=drop_scheme
        self.use_zero_123 = use_zero_123


    def drop(self, cond, mask):
        shape = cond.shape
        B = shape[0]
        cond = mask.view(B,*[1 for _ in range(len(shape)-1)]) * cond
        return cond

    def get_trainable_parameters(self):
        return self.diffusion_model.get_trainable_parameters()

    def get_drop_scheme(self, B, device):
        if self.drop_scheme=='default':
            random = torch.rand(B, dtype=torch.float32, device=device)
            drop_clip = (random > 0.15) & (random <= 0.2)
            drop_volume = (random > 0.1) & (random <= 0.15)
            drop_concat = (random > 0.05) & (random <= 0.1)
            drop_all = random <= 0.05
        else:
            raise NotImplementedError
        return drop_clip, drop_volume, drop_concat, drop_all

    def forward(self, x, t, clip_embed,y, volume_feats, x_concat, is_train=False,mode='crossattn'):
        """

        @param x:             B,4,H,W
        @param t:             B,
        @param clip_embed:    B,M,768
        @param volume_feats:  B,C,D,H,W
        @param x_concat:      B,C,H,W
        @param is_train:
        @return:
        """
        if self.drop_conditions and is_train:
            B = x.shape[0]
            drop_clip, drop_volume, drop_concat, drop_all = self.get_drop_scheme(B, x.device)

            clip_mask = 1.0 - (drop_clip | drop_all).float()
            clip_embed = self.drop(clip_embed, clip_mask)

            volume_mask = 1.0 - (drop_volume | drop_all).float()
            for k, v in volume_feats.items():
                volume_feats[k] = self.drop(v, mask=volume_mask)

            concat_mask = 1.0 - (drop_concat | drop_all).float()
            x_concat = self.drop(x_concat, concat_mask)

        if self.use_zero_123:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_ = x_concat * 1.0
            x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
        else:
            x_concat_ = x_concat

        if mode == 'hybrid':
            x = torch.cat([x, x_concat_], 1)
        pred = self.diffusion_model(x, t, clip_embed,y, source_dict=volume_feats)
        return pred

    def predict_with_unconditional_scale(self, x, t, clip_embed,y, volume_feats, x_concat, unconditional_scale ,mode='crossattn'):
        x_ = torch.cat([x] * 2, 0)
        t_ = torch.cat([t] * 2, 0)
        y_ = torch.cat([y] * 2, 0)
        clip_embed_ = torch.cat([clip_embed, torch.zeros_like(clip_embed)], 0)

        v_ = {}
        for k, v in volume_feats.items():
            v_[k] = torch.cat([v, torch.zeros_like(v)], 0)

        x_concat_ = torch.cat([x_concat, torch.zeros_like(x_concat)], 0)
        if self.use_zero_123:
            # zero123 does not multiply this when encoding, maybe a bug for zero123
            first_stage_scale_factor = 0.18215
            x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
        if mode == 'hybrid':
            x_ = torch.cat([x_, x_concat_], 1)
        s, s_uc = self.diffusion_model(x_, t_, clip_embed_, y_,source_dict=v_).chunk(2)
        s = s_uc + unconditional_scale * (s - s_uc)
        return s


class SpatialVolumeNet(nn.Module):
    def __init__(self, time_dim, view_num,
                 input_image_size=256, frustum_volume_depth=48,
                 spatial_volume_size=32, spatial_volume_length=0.5,
                 frustum_volume_length=0.86603 # sqrt(3)/2
                 ):
        super().__init__()
        self.target_encoder = NoisyTargetViewEncoder(time_dim, output_dim=16)
        self.spatial_volume_feats = SpatialTime3DNet(input_dim=16, time_dim=time_dim, dims=(64, 128, 256, 512))
        self.frustum_volume_feats = FrustumTV2DNet(64, time_dim, dims=(64, 128, 256, 512))

        self.frustum_volume_length = frustum_volume_length
        self.input_image_size = input_image_size
        self.spatial_volume_size = spatial_volume_size
        self.spatial_volume_length = spatial_volume_length

        self.frustum_volume_size = self.input_image_size // 8
        self.frustum_volume_depth = frustum_volume_depth
        self.time_dim = time_dim
        self.default_origin_depth = 1.5 # our rendered images are 1.5 away from the origin, we assume camera is 1.5 away from the origin

    def construct_spatial_volume(self, x, t_embed):
        """
        @param x:            B,N,4,H,W
        @param t_embed:      B,t_dim
        @return:
        """
        B, N, _, H, W = x.shape
        V = self.spatial_volume_size
        L= self.spatial_volume_length
        device = x.device
        
        verts = torch.linspace(-L, L, V, dtype=torch.float32, device=device)
        grid = torch.stack(torch.meshgrid(verts, verts, verts, indexing='ij'), dim=-1)
        volume_verts = grid.permute(3, 0, 1, 2).unsqueeze(0).repeat(B, 1, 1, 1, 1)
        coords_cc, coords_mlo = get_2d_coords(volume_verts)
        
        # encode source features
        t_embed_ = t_embed.view(B, 1, self.time_dim).repeat(1, N, 1).view(B, N, self.time_dim)
        # extract 2D image features
        x_cc = self.target_encoder(x[:, 0], t_embed_[:, 0])
        x_mlo = self.target_encoder(x[:, 1], t_embed_[:, 1])
        C = x_mlo.shape[1]
        feats_cc = F.grid_sample(x_cc, coords_cc, mode='bilinear', align_corners=True).squeeze(-1)
        feats_mlo = F.grid_sample(x_mlo, coords_mlo, mode='bilinear', align_corners=True).squeeze(-1)
        volume = feats_cc + feats_mlo

        spatial_volume_feats = self.spatial_volume_feats(volume.view(B, C, V, V, V), t_embed)  # b,64,32,32,32
        return spatial_volume_feats

    def construct_view_frustum_volume(self, spatial_volume, t_embed, target_indices):
        """
        @param spatial_volume:    B,C,V,V,V
        @param t_embed:           B,t_dim
        @param v_embed:           B,N,v_dim
        @param poses:             N,3,4
        @param Ks:                N,3,3
        @param target_indices:    B,TN
        @return: B*TN,C,H,W
        """        
        proj_feats = project_volume_to_plane(spatial_volume, target_indices)
        volume_feats_dict = self.frustum_volume_feats(proj_feats, t_embed)
        return volume_feats_dict

class SyncMultiviewDiffusion(pl.LightningModule):
    def __init__(self, unet_config, scheduler_config,
                 finetune_unet=False, finetune_projection=True,
                 view_num=16, image_size=256,
                 cfg_scale=3.0, output_num=8, batch_view_num=4,
                 drop_conditions=False, drop_scheme='default',
                 clip_image_encoder_path="/apdcephfs/private_rondyliu/projects/clip/ViT-L-14.pt",
                 sample_type='ddim', sample_steps=50,conditioning_key='hybrid',conditioning_type=''):
        super().__init__()

        self.finetune_unet = finetune_unet
        self.finetune_projection = finetune_projection

        self.view_num = view_num
        self.viewpoint_dim = 4
        self.output_num = output_num
        self.image_size = image_size

        self.batch_view_num = batch_view_num
        self.cfg_scale = cfg_scale

        self.clip_image_encoder_path = clip_image_encoder_path

        self._init_time_step_embedding()
        self._init_first_stage()
        self._init_schedule()

        self.spatial_volume = SpatialVolumeNet(self.time_embed_dim, self.view_num)
        self.model = UNetWrapper(unet_config, drop_conditions=drop_conditions, drop_scheme=drop_scheme)
        self.scheduler_config = scheduler_config
        self.conditioning_key = conditioning_key
        self.conditioning_type = conditioning_type

        latent_size = image_size//8
        if sample_type=='ddim':
            self.sampler = SyncDDIMSampler(self, sample_steps , "uniform", 1.0, latent_size=latent_size)
        else:
            raise NotImplementedError
                
    def _init_first_stage(self):
        first_stage_config={
            "target": "ldm.models.autoencoder.AutoencoderKL",
            "params": {
                "embed_dim": 4,
                "monitor": "val/rec_loss",
                "ddconfig":{
                  "double_z": True,
                  "z_channels": 4,
                  "resolution": self.image_size,
                  "in_channels": 3,
                  "out_ch": 3,
                  "ch": 128,
                  "ch_mult": [1,2,4,4],
                  "num_res_blocks": 2,
                  "attn_resolutions": [],
                  "dropout": 0.0
                },
                "lossconfig": {"target": "torch.nn.Identity"},
            }
        }
        self.first_stage_scale_factor = 0.18215
        self.first_stage_model = instantiate_from_config(first_stage_config)
        self.first_stage_model = disable_training_module(self.first_stage_model)

    def _init_schedule(self):
        self.num_timesteps = 1000
        linear_start = 0.00085
        linear_end = 0.0120
        num_timesteps = 1000
        betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2 # T
        assert betas.shape[0] == self.num_timesteps

        # all in float64 first
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0) # T
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # T
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_log_variance_clipped = torch.clamp(posterior_log_variance_clipped, min=-10)

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float())
        self.register_buffer("posterior_variance", posterior_variance.float())
        self.register_buffer('posterior_log_variance_clipped', posterior_log_variance_clipped.float())

    def _init_time_step_embedding(self):
        self.time_embed_dim = 256
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.SiLU(True),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )

    def encode_first_stage(self, x, sample=True):
        with torch.no_grad():
            posterior = self.first_stage_model.encode(x)  # b,4,h//8,w//8
            if sample:
                return posterior.sample().detach() * self.first_stage_scale_factor
            else:
                return posterior.mode().detach() * self.first_stage_scale_factor

    def decode_first_stage(self, z):
        with torch.no_grad():
            z = 1. / self.first_stage_scale_factor * z
            return self.first_stage_model.decode(z)

    def prepare(self, batch, target_index=None):

        if 'target_image' in batch:
            image_target = batch['target_image'].permute(0, 1, 4, 2, 3) # b,n,3,h,w
            N = image_target.shape[1]
            x = [self.encode_first_stage(image_target[:,ni], True) for ni in range(N)]
            x = torch.stack(x, 1) # b,n,4,h//8,w//8
            B= x.shape[0]
            image_input = image_target[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        else:
            x = None
        
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        uncond = 0.05
        random = torch.rand(x.size(0), device=x.device)
        prompt_mask = 1 - rearrange((random < 2 * uncond).float(), "n -> n 1 1 1")
        input_mask = 1 - rearrange((random >= uncond).float() * (random < 3 * uncond).float(), "n -> n 1 1 1")
        
        x_input = self.encode_first_stage(image_input)*input_mask
        image_target = image_target[torch.arange(B)[:,None],(1-target_index)][:,0] # B,4,H,W

        input_info = {'image': image_input, 'x': x_input,'target_image':image_target}
        with torch.no_grad():
            clip_embed = self.encode_first_stage(image_input)*prompt_mask
        return x, clip_embed, input_info

    def embed_time(self, t):
        t_embed = timestep_embedding(t, self.time_embed_dim, repeat_only=False) # B,TED
        t_embed = self.time_embed(t_embed) # B,TED
        return t_embed

    def get_target_view_feats(self, x_input, spatial_volume, clip_embed, t_embed, target_index):
        """
        @param x_input:        B,4,H,W
        @param spatial_volume: B,C,V,V,V
        @param clip_embed:     B,1,768
        @param t_embed:        B,t_dim
        @param v_embed:        B,N,v_dim
        @param target_index:   B,TN
        @return:
            tensors of size B*TN,*
        """
        B, _, H, W = x_input.shape
        frustum_volume_feats = self.spatial_volume.construct_view_frustum_volume(spatial_volume, t_embed, target_index)

        # clip
        TN = target_index.shape[1]

        clip_embed_ = clip_embed
        y_=target_index
        x_input_ = x_input.unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, 4, H, W)

        x_concat = x_input_
        return clip_embed_, frustum_volume_feats, x_concat, y_

    def training_step(self, batch):
        B = batch['target_image'].shape[0]
        time_steps = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()
        N = self.view_num
        target_index = torch.randint(0, N,(B, 1), device=self.device).long() # B, 1
        x, clip_embed, input_info = self.prepare(batch, target_index)
        x_noisy, noise = self.add_noise(x, time_steps)  # B,N,4,H,W

        t_embed = self.embed_time(time_steps)
        spatial_volume = self.spatial_volume.construct_spatial_volume(x_noisy, t_embed)

        clip_embed, volume_feats, x_concat, y = self.get_target_view_feats(input_info['x'], spatial_volume, clip_embed, t_embed, target_index)

        target_index = 1 - target_index
        x_noisy_ = x_noisy[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        noise_predict = self.model(x_noisy_, time_steps, clip_embed,y, volume_feats, x_concat, is_train=True,mode=self.conditioning_key) # B,4,H,W

        noise_target = noise[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        loss_simple = torch.nn.functional.mse_loss(noise_target, noise_predict, reduction='none')
        
        fg_loss = loss_simple.mean()
        
        self.log('sim', loss_simple.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)

        # log others
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return fg_loss

    def add_noise(self, x_start, t,mask=None):
        """
        @param x_start: B,*
        @param t:       B,
        @return:
        """
        B = x_start.shape[0]
        noise = torch.randn_like(x_start) # B,*

        sqrt_alphas_cumprod_  = self.sqrt_alphas_cumprod[t] # B,
        sqrt_one_minus_alphas_cumprod_ = self.sqrt_one_minus_alphas_cumprod[t] # B
        sqrt_alphas_cumprod_ = sqrt_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        sqrt_one_minus_alphas_cumprod_ = sqrt_one_minus_alphas_cumprod_.view(B, *[1 for _ in range(len(x_start.shape)-1)])
        x_noisy = sqrt_alphas_cumprod_ * x_start + sqrt_one_minus_alphas_cumprod_ * noise
        if mask is not None:
            x_noisy = x_noisy * mask
            noise = noise * mask
            
        return x_noisy, noise

    def sample(self, sampler, batch, cfg_scale, batch_view_num,target_index, return_inter_results=False, inter_interval=50, inter_view_interval=2):
        
        _, clip_embed, input_info = self.prepare(batch,target_index)
        
        x_sample, inter = sampler.sample(input_info, clip_embed, unconditional_scale=cfg_scale, log_every_t=inter_interval, 
                                         batch_view_num=batch_view_num,target_indices=target_index)

        x_sample = self.decode_first_stage(x_sample)
        if return_inter_results:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            inter = torch.stack(inter['x_inter'], 1) # # B,T,C,H,W
            B,T,C,H,W = inter.shape
            inter_results = []
            for ti in range(T):
                inter_results.append(self.decode_first_stage(inter[:, ti]))
            inter_results = torch.stack(inter_results,1) # B,T,3,H,W
            return x_sample, inter_results
        else:
            return x_sample, input_info

    def log_image(self,  x_sample, input_info, step, output_dir):
        process = lambda x: ((torch.clip(x, min=-1, max=1).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        B = x_sample.shape[0]
        clip_inputs= input_info['image']
        targets=input_info['target_image']
        
        image_cond = []
        for bi in range(8):
            img_pr_ = concat_images_list(process(clip_inputs[bi].permute(1, 2, 0)),*[process(targets[bi].permute(1, 2, 0)),process(x_sample[bi].permute(1, 2, 0))])
            image_cond.append(img_pr_)

        output_dir = Path(output_dir)
        imsave(str(output_dir/f'{step}.jpg'), concat_images_list(*image_cond, vert=True))
        grid = make_grid(x_sample, nrow=8,normalize=True)  # or whatever your layout is
        self.logger.experiment.add_image("Images/samples", grid, global_step=self.global_step)
        grid = make_grid(clip_inputs, nrow=8,normalize=True)  # or whatever your layout is
        self.logger.experiment.add_image("Images/inputs", grid, global_step=self.global_step)
        grid = make_grid(targets, nrow=8,normalize=True)  # or whatever your layout is
        self.logger.experiment.add_image("Images/targets", grid, global_step=self.global_step)


    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx==0 and self.global_rank==0:
            self.eval()
            step = self.global_step
            batch_ = {}
            for k, v in batch.items(): batch_[k] = v[:self.output_num]
            B = batch_['target_image'].shape[0]
            target_index = torch.randint(0, 2,(B, 1), device=self.device).long() # B, 1
            x_sample, input_info = self.sample(self.sampler, batch_, self.cfg_scale, self.batch_view_num,target_index)
            output_dir = Path(self.image_dir) / 'images' / 'val'
            output_dir.mkdir(exist_ok=True, parents=True)
            self.log_image(x_sample, input_info, step, output_dir=output_dir)

    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'setting learning rate to {lr:.4f} ...')
        paras = []
        if self.finetune_unet:
            paras.append({"params": self.model.parameters(), "lr": lr},)
        else:
            paras.append({"params": self.model.get_trainable_parameters(), "lr": lr},)

        paras.append({"params": self.time_embed.parameters(), "lr": lr*10.0},)
        paras.append({"params": self.spatial_volume.parameters(), "lr": lr*10.0},)

        opt = torch.optim.AdamW(paras, lr=lr)

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [{'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]
        return [opt], scheduler

class SyncDDIMSampler:
    def __init__(self, model: SyncMultiviewDiffusion, ddim_num_steps, ddim_discretize="uniform", ddim_eta=1.0, latent_size=32):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.latent_size = latent_size
        self._make_schedule(ddim_num_steps, ddim_discretize, ddim_eta)
        self.eta = ddim_eta

    def _make_schedule(self,  ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose) # DT
        ddim_timesteps_ = torch.from_numpy(self.ddim_timesteps.astype(np.int64)) # DT

        alphas_cumprod = self.model.alphas_cumprod # T
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        self.ddim_alphas = alphas_cumprod[ddim_timesteps_].double() # DT
        self.ddim_alphas_prev = torch.cat([alphas_cumprod[0:1], alphas_cumprod[ddim_timesteps_[:-1]]], 0) # DT
        self.ddim_sigmas = ddim_eta * torch.sqrt((1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) * (1 - self.ddim_alphas / self.ddim_alphas_prev))

        self.ddim_alphas_raw = self.model.alphas[ddim_timesteps_].float() # DT
        self.ddim_sigmas = self.ddim_sigmas.float()
        self.ddim_alphas = self.ddim_alphas.float()
        self.ddim_alphas_prev = self.ddim_alphas_prev.float()
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1. - self.ddim_alphas).float()


    @torch.no_grad()
    def denoise_apply_impl(self, x_target_noisy, index, noise_pred, is_step0=False):
        """
        @param x_target_noisy: B,N,4,H,W
        @param index:          index
        @param noise_pred:     B,N,4,H,W
        @param is_step0:       bool
        @return:
        """
        device = x_target_noisy.device

        # apply noise
        a_t = self.ddim_alphas[index].to(device).float().view(1,1,1,1)
        a_prev = self.ddim_alphas_prev[index].to(device).float().view(1,1,1,1)
        sqrt_one_minus_at = self.ddim_sqrt_one_minus_alphas[index].to(device).float().view(1,1,1,1)
        sigma_t = self.ddim_sigmas[index].to(device).float().view(1,1,1,1)

        pred_x0 = (x_target_noisy - sqrt_one_minus_at * noise_pred) / a_t.sqrt()
        dir_xt = torch.clamp(1. - a_prev - sigma_t**2, min=1e-7).sqrt() * noise_pred
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt
        if not is_step0:
            noise = sigma_t * torch.randn_like(x_target_noisy)
            x_prev = x_prev + noise
        return x_prev

    @torch.no_grad()
    def denoise_apply(self, x_target_noisy, input_info, clip_embed, time_steps, index, unconditional_scale, batch_view_num=1, is_step0=False,target_indices=None):
        """
        @param x_target_noisy:   B,4,H,W
        @param input_info:
        @param clip_embed:       B,M,768
        @param time_steps:       B,
        @param index:            int
        @param unconditional_scale:
        @param batch_view_num:   int
        @param is_step0:         bool
        @return:
        """
        x_input = input_info['x']
        image_input = self.model.encode_first_stage(input_info['image'])
        B, C, H, W = x_target_noisy.shape

        x_noisy,_ = self.model.add_noise((image_input), time_steps)
        stacked = torch.stack([x_noisy, x_target_noisy], dim=1)  # [B, 2, C, H, W]
        stacked_rev = torch.stack([x_target_noisy, x_noisy], dim=1)
        if target_indices is None:
            target_indices = torch.zeros((B,1),device=x_target_noisy.device) # N
        x_stack=torch.where(target_indices.view(-1,1,1,1,1) == 0, stacked, stacked_rev)

        # construct source data
        t_embed = self.model.embed_time(time_steps)  # B,t_dim
        spatial_volume = self.model.spatial_volume.construct_spatial_volume(x_stack, t_embed)

        e_t = []
        time_steps_ = repeat_to_batch(time_steps, B, 1)
        clip_embed_, volume_feats_, x_concat_ ,y_= self.model.get_target_view_feats(x_input, spatial_volume, clip_embed, t_embed, target_indices)
        if unconditional_scale!=1.0:
            noise = self.model.model.predict_with_unconditional_scale(x_target_noisy, time_steps_, clip_embed_,y_, volume_feats_, x_concat_, unconditional_scale,mode=self.model.conditioning_key)
        else:
            noise = self.model.model(x_target_noisy, time_steps_, clip_embed_, y_, volume_feats_, x_concat_, is_train=False, mode=self.model.conditioning_key)
        e_t.append(noise)

        e_t = torch.cat(e_t, 1)
        x_prev = self.denoise_apply_impl(x_target_noisy, index, e_t, is_step0)
        return x_prev

    @torch.no_grad()
    def sample(self, input_info, clip_embed,unconditional_scale=1.0, log_every_t=50, batch_view_num=1,target_indices=None):
        """
        @param input_info:      x, elevation
        @param clip_embed:      B,M,768
        @param unconditional_scale:
        @param log_every_t:
        @param batch_view_num:
        @return:
        """
        print(f"unconditional scale {unconditional_scale:.1f}")
        C, H, W = 4, self.latent_size, self.latent_size
        B = clip_embed.shape[0]
        N = self.model.view_num
        device = self.model.device
        x_target_noisy = torch.randn([B, C, H, W], device=device)

        timesteps = self.ddim_timesteps
        intermediates = {'x_inter': []}
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)
            x_target_noisy = self.denoise_apply(x_target_noisy, input_info, clip_embed, time_steps, index, unconditional_scale, batch_view_num=batch_view_num, 
                                                is_step0=index==0,target_indices=target_indices)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_target_noisy)

        return x_target_noisy, intermediates