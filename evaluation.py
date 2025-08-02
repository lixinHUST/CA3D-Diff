import lpips
import torch
from PIL import Image
from torchvision import transforms

import os
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import numpy as np
import torch
from PIL import Image

def MAE(img1, img2):
    mae = np.mean( abs(img1 - img2)  )
    return mae  

def compute_fid(data_folder1, data_folder2, device='cuda:0'):
    from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
    feat_model = build_feature_extractor("clean", device, use_dataparallel=False)
    ref_features = get_folder_features(data_folder1, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device(device),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
    a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
    
    gen_features = get_folder_features(data_folder2, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device(device),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
    ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
    print(f"fid={score_fid_a2b:.4f}")


def compute_metrics(real_MLO_dir, result_dir, device="cuda:0"):
    print('real_MLO_dir:', real_MLO_dir)
    print('result_dir:', result_dir)
    compute_fid(real_MLO_dir, result_dir, device)

    psnr_sum, ssim_sum,mae_sum,lpips_sum = 0,0,0,0
    nums = 0
    names=os.listdir(real_MLO_dir)
    for name in names:
        mlo_real_img=np.array(Image.open(os.path.join(real_MLO_dir,name)).convert('L').resize((256,256)))
        mlo_fake_img=np.array(Image.open(os.path.join(result_dir,name)).convert('L').resize((256,256)))
        psnr=PSNR(mlo_real_img,mlo_fake_img)
        ssim=SSIM(mlo_real_img,mlo_fake_img)
        mae=MAE(mlo_real_img,mlo_fake_img)
        mlo_real_img_tensor=torch.from_numpy(mlo_real_img).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float()/255.0
        mlo_fake_img_tensor=torch.from_numpy(mlo_fake_img).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).float()/255.0
        lpips1=lpips_model(mlo_real_img_tensor.to(device),mlo_fake_img_tensor.to(device)).item()
        psnr_sum+=psnr
        ssim_sum+=ssim
        mae_sum+=mae
        lpips_sum+=lpips1
        nums+=1
    print('psnr:',psnr_sum/nums)
    print('ssim:',ssim_sum/nums)
    print('mae:',mae_sum/nums)
    print('lpips:',lpips_sum/nums)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='vgg').to(device)

result_dir=''
real_MLO_dir=''
compute_metrics(real_MLO_dir,result_dir, device)