conda activate CA3D-Diff
python train.py -b configs/train.yaml \
                           --finetune_from ckpt/sd-image-conditioned-v2.ckpt \
                           -l ckpt/log  \
                           -c ckpt/checkpoint \
                           --gpus 0,