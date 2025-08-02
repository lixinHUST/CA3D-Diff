python generate.py \
    --cfg configs/train.yaml \
    --ckpt ckpt/train/step=00039999.ckpt \
    --input pair_images/CC/test \
    --cfg_scale 3.0 \
    --device cuda:0 \
    --batch_size 8 \
    --cc2mlo \