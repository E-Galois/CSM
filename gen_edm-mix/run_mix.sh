torchrun --standalone --nproc_per_node=4 edm_mix.py \
--outdir=out_c10_mix \
--seeds=d8,0-19979 \
--batch=512 \
--steps=20 \
--class_num=10 \
--network=./pretrained/edm-cifar10-32x32-cond-vp.pkl

# torchrun --standalone --nproc_per_node=4 edm_mix.py \
# --outdir=out_c100_mix \
# --seeds=d8,0-19799 \
# --batch=512 \
# --steps=25 \
# --class_num=100 \
# --network=./pretrained/edm-cifar100-32x32-cond-vp.pkl

# torchrun --standalone --nproc_per_node=4 edm_mix.py \
# --outdir=out_tiny_mix \
# --seeds=d8,0-39799 \
# --batch=256 \
# --steps=25 \
# --class_num=200 \
# --network=./pretrained/edm-tiny-64x64-cond-vp.pkl