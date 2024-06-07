FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"

export CUDA_VISIBLE_DEVICES=$1;
python distill_tesla_lors.py --dataset=coco \
    --buffer_path='buffer/coco/nfnet_bert/InfoNCE' --image_encoder=nfnet --text_encoder=bert \
    --image_root='distill_utils/data/COCO/' --merge_loss_branches \
    --syn_steps=8 --expert_epochs=1 --max_start_epoch=2 \
    --lr_img=5000 --lr_txt=5000 --lr_lr=1e-2 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim=500 --sim_type lowrank --sim_rank 40 --alpha 1.0 \
    --num_queries 499 --mini_batch_size=30 \
    --temperature 0.1 --no_aug \
    --loss_type WBCE --name ${EXP_NAME}