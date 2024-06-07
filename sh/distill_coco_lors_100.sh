FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"

export CUDA_VISIBLE_DEVICES=$1;
python distill_tesla_lors.py --dataset=coco \
    --buffer_path='buffer/coco/nfnet_bert/InfoNCE' --image_encoder=nfnet --text_encoder=bert \
    --image_root='distill_utils/data/COCO/' --merge_loss_branches \
    --syn_steps=8 --expert_epochs=1 --max_start_epoch=2 \
    --lr_img=1000 --lr_txt=1000 --lr_lr=1e-2 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim=5.0 --sim_type lowrank --sim_rank 10 --alpha 1.0 \
    --num_queries 99 --mini_batch_size=20 \
    --loss_type WBCE --name ${EXP_NAME}
