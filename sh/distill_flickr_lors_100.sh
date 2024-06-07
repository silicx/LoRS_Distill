FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"

export CUDA_VISIBLE_DEVICES=$1;
python distill_tesla_lors.py --dataset=flickr \
    --buffer_path='buffer/flickr/nfnet_bert/InfoNCE' --image_encoder=nfnet --text_encoder=bert \
    --syn_steps=8 --expert_epochs=1 --max_start_epoch=2 \
    --lr_img=100 --lr_txt=100 --lr_lr=1e-2 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim=10.0 --sim_type lowrank --sim_rank 10 --alpha 3 \
    --num_queries 99 --mini_batch_size=20 \
    --loss_type WBCE --name ${EXP_NAME}