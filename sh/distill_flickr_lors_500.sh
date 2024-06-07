FILE_NAME=$(basename -- "$0")
EXP_NAME="${FILE_NAME%.*}"

export CUDA_VISIBLE_DEVICES=$1;
python distill_tesla_lors.py --dataset=flickr \
    --buffer_path='buffer/flickr/nfnet_bert/InfoNCE' --image_encoder=nfnet --text_encoder=bert \
    --syn_steps=8 --expert_epochs=1 --max_start_epoch=3 \
    --lr_img=1000 --lr_txt=1000 --lr_lr=1e-2 \
    --lr_teacher_img 0.1 --lr_teacher_txt 0.1 \
    --lr_sim=100 --sim_type lowrank --sim_rank 20 --alpha 0.01 \
    --num_queries 499 --mini_batch_size=40 \
    --loss_type WBCE --name ${EXP_NAME} \
    --eval_it 300