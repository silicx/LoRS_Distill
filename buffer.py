import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm, trange
from distill_default_utils import load_or_process_file
from src.epoch import epoch, epoch_test, itm_eval
import copy
import wandb
import warnings
import datetime
from data import get_dataset_flickr, textprocess, textprocess_train
from src.networks import CLIPModel_full
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    if args.disabled_wandb == True:
        wandb.init(mode="disabled")
    else:  
        no_aug_suffix = "_NoAug" if args.no_aug else "" 
        wandb.init(project='LoRS-Buffer', config=args, 
                   name=f"{args.dataset}_{args.image_encoder}_{args.text_encoder}_{args.loss_type}{no_aug_suffix}")

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.distributed = torch.cuda.device_count() > 1

    
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.image_encoder+"_"+args.text_encoder, args.loss_type+no_aug_suffix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ''' organize the datasets '''
    trainloader, testloader, train_dataset, test_dataset = get_dataset_flickr(args)

    train_sentences = train_dataset.get_all_captions() 
    _ = load_or_process_file('text', textprocess, args, testloader)
    _ = load_or_process_file('train_text', textprocess_train, args, train_sentences)

    data = np.load(f'{args.dataset}_{args.text_encoder}_text_embed.npz')
    bert_test_embed_loaded = data['bert_test_embed']
    bert_test_embed = torch.from_numpy(bert_test_embed_loaded).cpu()


    img_trajectories = []
    txt_trajectories = []

    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        
        teacher_net = CLIPModel_full(args).to(args.device)
        img_teacher_net = teacher_net.image_encoder.to(args.device)
        txt_teacher_net = teacher_net.text_projection.to(args.device)

        if args.text_trainable:
            txt_teacher_net = teacher_net.text_encoder.to(args.device)
        if args.distributed:
            raise NotImplementedError()
            img_teacher_net = torch.nn.DataParallel(img_teacher_net)
            txt_teacher_net = torch.nn.DataParallel(txt_teacher_net)

        img_teacher_net.train()
        txt_teacher_net.train()

        teacher_optim = torch.optim.SGD([
            {'params': img_teacher_net.parameters(), 'lr': args.lr_teacher_img}, 
            {'params': txt_teacher_net.parameters(), 'lr': args.lr_teacher_txt},
        ], lr=0, momentum=args.mom, weight_decay=args.l2)
        teacher_optim.zero_grad()
        lr_schedule = [args.train_epochs // 2 + 1] if args.decay else []
        teacher_optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            teacher_optim, milestones=lr_schedule, gamma=0.1)


        img_timestamps = []
        txt_timestamps = []

        img_timestamps.append([p.detach().cpu() for p in img_teacher_net.parameters()])
        txt_timestamps.append([p.detach().cpu() for p in txt_teacher_net.parameters()])


        for e in trange(args.train_epochs):
            train_loss, train_acc = epoch(e, trainloader, teacher_net, teacher_optim, args)

            if (e+1) % args.eval_freq == 0:
                    
                score_val_i2t, score_val_t2i = epoch_test(testloader, teacher_net, args.device, bert_test_embed)
                val_result = itm_eval(score_val_i2t, score_val_t2i, testloader.dataset.txt2img, testloader.dataset.img2txt)  

                wandb.log({
                    "Loss/train_loss": train_loss,
                    "Loss/train_acc": train_acc,
                    "Results/txt_r1": val_result['txt_r1'],
                    "Results/txt_r5": val_result['txt_r5'],
                    "Results/txt_r10": val_result['txt_r10'],
                    # "txt_r_mean": val_result['txt_r_mean'],
                    "Results/img_r1": val_result['img_r1'],
                    "Results/img_r5": val_result['img_r5'],
                    "Results/img_r10": val_result['img_r10'],
                    # "img_r_mean": val_result['img_r_mean'],
                    "Results/r_mean": val_result['r_mean'],
                })

                print("Itr: {} Epoch={} Train Acc={} | Img R@1={} R@5={} R@10={} | Txt R@1={} R@5={} R@10={} | R@Mean={}".format(
                    it, e, train_acc,
                    val_result['img_r1'], val_result['img_r5'], val_result['img_r10'],
                    val_result['txt_r1'], val_result['txt_r5'], val_result['txt_r10'], val_result['r_mean'])) 


            img_timestamps.append([p.detach().cpu() for p in img_teacher_net.parameters()])
            txt_timestamps.append([p.detach().cpu() for p in txt_teacher_net.parameters()])

            teacher_optim_scheduler.step()


        if not args.skip_save:
            img_trajectories.append(img_timestamps)
            txt_trajectories.append(txt_timestamps)
            n = 0
            while os.path.exists(os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n))))
            torch.save(img_trajectories, os.path.join(save_dir, "img_replay_buffer_{}.pt".format(n)))
            print("Saving {}".format(os.path.join(save_dir, "txt_replay_buffer_{}.pt".format(n))))
            torch.save(txt_trajectories, os.path.join(save_dir, "txt_replay_buffer_{}.pt".format(n)))

            img_trajectories = []
            txt_trajectories = []


def make_buffer_parser():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='flickr', choices=['flickr', 'coco'], help='dataset')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher_img', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--lr_teacher_txt', type=float, default=0.1, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='./data/Flickr30k/', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization') 
    parser.add_argument('--save_interval', type=int, default=10)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parser.add_argument('--name', type=str, default=current_time, help='name of wandb run')
    parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
    parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
    parser.add_argument('--text_trainable', type=bool, default=False, help='text_trainable')
    parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable') 
    parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train')
    parser.add_argument('--batch_size_test', type=int, default=128, help='batch_size_test')
    parser.add_argument('--image_root', type=str, default='distill_utils/data/Flickr30k/', help='location of image root')
    parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/', help='location of ann root')
    parser.add_argument('--image_size', type=int, default=224, help='image_size')
    parser.add_argument('--k_test', type=int, default=128, help='k_test')
    parser.add_argument('--load_npy', type=bool, default=False, help='load_npy')
    parser.add_argument('--image_encoder', type=str, default='resnet50', help='image encoder')
    #, choices=['nfnet', 'resnet18_gn', 'vit_tiny', 'nf_resnet50', 'nf_regnet'])
    parser.add_argument('--text_encoder', type=str, default='bert', choices=['bert', 'clip', 'distilbert','gpt1'], help='text encoder')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--measure', default='cosine',
                    help='Similarity measure used (cosine|order)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--only_has_image_projection', type=bool, default=False, help='None')
    parser.add_argument('--grounding', type=bool, default=False, help='None')
    
    parser.add_argument('--distill', type=bool, default=False, help='whether distill')
    parser.add_argument('--loss_type', type=str, default="InfoNCE")

    parser.add_argument('--eval_freq', type=int, default=5, help='eval_freq')
    parser.add_argument('--no_aug', action='store_true', help='no_aug')
    parser.add_argument('--skip_save', action='store_true', help='skip save buffer')
    parser.add_argument('--disabled_wandb', type=bool, default=False)
    return parser


if __name__ == '__main__':
    parser = make_buffer_parser()
    args = parser.parse_args()

    args.image_root = {
        'flickr': "distill_utils/data/Flickr30k/",
        'coco': "distill_utils/data/COCO/",
    }[args.dataset]

    main(args)