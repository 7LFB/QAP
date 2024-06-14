from datetime import datetime
import argparse
import os


ROOT_DIR=''
subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
SNAPSHOT_DIR = ROOT_DIR+'/checkpoints/XPrompt/'+subdir+'/'

def parse_args():
    parser = argparse.ArgumentParser(description='Train network')
    
   
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--val_fold', default=0, type=int)
    parser.add_argument('--test_fold1', default=0, type=int)
    parser.add_argument('--test_fold2', default=0, type=int)

    parser.add_argument('--data_version', default=0, type=int)
    parser.add_argument('--file_list_folder', default='', type=str)
    parser.add_argument('--data_dir', default='', type=str)

    parser.add_argument('--cv_mode',default=5,type=int)
    parser.add_argument('--classifier', default=0, type=int)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--ratio', default=1.0, type=float)


    parser.add_argument('--model_name', default='XPrompt',type=str)
    parser.add_argument('--snapshot_dir',type=str, default=SNAPSHOT_DIR)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--resume_from', type=str, default='')
    parser.add_argument('--resume_keywords', type=str, default='')
    parser.add_argument('--logdir', type=str, default='XPrompt')

    parser.add_argument('--gpus', default=1,type=int)
    parser.add_argument('--workers', default=32,type=int)

   
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_for_test', type=int, default=1)
    parser.add_argument('--ckpt_saved_epoch', type=int, default=24)

    parser.add_argument('--img_h', type=int, default=224)
    parser.add_argument('--img_w', type=int, default=224)

    parser.add_argument('--augment', type=int, default=0)
    parser.add_argument('--auto_augment', type=int, default=0)
    parser.add_argument('--augment_index', type=str, default='none')
    parser.add_argument('--auto_augment_index', type=str, default='none')


    parser.add_argument('--mversion',default=36,type=int)
    parser.add_argument('--loss_type',default='ce',type=str)


    parser.add_argument('--lr',type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--record_epoch', type=int, default=0)
    parser.add_argument('--weight_decay',type=float, default=1e-2)
    parser.add_argument('--momentum',type=float, default=1e-4)

    parser.add_argument('--seed', type=int, default=1024)

    # pretrained fundation model
    parser.add_argument('--vit_num_layers', type=int, default=12)
    parser.add_argument('--pretrained_weights', type=str, default='/path-to-root-dir/pretrained/vit256_small_dino.pth')
    # vit256_small_dino.pth dict=['student','teacher','optimizer']

    # freeze components
    parser.add_argument('--freeze_pattern', type=str, default='vit')

    # encoder hidden feature size
    parser.add_argument('--hidden_size', default=384, type=int)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--prompt_token_length', default=1, type=int)
    parser.add_argument('--prompt_img_h', default=1, type=int)
    parser.add_argument('--prompt_img_w', default=1, type=int)
    parser.add_argument('--prompt_combine_operator', default='cat', type=str)
    parser.add_argument('--prompt_deep', default=1, type=int)
    parser.add_argument('--prompt_project', default=-1, type=int)
    parser.add_argument('--prompt_dropout', default=0.0, type=float)
    parser.add_argument('--xprompt', default=0, type=int)
    parser.add_argument('--prompt_dir', default='', type=str)
    parser.add_argument('--build_prompt', default=0, type=int)
    parser.add_argument('--subfolder', default='nuclei-kfunc', type=str)

    # attributes embeddings
    parser.add_argument('--embed_num', default=5, type=int)
    parser.add_argument('--proj_input_dim', default=28, type=int)
    parser.add_argument('--proj_hidd_dim', default=256, type=int)
    parser.add_argument('--prompt_index_str', default='28', type=str)
    parser.add_argument('--prompt_index_str_s', default='0-1-2-3', type=str)
    parser.add_argument('--prompt_index_str_m', default='4', type=str)
    parser.add_argument('--kfunc_version', default='S0V0', type=str)
    parser.add_argument('--prompt_s_num', default=4, type=int)
    parser.add_argument('--prompt_m_num', default=1, type=int)
    parser.add_argument('--norm_props', default=0, type=int)
    parser.add_argument('--norm_feats', default=1, type=int)

    # prompt generation implemented with tranformerEncoderLayer
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--layers', default=1, type=int)
    parser.add_argument('--activation', default='relu', type=str)


    # for properties generatoin
    parser.add_argument('--max', default=1, type=int)
    parser.add_argument('--filter', default=0.1, type=float)
    parser.add_argument('--prior_nuclei', default=0, type=int)
    parser.add_argument('--prior_white', default=0, type=int)
    parser.add_argument('--dilate', default=0, type=int)
    parser.add_argument('--se_kernel', default=11, type=int)
    parser.add_argument('--area_thd', default=0, type=int)
    parser.add_argument('--convex_hull', default=0, type=int)
    parser.add_argument('--clear_border', default=0, type=int)
    parser.add_argument('--generate_props', default=0, type=int)
    parser.add_argument('--sample_number', default=10, type=int) # convering distribution into sampling vector, how many points to sampler, default=10
    parser.add_argument('--nuclei_folder', default='nuclei_segment_watershed', type=str)
    parser.add_argument('--white_folder', default='white_segment', type=str)


    # 
    # parser.add_argument('--forward_embeds', default=0, type=int)




    #
    

   
    args = parser.parse_args()

    return args
