# %%
import sys
sys.path.append('/n/home12/binxuwang/Github/DiffusionReasoning/')
if '/n/home12/binxuwang/.local/lib/python3.10/site-packages' in sys.path:
    sys.path.remove('/n/home12/binxuwang/.local/lib/python3.10/site-packages',)
import os
from os.path import join
import time
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import json
from tqdm import tqdm, trange
# from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AdamW
from mamba_ssm import MambaLMHeadModel, Mamba
from mamba_ssm.models.config_mamba import MambaConfig
from GPT_models.GPT_RAVEN_model_lib import completion_eval, sample_next_token, \
    multi_attr_loss_vec
from GPT_models.GPT_RAVEN_model_lib import SepWordEmbed, CmbWordEmbed, SepLMhead, CmbLMhead
from GPT_models.GPT_RAVEN_model_lib import seqtsr2imgtsr, seqtsr2attrtsr, compute_rule_statistics
from rule_new_utils import check_r3_r2_batch, infer_rule_from_sample_batch
from torch.utils.data import TensorDataset, DataLoader
import argparse

class MultiIdxMambaModel(nn.Module):
    def __init__(self, attribute_dims=(7,10,10), vocab_size=0, n_embd=768, n_class=0, is_sep_embed=True, **kwargs):
        super().__init__()
        # Combine embeddings
        combined_embedding_size = n_embd  # Adjust based on your combination strategy
        if is_sep_embed:
            self.sep_word_embed = SepWordEmbed(attribute_dims, embed_size=n_embd//3)
            self.multi_lmhead = SepLMhead(attribute_dims, embed_size=n_embd//3)
        else:
            self.sep_word_embed = CmbWordEmbed(attribute_dims, embed_size=n_embd)
            self.multi_lmhead = CmbLMhead(attribute_dims, embed_size=n_embd)
        config = MambaConfig(vocab_size=vocab_size, d_model=n_embd, **kwargs, 
                  ssm_cfg={"layer": "Mamba1"})
        self.mamba = MambaLMHeadModel(config, device='cuda')
        self.mamba.backbone.embedding = nn.Identity()
        self.mamba.lm_head = nn.Identity()
        self.context_embed = nn.Embedding(1+n_class, n_embd)

    def forward(self, input_ids, y=None):
        # input_ids is expected to be a list of three tensors [attr1, attr2, attr3]
        if y is None:
            y = torch.zeros(input_ids.shape[0], dtype=th.long).to(input_ids[0].device)
        ctx_vec = self.context_embed(y)
        combined_embedding = self.sep_word_embed(input_ids)
        combined_embedding = torch.concat([ctx_vec[:,None,:], combined_embedding, ], dim=1)
        outputs = self.mamba(combined_embedding) # this is actually hidden states not logits
        logits_attr1, logits_attr2, logits_attr3 = self.multi_lmhead(outputs.logits)
        return outputs, logits_attr1, logits_attr2, logits_attr3
    
# %%
# Initialize the GPT-2 model and tokenizer
def preprocess_ids(attr_seq_tsr, ):
    attr_seq_tsr_pps = attr_seq_tsr + 1 # clone() removed
    return attr_seq_tsr_pps


def main(args):
    # data_path = args.data_path 
    # heldout_id = [1, 16, 20, 34, 37]
    # heldout_id = []
    heldout_id = args.heldout_ids 
    save_ckpt_every_step = args.save_ckpt_every_step 
    eval_model_every_step = args.eval_model_every_step 
    explabel = args.explabel 
    
    n_embd = args.n_embd 
    n_layer = args.n_layer 
    if args.embedding_type == 'sep':
        is_sep_embed = True
    elif args.embedding_type == 'cmb':
        is_sep_embed = False
    n_class = args.n_class 
    cmb_per_class = args.cmb_per_class
    batch_size = args.batch_size 
    total_steps = args.total_steps 
    # epoch_total = args.epoch_total 
    lr = args.lr 
    num_warmup_steps = args.num_warmup_steps 
    eval_temperature = args.eval_temperature 
    
    # check if n_class is set correctly
    if n_class > 0:
        assert n_class == 40 # len(attr_all)
    is_class_cond = n_class > 0
    
    # Create a mask with all True values
    # Set the specified rows to False
    train_mask = torch.ones(40, dtype=torch.bool)
    train_mask[heldout_id] = False
    # old version
    # data_dir = '/n/home12/binxuwang/Github/DiffusionReasoning/'
    # attr_all = np.load(data_dir+'attr_all.npy')
    data_dir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/RPM_dataset/RPM1000k"
    attr_all = np.load(join(data_dir, "attr_all_1000k.npy"))
    print(attr_all.shape)
    attr_all_rows = torch.from_numpy(attr_all, )
    del attr_all
    # attr_img_tsr = einops.rearrange(attr_all_rows,  'class (B R) p (h w) attr -> class B attr (R h) (p w)', h=3,w=3,p=3,R=3)
    attr_seq_tsr = einops.rearrange(attr_all_rows,  'class (B R) p (h w) attr -> class B (R p h w) attr', h=3,w=3,p=3,R=3)
    del attr_all_rows
    # Set the y of the dataset, which is the class index; split the y into training and validation
    y_rule = th.arange(attr_seq_tsr.shape[0], dtype=th.long).unsqueeze(1)
    y_rule = y_rule.repeat(1, attr_seq_tsr.shape[1])
    attr_seq_tsr = preprocess_ids(attr_seq_tsr)
    # if the cmb_per_class is too large, change it such that it won't overlap with the validation set
    if cmb_per_class > attr_seq_tsr.shape[1] - 500:
        cmb_per_class = attr_seq_tsr.shape[1] - 500
    attr_seq_tsr_train, attr_seq_tsr_val, attr_seq_tsr_val_eval = \
        attr_seq_tsr[train_mask, :cmb_per_class], attr_seq_tsr[:, -500:], attr_seq_tsr[:, -50:] # changed June 30, 2024, also eval on untrained rules.
    y_rule_train, y_rule_val, y_rule_val_eval = \
        y_rule[train_mask, :cmb_per_class], y_rule[:, -500:], y_rule[:, -50:]
    y_rule_train = einops.rearrange(y_rule_train, 'class B -> (class B)', )
    y_rule_val = einops.rearrange(y_rule_val, 'class B -> (class B)', )
    y_rule_val_eval = einops.rearrange(y_rule_val_eval, 'class B -> (class B)', )
    # combine the first 2 axes into 1
    attr_seq_tsr_train = einops.rearrange(attr_seq_tsr_train, 'class B (R p h w) attr -> (class B) (R p h w) attr', R=3, p=3, h=3, w=3)
    attr_seq_tsr_val = einops.rearrange(attr_seq_tsr_val, 'class B (R p h w) attr -> (class B) (R p h w) attr', R=3, p=3, h=3, w=3)
    attr_seq_tsr_val_eval = einops.rearrange(attr_seq_tsr_val_eval, 'class B (R p h w) attr -> (class B) (R p h w) attr', R=3, p=3, h=3, w=3)
    print(attr_seq_tsr_train.shape, attr_seq_tsr_val.shape, attr_seq_tsr_val_eval.shape)
    del attr_seq_tsr
    # attr_img_tsr_train, attr_img_tsr_val = attr_img_tsr[train_mask, :-500], attr_img_tsr[:, -500:] # changed June 30, 2024, also eval on untrained rules. 
    # attr_seq_tsr_train = einops.rearrange(attr_img_tsr_train,  'class B attr (R h) (p w) -> (class B) (R p h w) attr', h=3,w=3,p=3,R=3)
    # attr_seq_tsr_val = einops.rearrange(attr_img_tsr_val,  'class B attr (R h) (p w) -> (class B) (R p h w) attr', h=3,w=3,p=3,R=3)
    # attr_seq_tsr_train = preprocess_ids(attr_seq_tsr_train)
    # attr_seq_tsr_val = preprocess_ids(attr_seq_tsr_val)
    # del attr_img_tsr, attr_img_tsr_train, attr_img_tsr_val
    #%%
    saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/Mamba_raven"
    expdir = join(saveroot, f"{explabel}-{time.strftime('%Y%m%d-%H%M%S')}")
    ckptdir = join(expdir, "ckpt")
    sampledir = join(expdir, "samples")
    for d in [expdir, ckptdir, sampledir]:
        os.makedirs(d, exist_ok=True)
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=join(expdir, 'tensorboard_logs'))

    train_dataset = TensorDataset(attr_seq_tsr_train, y_rule_train)
    val_dataset = TensorDataset(attr_seq_tsr_val, y_rule_val)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)
    epoch_total = total_steps // len(data_loader)
    print(f"Dataset size {len(train_dataset)}, batch size {batch_size}\n total steps {total_steps}, total epochs {epoch_total}")
    config = {"batch_size": batch_size, "epoch_total": epoch_total, #"save_ckpt_every": save_ckpt_every,
            "save_ckpt_every_step": save_ckpt_every_step, "eval_model_every_step": eval_model_every_step, 
            "lr": lr, "num_warmup_steps": num_warmup_steps,
            "n_embd": n_embd, "n_class": n_class, "n_layer": n_layer, 
            "is_sep_embed": is_sep_embed,
            "heldout_id": heldout_id, 
            "train_sample_num": len(attr_seq_tsr_train), 
            "val_sample_num": len(attr_seq_tsr_val),
            "eval_temperature": eval_temperature,
            "is_class_cond": is_class_cond, }
    json.dump(config, open(join(expdir, "config.json"), 'w'))
    #%%
    # bug fix @2024-06-28, before which, the "n_embd": n_embd, "n_class": n_class, "n_layer": n_layer, "n_head": n_head, are no effect
    mamba_raven = MultiIdxMambaModel(attribute_dims=(7,10,10), vocab_size=27, # max_length=83, 
                                n_class=n_class, n_embd=n_embd, n_layer=n_layer, is_sep_embed=is_sep_embed)
    # train loop

    # bug fix @2024-08-18, before which, the num_training_steps is not the total_steps, so wrong scheduler 
    # num_training_steps = len(data_loader) * epoch_total
    optimizer = AdamW(mamba_raven.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    
    mamba_raven.train().to('cuda')
    th.save(mamba_raven.state_dict(), join(ckptdir, 'mamba_init.pth'))
    global_step = 0
    if False and os.path.exists(join(ckptdir, 'mamba_optimizer_latest.pth')):
        optimizer_ckpt_path = join(ckptdir, 'mamba_optimizer_latest.pth')
        print(f"Loading checkpoint from {latest_checkpoint_path}")
        checkpoint = torch.load(optimizer_ckpt_path)
        # mamba_raven.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint['global_step'] + 1  # Start from the next step
        assert os.path.exists(join(ckptdir, f'mamba_step{checkpoint['global_step']}.pth'))
        mamba_raven.load_state_dict(torch.load(join(ckptdir, f'mamba_step{checkpoint['global_step']}.pth')))

    train_loss_sum = []
    pbar = trange(total_steps)
    data_iter = iter(data_loader)
    for step in pbar:
        try:
            inputs, ys = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            inputs, ys = next(data_iter)
        inputs = inputs.cuda()
        ys = ys.cuda()
        optimizer.zero_grad()
        if is_class_cond:
            outputs, logits_attr1, logits_attr2, logits_attr3 = mamba_raven(inputs, y=ys)
        else:
            outputs, logits_attr1, logits_attr2, logits_attr3 = mamba_raven(inputs, y=None)
        # note the inputs were pre-pended in mamba to add context
        loss = multi_attr_loss_vec([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
        # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_sum.append(loss.item())
        pbar.set_description(f'Loss: {loss.item()} lr: {scheduler.get_last_lr()[0]}')
        pbar.update()
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
        
        if (global_step + 1) % eval_model_every_step == 0 or \
           (global_step == 0) or \
           (global_step + 1 == total_steps):
            mamba_raven.eval()
            pbar = tqdm(val_loader)
            val_loss_sum = []
            for (inputs, ys) in pbar:
                inputs = inputs.cuda()
                ys = ys.cuda()
                with torch.no_grad():
                    if is_class_cond:
                        outputs, logits_attr1, logits_attr2, logits_attr3 = mamba_raven(inputs, y=ys)
                    else:
                        outputs, logits_attr1, logits_attr2, logits_attr3 = mamba_raven(inputs, y=None)
                    loss = multi_attr_loss_vec([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
                # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
                pbar.set_description(f'Loss: {loss.item()}')
                val_loss_sum.append(loss.item())
            print("Validation Cross Entropy Loss ",np.mean(val_loss_sum))
            writer.add_scalar('Val/Avg_Loss', np.mean(val_loss_sum), global_step)
            # evaluatate on the completion of validation set
            # rnd_idx = np.random.choice(len(attr_seq_tsr_val), 512)
            eval_samples = attr_seq_tsr_val_eval[:,:,:]
            eval_complete, C3_list, C2_list, rule_col_list, stats = completion_eval(eval_samples, mamba_raven, num_mask=9, batch_size=256, 
                                                                                device='cuda', strategy="greedy", return_stats=True,
                                                                                cond=y_rule_val_eval.cuda() if is_class_cond else None)
            # evaluation by ab initio generation of samples
            y_rule_abinit = th.arange(40).repeat_interleave(50)
            eval_samples_empty = th.zeros(len(y_rule_abinit), 81, 3, dtype=th.long).to('cuda')
            eval_complete_abinit, C3_list_abinit, C2_list_abinit, rule_col_list_abinit, stats_abinit = completion_eval(eval_samples_empty, mamba_raven, num_mask=81, batch_size=256, 
                                                        device='cuda', strategy="sample", temperature=eval_temperature, return_stats=True,
                                                        cond=y_rule_abinit.cuda() if is_class_cond else None)
            th.save({"eval_complete": eval_complete, "C3_list": C3_list, "C2_list": C2_list, "rule_col_list": rule_col_list, "stats": stats,
                        "eval_complete_abinit": eval_complete_abinit, "C3_list_abinit": C3_list_abinit, "C2_list_abinit": C2_list_abinit, "rule_col_list_abinit": rule_col_list_abinit, "stats_abinit": stats_abinit}, 
                        join(sampledir, f"eval_step{global_step}.pt"))
            writer.add_scalar('Val/C3', stats['C3'] / stats['total'], global_step)
            writer.add_scalar('Val/C2', stats['C2'] / stats['total'], global_step)
            writer.add_scalar('Val/AnyValid', stats['anyvalid'] / stats['total'] / 3, global_step)
            writer.add_scalar('Val/C3_abinit', stats_abinit['C3'] / stats_abinit['total'], global_step)
            writer.add_scalar('Val/C2_abinit', stats_abinit['C2'] / stats_abinit['total'], global_step)
            writer.add_scalar('Val/AnyValid_abinit', stats_abinit['anyvalid'] / stats_abinit['total'] / 3, global_step)
            mamba_raven.train()
            
        if (global_step + 1) % save_ckpt_every_step == 0:
            th.save(mamba_raven.state_dict(), join(ckptdir, f'mamba_step{global_step}.pth'))
            checkpoint = {
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step
            }
            torch.save(checkpoint, join(ckptdir, f'mamba_optimizer_latest.pth')) 
            # CAVEAT: the optimizer and scheduler are not saved in the checkpoint file, to save space only save the latest optimizer and scheduler states. 
        global_step += 1

    train_loss_avg = np.mean(train_loss_sum)
    writer.add_scalar('Train/Avg_Loss', train_loss_avg, global_step)

    th.save(mamba_raven.state_dict(), join(ckptdir, 'mamba_final.pth'))
    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mamba Model Training for RPM')
    # parser.add_argument('--data_path', type=str, required=True, help='Path to the attribute data file')
    parser.add_argument('--explabel', type=str, default="Mamba_medium_RAVEN_uncond_heldout0_stream16M", help='Experiment label')
    parser.add_argument('--heldout_ids', nargs='+', type=int, help='Indices of data to hold out during training', default=[1, 16, 20, 34, 37])
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--total_steps', type=int, default=250000, help='Total steps of training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_warmup_steps', type=int, default=100, help='Number of warmup steps')
    parser.add_argument('--eval_temperature', type=float, default=1.0, help='Temperature for evaluation')
    parser.add_argument("--cmb_per_class", type=int, default=4000)
    # parser.add_argument('--epoch_total', type=int, default=1, help='Total number of epochs')
    parser.add_argument('--save_ckpt_every_step', type=int, default=12500, help='Steps interval to save checkpoint')
    parser.add_argument('--eval_model_every_step', type=int, default=2500, help='Steps interval to evaluate the model')
    parser.add_argument('--n_embd', type=int, default=768, help='Embedding size')
    parser.add_argument('--n_layer', type=int, default=24, help='Number of layers')
    # parser.add_argument('--n_head', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--n_class', type=int, default=0, help='Number of classes')
    parser.add_argument('--embedding_type', type=str, default='sep', help='Type of embedding', choices=['sep', 'cmb'])

    args = parser.parse_args()
    main(args)


#%%

# batch_size = 64
# epoch_total = 100
# save_ckpt_every = 5
# # explabel = "Mamba_base_RAVEN_uncond_heldout0"
# # n_embd = 768
# # n_layer = 12
# # is_sep_embed = True
# # explabel = "mamba_medium_RAVEN_uncond_heldout0"
# # explabel = "mamba_medium_RAVEN_uncond_all"
# # n_embd = 768
# # n_layer = 24
# # is_sep_embed = True
# # explabel = "mamba_big_RAVEN_uncond_heldout0"
# # explabel = "mamba_big_RAVEN_uncond_all"
# # n_embd = 1152
# # n_layer = 36
# # is_sep_embed = True
# # explabel = "mamba_huge_RAVEN_uncond_heldout0"
# explabel = "mamba_huge_RAVEN_uncond_all"
# n_embd = 1536
# n_layer = 48
# is_sep_embed = True
# n_class = 0
# lr = 1e-4
# num_warmup_steps = 100
# eval_temperature = 1.0