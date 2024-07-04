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
# Import necessary libraries
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm import tqdm, trange
# from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AdamW
from GPT_models.GPT_RAVEN_model_lib import completion_eval, sample_next_token, \
    multi_attr_loss_vec, MultiIdxGPT2Model
from GPT_models.GPT_RAVEN_model_lib import seqtsr2imgtsr, seqtsr2attrtsr, compute_rule_statistics
from rule_new_utils import check_r3_r2_batch, infer_rule_from_sample_batch
from torch.utils.data import TensorDataset, DataLoader

# %%
# Initialize the GPT-2 model and tokenizer
def preprocess_ids(attr_seq_tsr, ):
    attr_seq_tsr_pps = attr_seq_tsr.clone() + 1
    return attr_seq_tsr_pps

heldout_id = [1, 16, 20, 34, 37]
heldout_id = []
# Create a mask with all True values
# Set the specified rows to False
train_mask = torch.ones(40, dtype=torch.bool)
train_mask[heldout_id] = False
data_dir = '/n/home12/binxuwang/Github/DiffusionReasoning/'
attr_all = np.load(data_dir+'attr_all.npy')
print(attr_all.shape)
attr_all_rows = torch.tensor(attr_all)
attr_img_tsr = einops.rearrange(attr_all_rows,  'class (B R) p (h w) attr -> class B attr (R h) (p w)', h=3,w=3,p=3,R=3)
attr_img_tsr_train, attr_img_tsr_val = attr_img_tsr[train_mask, :3950], attr_img_tsr[:, 3950:] # changed June 30, 2024, also eval on untrained rules. 
attr_seq_tsr_train = einops.rearrange(attr_img_tsr_train,  'class B attr (R h) (p w) -> (class B) (R p h w) attr', h=3,w=3,p=3,R=3)
attr_seq_tsr_val = einops.rearrange(attr_img_tsr_val,  'class B attr (R h) (p w) -> (class B) (R p h w) attr', h=3,w=3,p=3,R=3)
attr_seq_tsr_train = preprocess_ids(attr_seq_tsr_train)
attr_seq_tsr_val = preprocess_ids(attr_seq_tsr_val)
print(attr_seq_tsr_train.shape, attr_seq_tsr_val.shape)
# Set the y of the dataset, which is the class index; split the y into training and validation
y_rule = th.arange(attr_img_tsr.shape[0], dtype=th.long).unsqueeze(1)
y_rule = y_rule.repeat(1, attr_img_tsr.shape[1])
y_rule_train, y_rule_val = y_rule[train_mask, :3950], y_rule[:, 3950:]
y_rule_train = einops.rearrange(y_rule_train, 'class B -> (class B)', )
y_rule_val = einops.rearrange(y_rule_val, 'class B -> (class B)', )
#%%
saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/GPT2_raven"

batch_size = 64
epoch_total = 100
save_ckpt_every = 5
# explabel = "GPT2_base_RAVEN_cond_heldout0"
# n_embd = 768
# n_layer = 12
# n_head = 12
# is_sep_embed = True
explabel = "GPT2_medium_RAVEN_cond_heldout0"
explabel = "GPT2_medium_RAVEN_cond_all"
n_embd = 768
n_layer = 24
n_head = 12
is_sep_embed = True
# explabel = "GPT2CmbEmb_base_RAVEN_cond_heldout0"
# n_embd = 768
# n_layer = 12
# n_head = 12
# is_sep_embed = False
n_class = 40
lr = 1e-4
num_warmup_steps = 100
eval_temperature = 1.0

if n_class > 0:
    assert n_class == len(attr_all)
is_class_cond = n_class > 0

expdir = join(saveroot, f"{explabel}-{time.strftime('%Y%m%d-%H%M%S')}")
ckptdir = join(expdir, "ckpt")
sampledir = join(expdir, "samples")
for d in [expdir, ckptdir, sampledir]:
    os.makedirs(d, exist_ok=True)
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=join(expdir, 'tensorboard_logs'))

config = {"batch_size": batch_size, "epoch_total": epoch_total, "save_ckpt_every": save_ckpt_every,
           "lr": lr, "num_warmup_steps": num_warmup_steps,
           "n_embd": n_embd, "n_class": n_class, "n_layer": n_layer, "n_head": n_head,
           "is_sep_embed": is_sep_embed,
           "heldout_id": heldout_id, 
           "train_sample_num": len(attr_seq_tsr_train), 
           "val_sample_num": len(attr_seq_tsr_val),
           "eval_temperature": eval_temperature,
           "is_class_cond": is_class_cond, }
json.dump(config, open(join(expdir, "config.json"), 'w'))

#%%
# bug fix @2024-06-28, before which, the "n_embd": n_embd, "n_class": n_class, "n_layer": n_layer, "n_head": n_head, are no effect
gpt2_raven = MultiIdxGPT2Model(attribute_dims=(7,10,10), vocab_size=27, max_length=83, 
                               n_class=n_class, n_embd=n_embd, n_layer=n_layer, n_head=n_head, is_sep_embed=is_sep_embed)
# train loop
# dataset = torch.utils.data.TensorDataset(attr_seq_tsr_pps)
train_dataset = TensorDataset(attr_seq_tsr_train, y_rule_train)
val_dataset = TensorDataset(attr_seq_tsr_val, y_rule_val)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)

num_training_steps = len(data_loader) * epoch_total
optimizer = AdamW(gpt2_raven.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
gpt2_raven.train().to('cuda')
th.save(gpt2_raven.state_dict(), join(ckptdir, 'gpt2_init.pth'))
global_step = 0
for epoch in range(epoch_total):
    gpt2_raven.train()
    pbar = tqdm(data_loader)
    train_loss_sum = []
    for step, (inputs, ys) in enumerate(pbar):
        inputs = inputs.cuda()
        ys = ys.cuda()
        optimizer.zero_grad()
        if is_class_cond:
            outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=ys)
        else:
            outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=None)
        # note the inputs were pre-pended in gpt2 to add context
        loss = multi_attr_loss_vec([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
        # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss_sum.append(loss.item())
        pbar.set_description(f'Loss: {loss.item()} lr: {scheduler.get_last_lr()[0]}')
        writer.add_scalar('Loss/train', loss.item(), global_step)
        writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
        global_step += 1
    
    train_loss_avg = np.mean(train_loss_sum)
    writer.add_scalar('Train/Avg_Loss', train_loss_avg, epoch)
    gpt2_raven.eval()
    pbar = tqdm(val_loader)
    val_loss_sum = []
    for (inputs, ys) in pbar:
        inputs = inputs.cuda()
        ys = ys.cuda()
        with torch.no_grad():
            if is_class_cond:
                outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=ys)
            else:
                outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=None)
            loss = multi_attr_loss_vec([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
        # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
        pbar.set_description(f'Loss: {loss.item()}')
        val_loss_sum.append(loss.item())
    print("Validation Cross Entropy Loss ",np.mean(val_loss_sum))
    writer.add_scalar('Val/Avg_Loss', np.mean(val_loss_sum), epoch)
    # evaluatate on the completion of validation set
    # rnd_idx = np.random.choice(len(attr_seq_tsr_val), 512)
    eval_samples = attr_seq_tsr_val[:,:,:]
    eval_complete, C3_list, C2_list, rule_col_list, stats = completion_eval(eval_samples, gpt2_raven, num_mask=9, batch_size=256, 
                                                                        device='cuda', strategy="greedy", return_stats=True,
                                                                        cond=y_rule_val.cuda() if is_class_cond else None)
    # evaluation by ab initio generation of samples
    y_rule_abinit = th.arange(40).repeat_interleave(50)
    eval_samples_empty = th.zeros(len(y_rule_abinit), 81, 3, dtype=th.long).to('cuda')
    eval_complete_abinit, C3_list_abinit, C2_list_abinit, rule_col_list_abinit, stats_abinit = completion_eval(eval_samples_empty, gpt2_raven, num_mask=81, batch_size=256, 
                                                device='cuda', strategy="sample", temperature=eval_temperature, return_stats=True,
                                                cond=y_rule_abinit.cuda() if is_class_cond else None)
    th.save({"eval_complete": eval_complete, "C3_list": C3_list, "C2_list": C2_list, "rule_col_list": rule_col_list, "stats": stats,
                "eval_complete_abinit": eval_complete_abinit, "C3_list_abinit": C3_list_abinit, "C2_list_abinit": C2_list_abinit, "rule_col_list_abinit": rule_col_list_abinit, "stats_abinit": stats_abinit}, 
                join(sampledir, f"eval_epoch{epoch}.pt"))
    writer.add_scalar('Val/C3', stats['C3'] / stats['total'], epoch)
    writer.add_scalar('Val/C2', stats['C2'] / stats['total'], epoch)
    writer.add_scalar('Val/AnyValid', stats['anyvalid'] / stats['total'] / 3, epoch)
    writer.add_scalar('Val/C3_abinit', stats_abinit['C3'] / stats_abinit['total'], epoch)
    writer.add_scalar('Val/C2_abinit', stats_abinit['C2'] / stats_abinit['total'], epoch)
    writer.add_scalar('Val/AnyValid_abinit', stats_abinit['anyvalid'] / stats_abinit['total'] / 3, epoch)

    if (epoch + 1) % save_ckpt_every == 0:
        th.save(gpt2_raven.state_dict(), join(ckptdir, f'gpt2_ep{epoch}.pth'))

th.save(gpt2_raven.state_dict(), join(ckptdir, 'gpt2_final.pth'))
# Close the TensorBoard writer
writer.close()


