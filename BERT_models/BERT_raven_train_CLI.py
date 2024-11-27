# %%
import os
from os.path import join
import json
import time
import einops
import numpy as np
# Standard PyTorch imports
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Transformers imports
from transformers import (
    BertConfig,
    BertForSequenceClassification, 
    BertTokenizer,
    BertModel
)
from tqdm.auto import trange
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
# Custom model components
import sys
sys.path.append('/n/home12/binxuwang/Github/DiffusionReasoning/')
from GPT_models.GPT_RAVEN_model_lib import (
    SepWordEmbed,
    CmbWordEmbed, 
    JointWordEmbed,
    SepLMhead,
    CmbLMhead
)

class MultiIdxBERTModel(nn.Module):
    def __init__(self, attribute_dims=(7,10,10), vocab_size=0, max_length=128, n_embd=768, n_class=40, embed_type="sep", **kwargs):

        super().__init__()
        # Combine embeddings
        combined_embedding_size = n_embd  # Adjust based on your combination strategy
        self.embed_type = embed_type
        if embed_type == "sep":
            self.sep_word_embed = SepWordEmbed(attribute_dims, embed_size=n_embd//3)
            # self.multi_lmhead = SepLMhead(attribute_dims, embed_size=n_embd//3)
        elif embed_type == "cmb":
            self.sep_word_embed = CmbWordEmbed(attribute_dims, embed_size=n_embd)
            # self.multi_lmhead = CmbLMhead(attribute_dims, embed_size=n_embd)
        elif embed_type == "joint":
            self.sep_word_embed = JointWordEmbed(attribute_dims, embed_size=n_embd)
        config = BertConfig(vocab_size=vocab_size, 
                            max_position_embeddings=max_length, 
                            hidden_size=combined_embedding_size, **kwargs)
        self.bert = BertModel(config)
        self.context_embed = nn.Embedding(1, n_embd) # dummy embedding for start token
        self.classifier = nn.Linear(n_embd, n_class)

    def forward(self, input_ids, y=None):
        # input_ids is expected to be a list of three tensors [attr1, attr2, attr3]
        SOS = torch.zeros(input_ids.shape[0], dtype=th.long).to(input_ids[0].device)
        SOS_vec = self.context_embed(SOS)
        combined_embedding = self.sep_word_embed(input_ids)
        combined_embedding = torch.concat([SOS_vec[:,None,:], combined_embedding, ], dim=1)
        outputs = self.bert(inputs_embeds=combined_embedding)
        logits = self.classifier(outputs.pooler_output)
        return logits, outputs.last_hidden_state


def preprocess_ids(attr_seq_tsr, ):
    attr_seq_tsr_pps = attr_seq_tsr + 1 # clone() removed
    return attr_seq_tsr_pps

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--explabel', type=str, default='bert_raven', help='Experiment label')
parser.add_argument('--cmb_per_class', type=int, default=4000, help='Number of combinations per class')
parser.add_argument('--heldout_ids', type=int, nargs='+', default=[1, 16, 20, 34, 37], help='IDs to hold out')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--num_warmup_steps', type=int, default=1000)
# parser.add_argument('--epoch_total', type=int, default=20)
parser.add_argument('--total_steps', type=int, default=1000000)
parser.add_argument('--eval_model_every_step', type=int, default=500)
parser.add_argument('--save_ckpt_every_step', type=int, default=5000)
parser.add_argument('--n_embd', type=int, default=384, help='Embedding dimension')
parser.add_argument('--n_layer', type=int, default=12, help='Number of layers')
parser.add_argument('--n_head', type=int, default=6, help='Number of attention heads')
parser.add_argument('--n_class', type=int, default=40, help='Number of classes')
parser.add_argument('--embed_type', type=str, default='sep', choices=['sep', 'cmb', 'joint'], help='Type of embedding')
args = parser.parse_args()

explabel = args.explabel
cmb_per_class = args.cmb_per_class
heldout_id = args.heldout_ids
batch_size = args.batch_size
lr = args.lr
num_warmup_steps = args.num_warmup_steps
# epoch_total = args.epoch_total
total_steps = args.total_steps
eval_model_every_step = args.eval_model_every_step
save_ckpt_every_step = args.save_ckpt_every_step
n_embd = args.n_embd
n_layer = args.n_layer
n_head = args.n_head
n_class = args.n_class
embed_type = args.embed_type

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


saveroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/BERT_raven_classify"
expdir = join(saveroot, f"{explabel}-{time.strftime('%Y%m%d-%H%M%S')}")
ckptdir = join(expdir, "ckpt")
# sampledir = join(expdir, "samples")
for d in [expdir, ckptdir]: # , sampledir
    os.makedirs(d, exist_ok=True)
# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=join(expdir, 'tensorboard_logs'))
# Initialize dataset
train_dataset = TensorDataset(attr_seq_tsr_train, y_rule_train)
val_dataset = TensorDataset(attr_seq_tsr_val, y_rule_val)
val_dataset_eval = TensorDataset(attr_seq_tsr_val_eval, y_rule_val_eval)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)
val_loader_eval = DataLoader(val_dataset_eval, batch_size=256, shuffle=False, drop_last=False)

# train loop
epoch_total = total_steps // len(data_loader)
print(f"Total steps {total_steps}, epoch total {epoch_total}, len(data_loader) {len(data_loader)}")
config = {"batch_size": batch_size, "epoch_total": epoch_total, #"save_ckpt_every": save_ckpt_every,
        "save_ckpt_every_step": save_ckpt_every_step, "eval_model_every_step": eval_model_every_step, 
        "lr": lr, "num_warmup_steps": num_warmup_steps,
        "n_embd": n_embd, "n_class": n_class, "n_layer": n_layer, "n_head": n_head,
        "embed_type": embed_type,
        "heldout_id": heldout_id, 
        "train_sample_num": len(attr_seq_tsr_train), 
        "val_sample_num": len(attr_seq_tsr_val), }
json.dump(config, open(join(expdir, "config.json"), 'w'))

bert_raven = MultiIdxBERTModel(attribute_dims=(7,10,10), vocab_size=27, max_length=83, 
                                n_class=40, n_embd=n_embd, embed_type=embed_type, n_layer=n_layer, n_head=n_head)
# total_steps = len(data_loader) * epoch_total
optimizer = AdamW(bert_raven.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
bert_raven.train().to('cuda')

epoch = 0
data_iter = iter(data_loader)
pbar = trange(total_steps)
for step in pbar:
    try:
        inputs, ys = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        inputs, ys = next(data_iter)
        epoch += 1
        writer.add_scalar('Training/Epoch', epoch, step)
    
    inputs = inputs.cuda()
    ys = ys.cuda()
    optimizer.zero_grad()
    logits, outputs = bert_raven(inputs, y=ys)
    loss = F.cross_entropy(logits, ys)
    acc_cnt = (logits.argmax(dim=-1) == ys).float().sum().item()
    loss.backward()
    optimizer.step()
    scheduler.step()
    pbar.set_postfix(loss=loss.item(), acc=acc_cnt/len(ys))
    # Log training metrics
    writer.add_scalar('Training/Loss', loss.item(), step)
    writer.add_scalar('Training/Accuracy', acc_cnt/len(ys), step)
    writer.add_scalar('Training/Learning_Rate', scheduler.get_last_lr()[0], step)
    # evaluate test set
    if (step + 1) % eval_model_every_step == 0 or step == total_steps - 1 or step == 0:
        bert_raven.eval()
        with torch.no_grad():
            val_loss = 0
            acc_cnt = 0
            for inputs, ys in val_loader_eval:
                inputs = inputs.cuda()
                ys = ys.cuda()
                logits, outputs = bert_raven(inputs, y=ys)
                loss = F.cross_entropy(logits, ys)
                val_loss += loss.item()
                acc_cnt += (logits.argmax(dim=-1) == ys).float().sum().item()
            acc_ratio = acc_cnt / len(val_loader_eval.dataset)
            loss_avg = val_loss / len(val_loader_eval)
            print(f"Epoch {epoch} Step {step} Validation loss: {loss_avg}, accuracy: {acc_ratio}")
            
            # Log validation metrics
            writer.add_scalar('Valid/Loss', loss_avg, step)
            writer.add_scalar('Valid/Accuracy', acc_ratio, step)
            
        bert_raven.train()
    
    if (step + 1) % save_ckpt_every_step == 0 or step == total_steps - 1:
        torch.save(bert_raven.state_dict(), f'bert_raven_step{step}.pth')

writer.close()
