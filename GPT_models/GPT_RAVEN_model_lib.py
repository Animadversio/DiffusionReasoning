# %%
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
# Import necessary libraries
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tqdm.auto import tqdm, trange
# from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, AdamW
# Initialize the GPT-2 model and tokenizer

# %%
class SepWordEmbed(nn.Module):
    def __init__(self, embed_dims=(7,10,10), embed_size=256):
        super(SepWordEmbed, self).__init__()
        self.embedding1 = nn.Embedding(embed_dims[0]+1, embed_size)
        self.embedding2 = nn.Embedding(embed_dims[1]+1, embed_size)
        self.embedding3 = nn.Embedding(embed_dims[2]+1, embed_size)

    def forward(self, attr_seq_tsr):
        # split the attr_seq_tsr into three parts along the last dimension
        # attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3 = torch.split(attr_seq_tsr, [1,1,1], dim=-1)
        attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3 = attr_seq_tsr[...,0], attr_seq_tsr[...,1], attr_seq_tsr[...,2]
        # attr_seq_embed = self.embedding1(attr_seq_tsr_1) + self.embedding2(attr_seq_tsr_2) + self.embedding3(attr_seq_tsr_3)
        attr_seq_embed = th.concat([self.embedding1(attr_seq_tsr_1), 
                                    self.embedding2(attr_seq_tsr_2), 
                                    self.embedding3(attr_seq_tsr_3)], dim=-1)
        return attr_seq_embed
    
class SepLMhead(nn.Module):
    def __init__(self, embed_dims=(7,10,10), embed_size=256):
        super(SepLMhead, self).__init__()
        self.embed_size = embed_size
        self.lmhead1 = nn.Linear(embed_size, embed_dims[0]+1)
        self.lmhead2 = nn.Linear(embed_size, embed_dims[1]+1)
        self.lmhead3 = nn.Linear(embed_size, embed_dims[2]+1)
        
    def forward(self, attr_seq_embed):
        embed1, embed2, embed3 = torch.split(attr_seq_embed, [self.embed_size,self.embed_size,self.embed_size], dim=-1)
        attr_seq_tsr_1 = self.lmhead1(embed1)
        attr_seq_tsr_2 = self.lmhead2(embed2)
        attr_seq_tsr_3 = self.lmhead3(embed3)
        return attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3


class CmbWordEmbed(nn.Module):
    def __init__(self, embed_dims=(7,10,10), embed_size=768):
        super(CmbWordEmbed, self).__init__()
        self.embedding1 = nn.Embedding(embed_dims[0]+1, embed_size)
        self.embedding2 = nn.Embedding(embed_dims[1]+1, embed_size)
        self.embedding3 = nn.Embedding(embed_dims[2]+1, embed_size)

    def forward(self, attr_seq_tsr):
        # split the attr_seq_tsr into three parts along the last dimension
        # attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3 = torch.split(attr_seq_tsr, [1,1,1], dim=-1)
        attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3 = attr_seq_tsr[...,0], attr_seq_tsr[...,1], attr_seq_tsr[...,2]
        attr_seq_embed = self.embedding1(attr_seq_tsr_1)+\
                         self.embedding2(attr_seq_tsr_2)+\
                         self.embedding3(attr_seq_tsr_3)
        return attr_seq_embed
    
class CmbLMhead(nn.Module):
    def __init__(self, embed_dims=(7,10,10), embed_size=768):
        super(CmbLMhead, self).__init__()
        self.embed_size = embed_size
        self.lmhead1 = nn.Linear(embed_size, embed_dims[0]+1)
        self.lmhead2 = nn.Linear(embed_size, embed_dims[1]+1)
        self.lmhead3 = nn.Linear(embed_size, embed_dims[2]+1)
        
    def forward(self, attr_seq_embed):
        # embed1, embed2, embed3 = torch.split(attr_seq_embed, [self.embed_size,self.embed_size,self.embed_size], dim=-1)
        attr_seq_tsr_1 = self.lmhead1(attr_seq_embed)
        attr_seq_tsr_2 = self.lmhead2(attr_seq_embed)
        attr_seq_tsr_3 = self.lmhead3(attr_seq_embed)
        return attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3
    

class MultiIdxGPT2Model(nn.Module):
    def __init__(self, attribute_dims=(7,10,10), vocab_size=0, max_length=128, n_embd=768, n_class=0, is_sep_embed=True, **kwargs):
        # config = GPT2Config(
        #     vocab_size=27,
        #     n_positions=128,
        #     n_ctx=128,
        #     n_embd=768,
        #     n_layer=12,
        #     n_head=12,
        #     activation_function='gelu_new',
        #     resid_pdrop=0.1,
        #     embd_pdrop=0.1,
        #     attn_pdrop=0.1,
        #     layer_norm_epsilon=1e-5,
        #     initializer_range=0.02,
        #     summary_type='cls_index',
        #     summary_use_proj=True,
        #     summary_activation=None,
        #     summary_proj_to_labels=True,
        #     summary_first_dropout=0.1,
        #     bos_token_id=50256,
        #     eos_token_id=50256,
        #     gradient_checkpointing=False,
        # )
        super().__init__()
        # Combine embeddings
        combined_embedding_size = n_embd  # Adjust based on your combination strategy
        if is_sep_embed:
            self.sep_word_embed = SepWordEmbed(attribute_dims, embed_size=n_embd//3)
            self.multi_lmhead = SepLMhead(attribute_dims, embed_size=n_embd//3)
        else:
            self.sep_word_embed = CmbWordEmbed(attribute_dims, embed_size=n_embd)
            self.multi_lmhead = CmbLMhead(attribute_dims, embed_size=n_embd)
        config = GPT2Config(vocab_size=vocab_size, n_positions=max_length, n_embd=combined_embedding_size, **kwargs)
        self.gpt2 = GPT2Model(config)
        self.context_embed = nn.Embedding(1+n_class, n_embd)

    def forward(self, input_ids, y=None):
        # input_ids is expected to be a list of three tensors [attr1, attr2, attr3]
        if y is None:
            y = torch.zeros(input_ids.shape[0], dtype=th.long).to(input_ids[0].device)
        ctx_vec = self.context_embed(y)
        combined_embedding = self.sep_word_embed(input_ids)
        combined_embedding = torch.concat([ctx_vec[:,None,:], combined_embedding, ], dim=1)
        outputs = self.gpt2(inputs_embeds=combined_embedding)
        logits_attr1, logits_attr2, logits_attr3 = self.multi_lmhead(outputs.last_hidden_state)
        return outputs, logits_attr1, logits_attr2, logits_attr3
    

def multi_attr_loss(outputs, targets, loss_fn=F.cross_entropy, ):
    loss1 = loss_fn(outputs[0].permute(0,2,1), targets[..., 0])
    loss2 = loss_fn(outputs[1].permute(0,2,1), targets[..., 1])
    loss3 = loss_fn(outputs[2].permute(0,2,1), targets[..., 2])
    return loss1 + loss2 + loss3


def multi_attr_loss_vec(outputs, targets, loss_fn=F.cross_entropy, ):
    logits1, logits2, logits3 = outputs[0], outputs[1], outputs[2]
    loss1 = loss_fn(logits1.reshape(-1, logits1.size(-1)), targets[..., 0].view(-1))
    loss2 = loss_fn(logits2.reshape(-1, logits2.size(-1)), targets[..., 1].view(-1))
    loss3 = loss_fn(logits3.reshape(-1, logits3.size(-1)), targets[..., 2].view(-1))
    return loss1 + loss2 + loss3


def next_token_loss(outputs, targets, loss_fn=F.cross_entropy):
    logits1, logits2, logits3 = outputs[0], outputs[1], outputs[2]
    loss1 = loss_fn(logits1[:, :-1, :].permute(0,2,1), targets[:, 1:, 0])
    loss2 = loss_fn(logits2[:, :-1, :].permute(0,2,1), targets[:, 1:, 1])
    loss3 = loss_fn(logits3[:, :-1, :].permute(0,2,1), targets[:, 1:, 2])
    return loss1 + loss2 + loss3

# %% [markdown]
# ### Eval models
@torch.no_grad()
def sample_next_token(model, prefix_inputs, max_length=81, strategy="greedy", device="cuda", temperature=1.0, cond=None):
    prefix_inputs = prefix_inputs.to(device)
    model.eval().to(device)
    prefix_length = prefix_inputs.size(1)
    for i in range(max_length - prefix_length):
        outputs, logits1, logits2, logits3 = model(prefix_inputs, y=cond)
        if strategy == "greedy":
            next_token1 = torch.argmax(logits1[:, -1, :], dim=-1, keepdim=True)
            next_token2 = torch.argmax(logits2[:, -1, :], dim=-1, keepdim=True)
            next_token3 = torch.argmax(logits3[:, -1, :], dim=-1, keepdim=True)
        elif strategy == "sample":
            next_token1 = torch.multinomial(F.softmax(logits1[:, -1, :] / temperature, dim=-1), num_samples=1)
            next_token2 = torch.multinomial(F.softmax(logits2[:, -1, :] / temperature, dim=-1), num_samples=1)
            next_token3 = torch.multinomial(F.softmax(logits3[:, -1, :] / temperature, dim=-1), num_samples=1)
        else:
            raise ValueError("Invalid strategy")
        next_token = torch.cat([next_token1, next_token2, next_token3], dim=-1)
        prefix_inputs = torch.cat([prefix_inputs, next_token[:,None,:]], dim=1)
    return prefix_inputs
# %%
import sys
sys.path.append('/n/home12/binxuwang/Github/DiffusionReasoning/')
from rule_new_utils import check_r3_r2_batch, infer_rule_from_sample_batch

def seqtsr2imgtsr(seqtsr, h=3, w=3, p=3, R=3):
    imgtsr = einops.rearrange(seqtsr, 'B (R p h w) attr -> B attr (R h) (p w)', h=h, w=w, p=p, R=R)
    return imgtsr

def seqtsr2attrtsr(seqtsr, h=3, w=3, p=3, R=3):
    attrtsr = einops.rearrange(seqtsr, 'B (R p h w) attr -> B R p (h w) attr', h=h, w=w, p=p, R=R)
    return attrtsr

def compute_rule_statistics(r3_list, r2_list, rule_col):
    r3_count = sum([len(x) > 0 for x in r3_list])
    r2_count = sum([len(x) > 0 for x in r2_list])
    rule_flatten = np.array(rule_col, dtype=object).flatten() # [3 * 1024]
    anyvalid_count = sum([len(x) > 0 for x in rule_flatten])
    total = len(r3_list)
    return r3_count, r2_count, anyvalid_count, total

# %%
@torch.no_grad()
def completion_eval(eval_samples, model, cond=None, device='cuda', num_mask=9, batch_size=512, 
                    strategy="greedy", temperature=1.0, return_stats=False):
    eval_samples = eval_samples.to(device)
    if batch_size is None:
        batch_size = eval_samples.size(0)
    eval_complete = []
    for idx in trange(0, eval_samples.size(0), batch_size):
        eval_batch = eval_samples[idx:idx+batch_size]
        cond_batch = cond[idx:idx+batch_size] if cond is not None else None
        eval_complete_batch = sample_next_token(model, eval_batch[:,:-num_mask,:], temperature=temperature,
                                          max_length=81, strategy=strategy, device=device, cond=cond_batch).cpu()
        eval_complete.append(eval_complete_batch)
        
    eval_complete = torch.cat(eval_complete, dim=0)
    # eval_complete = sample_next_token(model, eval_samples[:,:-num_mask,:], 
    #                                   max_length=81, strategy=strategy, device=device).cpu()
    # eval_complete_attr = seqtsr2attrtsr(eval_complete, h=3, w=3, p=3, R=3)
    # note we need to denormalize, offset by - 1
    eval_complete = eval_complete - 1
    eval_complete_img = seqtsr2imgtsr(eval_complete, h=3, w=3, p=3, R=3)
    C3_list, C2_list, rule_col_list = infer_rule_from_sample_batch(eval_complete_img)
    C3_count, C2_count, anyvalid_count, total = compute_rule_statistics(C3_list, C2_list, rule_col_list)
    # final_row = np.array(rule_col_list, dtype=object)[:,-1]
    # anyvalid_count = sum([len(x) > 0 for x in final_row])
    print(f"Completion: C3: {C3_count / total:.3f} [{C3_count}/{total}],  valid: {anyvalid_count / total / 3:.3f} [{anyvalid_count}/{total*3}]")
    if return_stats:
        return eval_complete, C3_list, C2_list, rule_col_list, {"C3": C3_count, "C2": C2_count, "anyvalid": anyvalid_count, "total": total}
    else:
        return eval_complete, C3_list, C2_list, rule_col_list


def preprocess_ids(attr_seq_tsr, ):
    """ Add 1 to all attribute values, such that -1 becomes 0. 
    then the index is valid as token. """
    attr_seq_tsr_pps = attr_seq_tsr.clone() + 1
    return attr_seq_tsr_pps


def get_train_val_seq_data():
    data_dir = '/n/home12/binxuwang/Github/DiffusionReasoning/'
    attr_all = np.load(data_dir+'attr_all.npy')
    print(attr_all.shape)
    attr_all_rows = torch.tensor(attr_all)
    attr_img_tsr = einops.rearrange(attr_all_rows,  'class (B R) p (h w) attr -> class B attr (R h) (p w)', h=3,w=3,p=3,R=3)
    attr_img_tsr_train, attr_img_tsr_val = attr_img_tsr[:, :3950], attr_img_tsr[:, 3950:]
    # Warning, add held out rules and decide if evaluation test on all held outs or just trained ones. 
    attr_seq_tsr_train = einops.rearrange(attr_img_tsr_train,  'class B attr (R h) (p w) -> (class B) (R p h w) attr', h=3,w=3,p=3,R=3)
    attr_seq_tsr_val = einops.rearrange(attr_img_tsr_val,  'class B attr (R h) (p w) -> (class B) (R p h w) attr', h=3,w=3,p=3,R=3)
    attr_seq_tsr_train = preprocess_ids(attr_seq_tsr_train)
    attr_seq_tsr_val = preprocess_ids(attr_seq_tsr_val)
    return attr_seq_tsr_train, attr_seq_tsr_val


# import logging
# def train_gpt2_raven(attr_seq_tsr_train, attr_seq_tsr_val, batch_size=64, 
#                      learning_rate=1e-4, num_epochs=50, model_save_path='gpt2_raven_fixed.pth', 
#                      log_file='training.log', checkpoint_interval=10):
#     logging.basicConfig(filename=log_file, level=logging.INFO)
    
#     gpt2_raven = MultiIdxGPT2Model(attribute_dims=(7,10,10), vocab_size=27, max_length=83, n_embd=768, n_class=0)
#     optimizer = AdamW(gpt2_raven.parameters(), lr=learning_rate)
#     data_loader = torch.utils.data.DataLoader(attr_seq_tsr_train, batch_size=batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(attr_seq_tsr_val, batch_size=256, shuffle=False, drop_last=False)
#     gpt2_raven.train().to('cuda')
    
#     for epoch in range(num_epochs):
#         gpt2_raven.train()
#         pbar = tqdm(data_loader)
#         for inputs in pbar:
#             inputs = inputs.cuda()
#             optimizer.zero_grad()
#             outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=None)
#             loss = multi_attr_loss_vec([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
#             loss.backward()
#             optimizer.step()
#             pbar.set_description(f'Loss: {loss.item()}')
        
#         gpt2_raven.eval()
#         pbar = tqdm(val_loader)
#         val_loss_sum = []
#         for inputs in pbar:
#             inputs = inputs.cuda()
#             with torch.no_grad():
#                 outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=None)
#                 loss = multi_attr_loss([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
#             pbar.set_description(f'Loss: {loss.item()}')
#             val_loss_sum.append(loss.item())
        
#         val_loss_avg = np.mean(val_loss_sum)
#         logging.info(f'Epoch {epoch}, Validation Cross Entropy Loss: {val_loss_avg}')
#         print("Validation Cross Entropy Loss ", val_loss_avg)

#         rnd_idx = np.random.choice(len(attr_seq_tsr_val), 512)
#         eval_samples = attr_seq_tsr_val[rnd_idx,:,:]
#         eval_complete, C3_list, C2_list, rule_col_list = completion_eval(eval_samples, gpt2_raven, num_mask=9, device='cuda', strategy="greedy")
#         torch.save({"eval_complete": eval_complete, 
#                     "C3_list": C3_list, "C2_list": C2_list, 
#                     "rule_col_list": rule_col_list}, 
#                    f"eval_epoch{epoch}_fixed.pt")
        
#         if epoch % checkpoint_interval == 0:
#             torch.save(gpt2_raven.state_dict(), f'{model_save_path}_epoch{epoch}.pth')
    
#     torch.save(gpt2_raven.state_dict(), model_save_path)
#     return gpt2_raven

# %%
# attr_seq_tsr_train, attr_seq_tsr_val = get_train_val_seq_data()
# batch_size = 64
# gpt2_raven = MultiIdxGPT2Model(attribute_dims=(7,10,10), vocab_size=27, max_length=83, n_embd=768, n_class=0)
# # train loop
# optimizer = AdamW(gpt2_raven.parameters(), lr=1e-4)
# # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
# # dataset = torch.utils.data.TensorDataset(attr_seq_tsr_pps)
# data_loader = torch.utils.data.DataLoader(attr_seq_tsr_train, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(attr_seq_tsr_val, batch_size=256, shuffle=False, drop_last=False)
# gpt2_raven.train().to('cuda')
# for epoch in range(50):
#     gpt2_raven.train()
#     pbar = tqdm(data_loader)
#     for inputs in pbar:
#         inputs = inputs.cuda()
#         optimizer.zero_grad()
#         outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=None)
#         # note the inputs were pre-pended in gpt2 to add context
#         loss = multi_attr_loss_vec([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], 
#                                    inputs)
#         # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
#         loss.backward()
#         optimizer.step()
#         pbar.set_description(f'Loss: {loss.item()}')
    
#     # evaluation
#     gpt2_raven.eval()
#     pbar = tqdm(val_loader)
#     val_loss_sum = []
#     for inputs in pbar:
#         inputs = inputs.cuda()
#         with torch.no_grad():
#             outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs)
#             loss = multi_attr_loss([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
#         # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
#         pbar.set_description(f'Loss: {loss.item()}')
#         val_loss_sum.append(loss.item())
#     print("Validation Cross Entropy Loss ",np.mean(val_loss_sum))

#     rnd_idx = np.random.choice(len(attr_seq_tsr_val), 512)
#     eval_samples = attr_seq_tsr_val[rnd_idx,:,:]
#     eval_complete, C3_list, C2_list, rule_col_list = completion_eval(eval_samples, gpt2_raven, num_mask=9, device='cuda', strategy="greedy")
#     torch.save({"eval_complete": eval_complete, 
#                 "C3_list": C3_list, "C2_list": C2_list, 
#                 "rule_col_list": rule_col_list}, 
#                f"eval_epoch{epoch}_fixed.pt")
    
# th.save(gpt2_raven.state_dict(), 'gpt2_raven_fixed.pth')
# #%%
# batch_size = 64
# gpt2_raven = MultiIdxGPT2Model(attribute_dims=(7,10,10), vocab_size=27, max_length=83, 
#                                n_embd=768, n_class=0, n_layer=24, n_head=12)
# # train loop
# optimizer = AdamW(gpt2_raven.parameters(), lr=1e-4)
# # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
# # dataset = torch.utils.data.TensorDataset(attr_seq_tsr_pps)
# data_loader = torch.utils.data.DataLoader(attr_seq_tsr_train, batch_size=batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(attr_seq_tsr_val, batch_size=256, shuffle=False, drop_last=False)
# gpt2_raven.train().to('cuda')
# for epoch in range(1,50):
#     gpt2_raven.train()
#     pbar = tqdm(data_loader)
#     for inputs in pbar:
#         inputs = inputs.cuda()
#         optimizer.zero_grad()
#         outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs, y=None)
#         # note the inputs were pre-pended in gpt2 to add context
#         loss = multi_attr_loss_vec([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], 
#                                    inputs)
#         # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
#         loss.backward()
#         optimizer.step()
#         pbar.set_description(f'Loss: {loss.item()}')
#         # print(loss.item())
    
#     gpt2_raven.eval()
    
#     pbar = tqdm(val_loader)
#     val_loss_sum = []
#     for inputs in pbar:
#         inputs = inputs.cuda()
#         with torch.no_grad():
#             outputs, logits_attr1, logits_attr2, logits_attr3 = gpt2_raven(inputs)
#             loss = multi_attr_loss([logits_attr1[:,:-1], logits_attr2[:,:-1], logits_attr3[:,:-1]], inputs)
#         # loss = next_token_loss((attr_seq_tsr_1, attr_seq_tsr_2, attr_seq_tsr_3), inputs)
#         pbar.set_description(f'Loss: {loss.item()}')
#         # print(loss.item())
#         val_loss_sum.append(loss.item())
#     print("Validation Cross Entropy Loss ",np.mean(val_loss_sum))

#     rnd_idx = np.random.choice(len(attr_seq_tsr_val), 512)
#     eval_samples = attr_seq_tsr_val[rnd_idx,:,:]
#     eval_complete, C3_list, C2_list, rule_col_list = completion_eval(eval_samples, gpt2_raven, num_mask=9, 
#                                             device='cuda', strategy="greedy", batch_size=128)
#     torch.save({"eval_complete": eval_complete, "C3_list": C3_list, "C2_list": C2_list, "rule_col_list": rule_col_list}, 
#                f"eval_epoch{epoch}_fixed_gpt2_Big.pt")
    
# th.save(gpt2_raven.state_dict(), 'gpt2_Big_raven_fixed.pth')

# # %%
# th.cuda.empty_cache()

# # %%
# th.save(gpt2_raven.state_dict(), 'gpt2_raven.pth')
