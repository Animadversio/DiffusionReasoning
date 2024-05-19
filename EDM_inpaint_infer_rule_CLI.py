# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")
import os
from os.path import join
import pickle as pkl
import einops
import torch
import torch as th
from easydict import EasyDict as edict
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from train_edm import create_model, edm_sampler, EDM
from edm_utils import edm_sampler_inpaint, create_edm, get_default_config, create_edm_new
from tqdm import trange, tqdm
# from rule_utils import get_rule_img, get_obj_list, get_rule_list
# from rule_utils import check_consistent
from dataset_utils import train_data2attr_tsr,load_raw_data,load_PGM_abstract
from rule_new_utils import infer_rule_from_sample_batch, check_r3_r2_batch

# %%
plt.rcParams['figure.dpi'] = 72
plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
plt.rcParams['figure.facecolor'] = (1, 1, 1, 0)
# vector graphics type
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# %%
device = "cuda"
# %%
# train_attrs = torch.load('/n/home12/binxuwang/Github/DiffusionReasoning/train_inputs_new.pt')
attr_all = np.load('/n/home12/binxuwang/Github/DiffusionReasoning/attr_all.npy')
attr_all = torch.from_numpy(attr_all)
# abstract RAVEN dataset
dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to("cuda")
dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to("cuda")

# %%
@torch.no_grad()
def edm_sampler_inpaint(
    edm, latents, target_img, mask, class_labels=None,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    use_ema=True, fixed_noise=False
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, edm.sigma_min)
    sigma_max = min(sigma_max, edm.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([edm.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    initial_noise = torch.randn_like(latents)
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        # x_hat = x_next
        t_hat = t_cur
        noise_perturb = initial_noise if fixed_noise else torch.randn_like(target_img)
        x_hat = (1 - mask[None, None]) * (target_img + noise_perturb * t_cur) + \
                     mask[None, None]  * x_next
        # Euler step.
        denoised = edm(x_hat, t_hat, class_labels, use_ema=use_ema).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = edm(x_next, t_next, class_labels, use_ema=use_ema).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
# expname = "WideBlnrX3_new_RAVEN10_abstract_20240315-1327"
expname = "WideBlnrX3_new_RAVEN10_abstract_20240412-1347"
epoch = 100000 # 999999 #700000 # 100000

DATASET = "RAVEN10_abstract"
expdir = join(exproot, expname)
ckptdir = join(expdir, "checkpoints")
savedir = join(expdir, "inpaint_pilot")
os.makedirs(savedir, exist_ok=True)

config_ft = get_default_config(DATASET, layers_per_block=2, 
                               model_channels=128, 
                               channel_mult=[1, 2, 4], 
                               attn_resolutions=[9, 3], 
                               spatial_matching="bilinear")
ckpt_path = join(ckptdir, f"ema_{epoch}.pth")
edm, model_EDM = create_edm_new(ckpt_path, config_ft, device) 

# %%
nreps = 50
batch_size = 100
total_steps = 18
# suffix = f"_Heun{total_steps}"
suffix = f"_Heun{total_steps}_ep{epoch}"
# model_kwargs = dict(y=th.zeros(batch_size, dtype=th.int, device="cuda"))
# diffusion_eval = create_diffusion(timestep_respacing="ddim200")  # default: ddim100
sample_col = []
meta_col = []
rule_all_col = []
stats_col = []
# iclass = 0
for iclass in range(40):
    idx_seq = np.random.choice(12000, 3 * nreps, replace=False)
    for ireps in trange(nreps):
        idxs = idx_seq[ireps*3:(ireps+1)*3]
        # idxs = np.random.randint(0, 12000, 3)#np.random.choice(12000, 3)
        attr_tsr = train_data2attr_tsr(attr_all[iclass, idxs])
        attr_mtg = einops.rearrange(attr_tsr, "row attr h PW -> attr (row h) PW")
        r3_list, r2_list, rule_col = infer_rule_from_sample_batch(attr_mtg)
        print(r3_list, "row idxs", list(idxs), rule_col)
        # assert np.all(orig_rules != -1)
        assert len(r3_list[0]) == 1
        orig_rule_id = r3_list[0][0]
        target_img = (attr_mtg.float().to(device) - dataset_Xmean) / dataset_Xstd
        for x_pos, y_pos in [(0, 2), (1, 2), (2, 2),]:
            mask_tsr = torch.zeros(target_img.shape[-2:], device=device)
            mask_tsr[3*x_pos:3*(x_pos+1), 3*y_pos:3*(y_pos+1)] = 1
            # samples_inpaint = ddim_sample_inpainting_loop(diffusion_eval, model, (batch_size, 3, 9, 9), target_img, mask_tsr, noise=None, clip_denoised=True, fixed_noise=False,
            #                                             model_kwargs=model_kwargs, device="cuda", progress=False, eta=0.0)
            latents = torch.randn(batch_size, 3, 9, 9).to(device)
            samples_inpaint = edm_sampler_inpaint(edm, latents, 
                                target_img=target_img, mask=mask_tsr,
                                num_steps=total_steps, use_ema=False,
                                fixed_noise=False).detach()
            samples_inpaint = ((samples_inpaint.detach() * dataset_Xstd) + dataset_Xmean).cpu()
            r3_list_ipt, r2_list_ipt, rule_col_ipt = infer_rule_from_sample_batch(samples_inpaint)
            consistent_r3 = sum([orig_rule_id in cst_rule_set for cst_rule_set in r3_list_ipt])
            inpaint_rule_col = np.concatenate(tuple(np.array(rule_col_ipt, dtype=object)[:, x_pos]))
            if len(inpaint_rule_col):
                uniq, counts = np.unique(inpaint_rule_col, return_counts=True)
                # sort by counts
                idx_sort = np.argsort(counts)[::-1]
                uniq_sort = uniq[idx_sort]
                counts_sort = counts[idx_sort]
            else:
                uniq_sort = []
                counts_sort = []
            print(f"Rule [{orig_rule_id}] id[{idxs[x_pos]}] Consistent rules ratio inpainting samples ({x_pos}, {y_pos}): {consistent_r3}/{batch_size} ({consistent_r3/batch_size:.3f})")
            meta_col.append({"iclass": iclass, "idxs": list(idxs), "rule": rule_col, "x_pos": x_pos, "y_pos": y_pos})
            sample_col.append(samples_inpaint)
            rule_all_col.append((r3_list_ipt, r2_list_ipt, rule_col_ipt))
            stats_col.append({"iclass": iclass, "idxs": list(idxs), "x_pos": x_pos, "y_pos": y_pos, "consistent_r3": consistent_r3, "batch_size": batch_size, 
                              "rule_uniq": uniq_sort, "rule_cnts": counts_sort}) # "rule": rule_col, 
        print(f"iclass {iclass} done {ireps}/{nreps}")
        df = pd.DataFrame(stats_col)
        df.to_csv(join(savedir, f"inpaint_rule_stats{suffix}.csv"))
                
    pkl.dump(sample_col, open(join(savedir, f"inpaint_sample_col{suffix}.pkl"), "wb"))
    pkl.dump(meta_col, open(join(savedir, f"inpaint_meta_col{suffix}.pkl"), "wb"))
    pkl.dump(rule_all_col, open(join(savedir, f"inpaint_rule_all_col{suffix}.pkl"), "wb"))
    pkl.dump(stats_col, open(join(savedir, f"inpaint_stats_col{suffix}.pkl"), "wb"))
    
df = pd.DataFrame(stats_col)
df.to_csv(join(savedir, f"inpaint_rule_stats{suffix}.csv"))

# %%
# batch_size = 100
# suffix = "_unif_baseline"
# total_steps = 35
# # model_kwargs = dict(y=th.zeros(batch_size, dtype=th.int, device="cuda"))
# # diffusion_eval = create_diffusion(timestep_respacing="ddim200")  # default: ddim100
# nreps = 50
# sample_col = []
# meta_col = []
# rule_all_col = []
# stats_col = []
# # iclass = 0
# for iclass in range(40):
#     idx_seq = np.random.choice(12000, 3 * nreps, replace=False)
#     for ireps in range(nreps):
#         idxs = idx_seq[ireps*3:(ireps+1)*3]
#         # idxs = np.random.randint(0, 12000, 3)#np.random.choice(12000, 3)
#         attr_tsr = train_data2attr_tsr(attr_all[iclass, idxs])
#         attr_mtg = einops.rearrange(attr_tsr, "row attr h PW -> attr (row h) PW")
#         r3_list, r2_list, rule_col = infer_rule_from_sample_batch(attr_mtg)
#         print(r3_list, "row idxs", list(idxs), rule_col)
#         # assert np.all(orig_rules != -1)
#         assert len(r3_list[0]) == 1
#         orig_rule_id = r3_list[0][0]
#         target_img = (attr_mtg.float().to(device) - dataset_Xmean) / dataset_Xstd
#         for x_pos, y_pos in [(0, 2), (1, 2), (2, 2),]:
#             mask_tsr = torch.zeros(target_img.shape[-2:], device=device)
#             mask_tsr[3*x_pos:3*(x_pos+1), 3*y_pos:3*(y_pos+1)] = 1
#             # samples_inpaint = ddim_sample_inpainting_loop(diffusion_eval, model, (batch_size, 3, 9, 9), target_img, mask_tsr, noise=None, clip_denoised=True, fixed_noise=False,
#             #                                             model_kwargs=model_kwargs, device="cuda", progress=False, eta=0.0)
#             inpaint_part = th.cat([th.randint(-1, 7, (batch_size, 1, 3, 3), device=device),
#                                    th.randint(-1, 10, (batch_size, 2, 3, 3), device=device)], dim=1)
#             vacant_mask = (inpaint_part == -1).any(dim=1, keepdim=True)
#             # for any entry that is -1, set the entire row (dim=1) to -1
#             inpaint_part[vacant_mask.expand(-1, 3, -1, -1)] = -1
            
#             samples_inpaint = attr_mtg.repeat(batch_size, 1, 1, 1)
#             samples_inpaint[:, :, 3*x_pos:3*(x_pos+1), 3*y_pos:3*(y_pos+1)] = inpaint_part
#             # raise ValueError("Not implemented")
#             # latents = torch.randn(batch_size, 3, 9, 9).to(device)
#             # samples_inpaint = edm_sampler_inpaint(edm, latents, 
#             #                     target_img=target_img, mask=mask_tsr,
#             #                     num_steps=total_steps, use_ema=False,
#             #                     fixed_noise=False).detach()
#             # samples_inpaint = ((samples_inpaint.detach() * dataset_Xstd) + dataset_Xmean).cpu()
#             r3_list_ipt, r2_list_ipt, rule_col_ipt = infer_rule_from_sample_batch(samples_inpaint)
#             consistent_r3 = sum([orig_rule_id in cst_rule_set for cst_rule_set in r3_list_ipt])
#             inpaint_rule_col = np.concatenate(tuple(np.array(rule_col_ipt, dtype=object)[:, x_pos]))
#             if len(inpaint_rule_col):
#                 uniq, counts = np.unique(inpaint_rule_col, return_counts=True)
#                 # sort by counts
#                 idx_sort = np.argsort(counts)[::-1]
#                 uniq_sort = uniq[idx_sort]
#                 counts_sort = counts[idx_sort]
#             else:
#                 uniq_sort = []
#                 counts_sort = []
#             print(f"Rule [{orig_rule_id}] id[{idxs[x_pos]}] Consistent rules ratio inpainting samples ({x_pos}, {y_pos}): {consistent_r3}/{batch_size} ({consistent_r3/batch_size:.3f})")
#             meta_col.append({"iclass": iclass, "idxs": list(idxs), "rule": rule_col, "x_pos": x_pos, "y_pos": y_pos})
#             sample_col.append(samples_inpaint)
#             rule_all_col.append((r3_list_ipt, r2_list_ipt, rule_col_ipt))
#             stats_col.append({"iclass": iclass, "idxs": list(idxs), "x_pos": x_pos, "y_pos": y_pos, "consistent_r3": consistent_r3, "batch_size": batch_size, 
#                               "rule_uniq": uniq_sort, "rule_cnts": counts_sort}) # "rule": rule_col, 
#         print(f"iclass {iclass} done {ireps}/{nreps}")
#         df = pd.DataFrame(stats_col)
#         df.to_csv(join(savedir, f"inpaint_rule_stats{suffix}.csv"))
                
#     pkl.dump(sample_col, open(join(savedir, f"inpaint_sample_col{suffix}.pkl"), "wb"))
#     pkl.dump(meta_col, open(join(savedir, f"inpaint_meta_col{suffix}.pkl"), "wb"))
#     pkl.dump(rule_all_col, open(join(savedir, f"inpaint_rule_all_col{suffix}.pkl"), "wb"))
#     pkl.dump(stats_col, open(join(savedir, f"inpaint_stats_col{suffix}.pkl"), "wb"))
    
# df = pd.DataFrame(stats_col)
# df.to_csv(join(savedir, f"inpaint_rule_stats{suffix}.csv"))

