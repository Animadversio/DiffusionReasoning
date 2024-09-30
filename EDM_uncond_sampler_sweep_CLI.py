import sys
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")
sys.path.append("/n/home12/binxuwang/Github/DiT")
import time
import os
from os.path import join
import pickle as pkl
import torch
import torch as th
from tqdm.auto import tqdm
import numpy as np
import seaborn as sns
import einops
import matplotlib.pyplot as plt
from collections import defaultdict
from easydict import EasyDict as edict
import matplotlib.pyplot as plt 
plt.rcParams['figure.dpi'] = 72
plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
plt.rcParams['figure.facecolor'] = (1, 1, 1, 0)
# vector graphics type
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
from train_edm import create_model, edm_sampler, EDM
from edm_utils import edm_sampler_stoch, edm_sampler_inpaint, create_edm, get_default_config, create_edm_new
from dataset_utils import train_data2attr_tsr,load_raw_data,load_PGM_abstract
from rule_new_utils import check_r3_r2_batch, infer_rule_from_sample_batch, compute_rule_statistics
import circuit_toolkit
from circuit_toolkit.layer_hook_utils import print_specific_layer, get_module_name_shapes, featureFetcher_module

config_mapping = {
    "BaseBlnrX3" : dict(layers_per_block=1, model_channels=64, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "WideBlnrX3" : dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "BigBlnrX3"  : dict(layers_per_block=3, model_channels=192, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
}

import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--expname", type=str, default="090-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20240711-0204")
parser.add_argument("--epoch", type=int, default=999999)
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--reps", type=int, default=10)
args = parser.parse_args()

# expname = r"090-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20240711-0204"
# epoch = 1000000
# batch_size = 2048
# reps = 10
expname = args.expname
epoch = args.epoch
batch_size = args.batch_size
reps = args.reps
device = "cuda"
train_steps = epoch

config_mapping = {
    "BaseBlnrX3" : dict(layers_per_block=1, model_channels=64, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "WideBlnrX3" : dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "BigBlnrX3"  : dict(layers_per_block=3, model_channels=192, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
}

DATASET = "RAVEN10_abstract"
use_ema = True
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
expdir = join(exproot, expname)
ckptdir = join(expdir, "checkpoints")
savedir = join(expdir, "uncond_samples")
os.makedirs(savedir, exist_ok=True)
model_scale = expname.split("_")[0]
config_ft = get_default_config(DATASET, **config_mapping[model_scale])
if epoch == -1:
    edm, model_EDM = create_edm_new(None, config_ft, device) 
    ckpt_str = "ckptRNDINIT"
    print("Random initialization")
else:
    ckpt_path = join(ckptdir, f"ema_{epoch}.pth")
    edm, model_EDM = create_edm_new(ckpt_path, config_ft, device) 
    ckpt_str = f"ckpt{epoch:07d}EMA"
    print(f"Loaded {ckpt_str}: from {ckpt_path}, use_ema: {use_ema}")
    
dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to("cuda")
dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to("cuda")

stoch_kwargs = dict(S_churn=40, S_min=0.05, S_max=50, S_noise=1.003)
for rep in range(reps):
    for stoch_kwargs in [dict(S_churn=40, S_min=0.05, S_max=50, S_noise=1.003),
                     dict(S_churn=80, S_min=0.05, S_max=50, S_noise=1.003),
                     dict(S_churn=30, S_min=0.01, S_max=1, S_noise=1.007),
                     dict(S_churn=80, S_min=0.05, S_max=1, S_noise=1.007),]:
        stoch_str = "_".join([f"{k}{v}" for k, v in stoch_kwargs.items()])
        # Now, let's do the same for the SDE DDPM sampling
        for steps in [5, 10, 20, 50, 100, 200, 500, 1000]:
            print(f"Sampling with Heun{steps}...rep{rep}")
            noise = th.randn(batch_size, 3, 9, 9, device="cuda", generator=th.Generator(device='cuda').manual_seed(rep))
            samples = edm_sampler_stoch(edm, noise, num_steps=steps,
                                        **stoch_kwargs).detach()
            samples = ((samples.detach() * dataset_Xstd) + dataset_Xmean).cpu()
            r3_list, r2_list, rule_col = infer_rule_from_sample_batch(samples)
            C3_count, C2_count, anyvalid_count, total = compute_rule_statistics(r3_list, r2_list, rule_col, verbose=True)
            torch.save(samples, f"{savedir}/{train_steps:07d}_HeunStoch{steps}_{stoch_str}_rep{rep}.pt")
            torch.save({'c3_list': r3_list, 'c2_list': r2_list, 'rule_col': rule_col, 
                    'c3_cnt': C3_count, 'c2_cnt': C2_count, 'anyvalid_cnt': anyvalid_count, 'total': total},
                                f'{savedir}/sample_rule_eval_{train_steps}_HeunStoch{steps}_{stoch_str}_rep{rep}.pt')

# Now, let's do the same for the SDE DDPM sampling
for rep in range(reps):
    for steps in [5, 10, 20, 50, 100, 200, 500, 1000]:
        print(f"Sampling with Heun{steps}...rep{rep}")
        noise = th.randn(batch_size, 3, 9, 9, device="cuda", generator=th.Generator(device='cuda').manual_seed(rep))
        samples = edm_sampler(edm, noise, num_steps=steps).detach()
        samples = edm_sampler_stoch(edm, noise, num_steps=steps,).detach()
        samples = ((samples.detach() * dataset_Xstd) + dataset_Xmean).cpu()
        r3_list, r2_list, rule_col = infer_rule_from_sample_batch(samples)
        C3_count, C2_count, anyvalid_count, total = compute_rule_statistics(r3_list, r2_list, rule_col, verbose=True)
        torch.save(samples, f"{savedir}/{train_steps:07d}_Heun{steps}_rep{rep}.pt")
        torch.save({'c3_list': r3_list, 'c2_list': r2_list, 'rule_col': rule_col, 
                'c3_cnt': C3_count, 'c2_cnt': C2_count, 'anyvalid_cnt': anyvalid_count, 'total': total},
                            f'{savedir}/sample_rule_eval_{train_steps}_Heun{steps}_rep{rep}.pt')
