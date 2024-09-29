# %load_ext autoreload
# %autoreload 2

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
# from train_edm import create_model, edm_sampler, EDM
# from edm_utils import edm_sampler_inpaint, create_edm, get_default_config
# from rule_utils import get_rule_img, get_obj_list, get_rule_list
# from rule_utils import check_consistent
from dataset_utils import train_data2attr_tsr,load_raw_data,load_PGM_abstract
from rule_new_utils import check_r3_r2_batch, infer_rule_from_sample_batch, compute_rule_statistics
import circuit_toolkit
from circuit_toolkit.layer_hook_utils import print_specific_layer, get_module_name_shapes, featureFetcher_module
### Load in DiT model
DiT_configs = {
    "DiT_XL_1": {"depth": 28, "hidden_size": 1152, "patch_size": 1, "num_heads": 16},
    "DiT_XL_3": {"depth": 28, "hidden_size": 1152, "patch_size": 3, "num_heads": 16},
    "DiT_L_1": {"depth": 24, "hidden_size": 1024, "patch_size": 1, "num_heads": 16},
    "DiT_L_3": {"depth": 24, "hidden_size": 1024, "patch_size": 3, "num_heads": 16},
    "DiT_B_1": {"depth": 12, "hidden_size": 768, "patch_size": 1, "num_heads": 12},
    "DiT_B_3": {"depth": 12, "hidden_size": 768, "patch_size": 3, "num_heads": 12},
    "DiT_S_1": {"depth": 12, "hidden_size": 384, "patch_size": 1, "num_heads": 6},
    "DiT_S_3": {"depth": 12, "hidden_size": 384, "patch_size": 3, "num_heads": 6},
}
from diffusion import create_diffusion
from models import DiT
def load_DiT_model(model_cfg, ckpt_path, device='cuda', 
                   class_dropout_prob=1.0, num_classes=0):
    model_DiT = DiT(input_size=9, in_channels=3, 
                **model_cfg,
                mlp_ratio=4.0,
                class_dropout_prob=class_dropout_prob,
                num_classes=num_classes,
                learn_sigma=True,)
    state_dict = th.load(ckpt_path, )
    model_DiT.load_state_dict(state_dict["model"])
    model_DiT.to(device).eval()
    return model_DiT

import json 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--expname", type=str, default="090-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20240711-0204")
parser.add_argument("--epoch", type=int, default=1000000)
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

exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
expdir = join(exproot, expname)
ckptdir = join(expdir, "checkpoints")
savedir = join(expdir, "uncond_samples")
os.makedirs(savedir, exist_ok=True)

DiTargs = json.load(open(join(expdir, "args.json")))
if DiTargs["cond"]:
    num_classes = DiTargs.num_classes
    class_dropout_prob = DiTargs["class_dropout_prob"]
else:
    num_classes = 0
    class_dropout_prob = 1.0

ckpt_path = join(ckptdir, f"{epoch:07d}.pt")
model_DiT = load_DiT_model(DiT_configs[DiTargs["model"]], ckpt_path, device=device,
                           num_classes=num_classes, class_dropout_prob=class_dropout_prob, )

# Example case.
# device = "cuda"
# class_dropout_prob = 1.0
# num_classes = 0
# model_cfg = DiT_configs["DiT_S_1"]
# model_DiT = DiT(input_size=9,
#             in_channels=3, **model_cfg,
#             mlp_ratio=4.0,
#             class_dropout_prob=class_dropout_prob,
#             num_classes=num_classes,
#             learn_sigma=True,)
# exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
# expname = r"090-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20240711-0204"
# expdir = join(exproot, expname)
# ckptdir = join(expdir, "checkpoints")
# savedir = join(expdir, "uncond_samples")
# os.makedirs(savedir, exist_ok=True)
# ckpt_path = join(ckptdir, "1000000.pt")
# state_dict = th.load(ckpt_path, )
# model_DiT.load_state_dict(state_dict["model"])
# model_DiT.to("cuda").eval();
# abstract RAVEN dataset
dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to("cuda")
dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to("cuda")

y = th.zeros(batch_size, dtype=torch.int, device="cuda")
model_kwargs = dict(y=y)
# Now, let's do the same for the SDE DDPM sampling
for rep in range(reps):
    for steps in [5, 10, 20, 50, 100, 200, 500, 1000]:
        print(f"Sampling with ddpm{steps}...rep{rep}")
        diffusion_sde = create_diffusion(str(steps))
        noise = th.randn(batch_size, 3, 9, 9, device="cuda", generator=th.Generator(device='cuda').manual_seed(rep))
        with th.no_grad():
            samples = diffusion_sde.p_sample_loop(model_DiT, noise.shape, noise=noise, #shape=(batch_size, 3, 9, 9), 
                clip_denoised=False, model_kwargs=model_kwargs, progress=True, device="cuda")
        samples = ((samples.detach() * dataset_Xstd) + dataset_Xmean).cpu()
        r3_list, r2_list, rule_col = infer_rule_from_sample_batch(samples)
        C3_count, C2_count, anyvalid_count, total = compute_rule_statistics(r3_list, r2_list, rule_col, verbose=True)
        torch.save(samples, f"{savedir}/{train_steps:07d}_ddpm{steps}_rep{rep}.pt")
        torch.save({'c3_list': r3_list, 'c2_list': r2_list, 'rule_col': rule_col, 
                'c3_cnt': C3_count, 'c2_cnt': C2_count, 'anyvalid_cnt': anyvalid_count, 'total': total},
                            f'{savedir}/sample_rule_eval_{train_steps}_ddpm{steps}_rep{rep}.pt')

# Do DDIM deterministic sampling. 
for rep in range(reps):
    for steps in [5, 10, 20, 50, 100, 200, 500, 1000]:
        print(f"Sampling with ddim{steps}...rep{rep}")
        diffusion_eval = create_diffusion(timestep_respacing=f"ddim{steps}")  # default: ddim100
        noise = th.randn(batch_size, 3, 9, 9, device="cuda", generator=th.Generator(device='cuda').manual_seed(rep))
        with th.no_grad():
            samples = diffusion_eval.ddim_sample_loop(model_DiT, noise.shape, noise=noise, #shape=(batch_size, 3, 9, 9), 
                clip_denoised=False, device="cuda", model_kwargs=model_kwargs, progress=True)
        samples = ((samples.detach() * dataset_Xstd) + dataset_Xmean).cpu()
        r3_list, r2_list, rule_col = infer_rule_from_sample_batch(samples)
        C3_count, C2_count, anyvalid_count, total = compute_rule_statistics(r3_list, r2_list, rule_col, verbose=True)
        torch.save(samples, f"{savedir}/{train_steps:07d}_ddim{steps}_rep{rep}.pt")
        torch.save({'c3_list': r3_list, 'c2_list': r2_list, 'rule_col': rule_col, 
                'c3_cnt': C3_count, 'c2_cnt': C2_count, 'anyvalid_cnt': anyvalid_count, 'total': total},
                            f'{savedir}/sample_rule_eval_{train_steps}_ddim{steps}_rep{rep}.pt')
        
