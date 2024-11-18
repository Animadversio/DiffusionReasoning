# %% [markdown]
# * Load multiple DiT models of different random seed. 
# * compare weights? should not be the same 
# * compar linkage from initial noise to final outcome

# %%
%load_ext autoreload
%autoreload 2

import sys
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")
sys.path.append("/n/home12/binxuwang/Github/DiT")

# %%
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
# %matplotlib inline
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

from diffusion import create_diffusion
from models import DiT
# %%

DiT_exproot = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results'

# abstract RAVEN dataset
dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to("cuda")
dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to("cuda")
device = "cuda"
diffusion_eval = create_diffusion(timestep_respacing="ddim100")  # default: ddim100



# %%
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

def load_DiT_model(expname, ckpt_step, use_ema=True, 
                   cfg = "DiT_S_1",
                   class_dropout_prob = 1.0,
                   num_classes = 0):
    model_cfg = DiT_configs[cfg]
    model_DiT = DiT(input_size=9,
                in_channels=3, **model_cfg,
                mlp_ratio=4.0,
                class_dropout_prob=class_dropout_prob,
                num_classes=num_classes,
                learn_sigma=True,)

    expdir = join(DiT_exproot, expname)
    ckptdir = join(expdir, "checkpoints")
    ckpt_path = join(ckptdir, f"{ckpt_step}.pt")
    state_dict = th.load(ckpt_path)
    model_DiT.load_state_dict(state_dict['ema' if use_ema else 'model'])
    model_DiT.to("cuda").eval()
    return model_DiT

# %%
def compute_confidence_interval(n, k, confidence=0.95, verbose=True):
    """Compute confidence interval for binomial proportion using beta distribution
    
    Args:
        n (int): Total number of trials
        k (int): Number of successes
        confidence (float): Confidence level (default 0.95)
        verbose (bool): Whether to print the result (default True)
        
    Returns:
        tuple: Lower and upper bounds of confidence interval
    """
    from scipy.stats import beta
    a = k + 1  # alpha parameter for beta distribution
    b = n - k + 1  # beta parameter
    ci = beta.interval(confidence, a, b)
    if verbose:
        print(f"ratio {k/n:.3f} {confidence*100}% CI: ({ci[0]:.3f}, {ci[1]:.3f})")
    return ci


# %%
figroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning"
outdir = join(figroot, "Diffusion_reproducibility_analysis")
os.makedirs(outdir, exist_ok=True)
# %%
expnames = [
    "109-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_rep_20241114-1600",
    "111-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_rep_20241114-1601",
    "113-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_rep_20241114-1601",
    "110-RAVEN10_abstract-uncond-DiT_B_1-stream0_16M_heldout0_rep_20241114-1600",
    "111-RAVEN10_abstract-uncond-DiT_B_1-stream0_16M_heldout0_rep_20241114-1601",
    "112-RAVEN10_abstract-uncond-DiT_B_1-stream0_16M_heldout0_rep_20241114-1601"
]
# %%
batch_size = 10240
RND_SEED = 43
for expname in expnames:
    model_DiT = load_DiT_model(expname, 1000000, use_ema=True, 
                               cfg="DiT_S_1" if "DiT_S_1" in expname else "DiT_B_1")
    noise = th.randn(batch_size, 3, 9, 9, device="cuda", generator=th.Generator(device="cuda").manual_seed(RND_SEED))
    y = th.zeros(batch_size, dtype=torch.int, device="cuda")
    model_kwargs = dict(y=y)
    with th.no_grad():
        samples = diffusion_eval.ddim_sample_loop(model_DiT, noise=noise, shape=(batch_size, 3, 9, 9), 
                                                  clip_denoised=False, device="cuda", model_kwargs=model_kwargs, progress=True)
    samples = ((samples.detach() * dataset_Xstd) + dataset_Xmean).cpu()
    r3_list, r2_list, rule_col = infer_rule_from_sample_batch(samples)
    C3_count, C2_count, anyvalid_count, total = compute_rule_statistics(r3_list, r2_list, rule_col, verbose=True)
    compute_confidence_interval(total, C3_count, confidence=0.95, verbose=True)
    th.save(samples, join(outdir, f"samples_RND{RND_SEED}_{expname}.pth"))
    th.save({"c3": r3_list, "c2": r2_list, "rule": rule_col,
             "C3_count": C3_count, "C2_count": C2_count, "anyvalid_count": anyvalid_count, "total": total
             }, join(outdir, f"eval_samples_RND{RND_SEED}_{expname}.pkl"))

# %%



