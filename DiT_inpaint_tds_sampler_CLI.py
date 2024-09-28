import os
from os.path import join
import argparse
import json
import random
import time
import sys
from functools import partial
import torch as th
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm, trange
from easydict import EasyDict as edict
from torchvision.utils import make_grid
import einops

# Custom modules
sys.path.extend([
    "/n/home12/binxuwang/Github/twisted_diffusion_sampler",
    "/n/home12/binxuwang/Github/twisted_diffusion_sampler/image_exp",
    "/n/home12/binxuwang/Github/mini_edm",
    "/n/home12/binxuwang/Github/DiffusionReasoning",
    "/n/home12/binxuwang/Github/DiT",
])
# tds library
from smc_utils.feynman_kac_pf import smc_FK
from image_diffusion.operators import get_operator, ConditioningMethod
from image_diffusion.script_util import create_gaussian_diffusion
# DiT library
from diffusion import create_diffusion
from models import DiT
# Diffusion Reasoning library
from rule_new_utils import infer_rule_from_sample_batch, compute_rule_statistics
from rule_new_utils import rule_table, relation_dict, attribute_dict, rule_table_brief
from stats_plot_utils import add_rectangles, saveallforms

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


def prepare_RAVEN_dataset(attr_path, device='cuda'):
    train_attrs = np.load(attr_path)
    train_attrs = th.from_numpy(train_attrs).to(int)
    train_row_img = einops.rearrange(train_attrs, 'c s pnl (H W) att -> c s att H (pnl W)', H=3, W=3, att=3, pnl=3)
    train_sample_img = einops.rearrange(train_row_img, 'c (S R) att H W -> c S att (R H) W', R=3,att=3, H=3, W=9)
    labels_tsr = th.arange(len(train_sample_img)).to(int).view(-1,1).repeat(1, train_sample_img.shape[1])
    return train_sample_img, labels_tsr


DiT_configs = {
    "DiT_XL_1": {"depth": 28, "hidden_size": 1152, "patch_size": 1, "num_heads": 16},
    "DiT_XL_3": {"depth": 28, "hidden_size": 1152, "patch_size": 3, "num_heads": 16},
    # "DiT_XL_2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
    # "DiT_XL_4": {"depth": 28, "hidden_size": 1152, "patch_size": 4, "num_heads": 16},
    # "DiT_XL_8": {"depth": 28, "hidden_size": 1152, "patch_size": 8, "num_heads": 16},
    "DiT_L_1": {"depth": 24, "hidden_size": 1024, "patch_size": 1, "num_heads": 16},
    "DiT_L_3": {"depth": 24, "hidden_size": 1024, "patch_size": 3, "num_heads": 16},
    # "DiT_L_2": {"depth": 24, "hidden_size": 1024, "patch_size": 2, "num_heads": 16},
    # "DiT_L_4": {"depth": 24, "hidden_size": 1024, "patch_size": 4, "num_heads": 16},
    # "DiT_L_8": {"depth": 24, "hidden_size": 1024, "patch_size": 8, "num_heads": 16},
    "DiT_B_1": {"depth": 12, "hidden_size": 768, "patch_size": 1, "num_heads": 12},
    "DiT_B_3": {"depth": 12, "hidden_size": 768, "patch_size": 3, "num_heads": 12},
    # "DiT_B_2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
    # "DiT_B_4": {"depth": 12, "hidden_size": 768, "patch_size": 4, "num_heads": 12},
    # "DiT_B_8": {"depth": 12, "hidden_size": 768, "patch_size": 8, "num_heads": 12},
    "DiT_S_1": {"depth": 12, "hidden_size": 384, "patch_size": 1, "num_heads": 6},
    "DiT_S_3": {"depth": 12, "hidden_size": 384, "patch_size": 3, "num_heads": 6},
    # "DiT_S_2": {"depth": 12, "hidden_size": 384, "patch_size": 2, "num_heads": 6},
    # "DiT_S_4": {"depth": 12, "hidden_size": 384, "patch_size": 4, "num_heads": 6},
    # "DiT_S_8": {"depth": 12, "hidden_size": 384, "patch_size": 8, "num_heads": 6},
}

# %%
heldout_id_dict = {
    'train_inputs_new.pt'       : [1, 16, 20, 34, 37], 
    'train_inputs_new_split0.pt': [1, 16, 20, 34, 37], 
    'train_inputs_new_split1.pt': [8, 12, 24, 36, 39],
    'train_inputs_new_split2.pt': [5, 17, 21, 33, 38],
    'train_inputs_new_split3.pt': [3, 10, 29, 31, 37],
    'train_inputs_new_split4.pt': [0, 14, 27, 35, 38],
    'train_inputs_new_split5.pt': [4, 19, 26, 30, 39],
    'train_inputs_new_split6.pt': [9, 13, 25, 32, 37],
    'train_inputs_new_split7.pt': [2, 18, 23, 30, 38],
    'train_inputs_new_split8.pt': [7, 15, 22, 34, 39],
    'train_inputs_new_split9.pt': [6, 11, 28, 33, 37],
}
# heldout_rules = heldout_id_dict["train_inputs_new.pt"]
# %%

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="Process parameters for the inpainting experiment."
    )
    parser.add_argument(
        '--tds_batch_size', type=int, default=25,
        help='Batch size for TDS (default: 25)'
    )
    parser.add_argument(
        '--sample_id_num', type=int, default=50,
        help='Number of sample IDs (default: 50)'
    )
    parser.add_argument(
        '--sample_id_offset', type=int, default=50,
        help='Offset for sample IDs (default: 50)'
    )
    parser.add_argument(
        '--expname', type=str, default="090-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20240711-0204",
        help='Name of the experiment (default: "090-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20240711-0204")'
    )
    parser.add_argument(
        '--epoch', type=int, default=1000000,
        help='Name of the experiment (default: 1000000)'
    )
    parser.add_argument(
        '--tds_step', type=int, default=100,
        help='step of tds sampling (default: 100)'
    )
    
    return parser



parser = create_arg_parser()
args = parser.parse_args()
# Assign parsed arguments to variables
# tds_batch_size = 25
# sample_id_num = 50
# sample_id_offset = 50
# expname = r"045-RAVEN10_abstract-uncond-DiT_S_1_20240311-1256"
# expname = r"090-RAVEN10_abstract-uncond-DiT_S_1-stream0_16M_heldout0_20240711-0204"
tds_batch_size = args.tds_batch_size 
sample_id_num = args.sample_id_num
sample_id_offset = args.sample_id_offset
expname = args.expname
epoch = args.epoch
tds_step = args.tds_step
device = 'cuda'
debug_plot = False
debug_statistics = False

# set up paths 
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
figroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/Figure_inpainting"
expdir = join(exproot, expname)
ckptdir = join(expdir, "checkpoints")
outdir = join(expdir, "inpainting_results")
os.makedirs(outdir, exist_ok=True)
figexpdir = join(figroot, expname)
os.makedirs(figexpdir, exist_ok=True)

# load the model property
DiTargs = json.load(open(join(expdir, "args.json")))
if DiTargs["cond"]:
    num_classes = DiTargs.num_classes
    class_dropout_prob = DiTargs["class_dropout_prob"]
else:
    num_classes = 0
    class_dropout_prob = 1.0

if "heldout_ids" in DiTargs:
    heldout_rules = DiTargs["heldout_ids"]
else:
    heldout_rules = heldout_id_dict[DiTargs["train_attr_fn"]]

ckpt_path = join(ckptdir, f"{epoch:07d}.pt")
model_DiT = load_DiT_model(DiT_configs[DiTargs["model"]], ckpt_path, device=device,
                           num_classes=num_classes, class_dropout_prob=class_dropout_prob, )

attr_path = "/n/home12/binxuwang/Github/DiffusionReasoning/attr_all.npy"
train_sample_img, labels_tsr = prepare_RAVEN_dataset(attr_path, device=device)
dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to(device)
dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to(device)

# %%
# ### configure of the TDS sampler and inpainting task
diffusion_config = {'learn_sigma': True,
'noise_schedule': 'linear',
'timestep_respacing': f'{tds_step}',
'sampler': 'tds',
'use_kl': False,
'predict_xstart': False,
'rescale_timesteps': False,
'rescale_learned_sigmas': False
 }

task_config = edict({'name': 'inpainting',
 'operator': 'inpainting',
 'classifier_guidance_scale': 1.0,
 'pred_xstart_var_type': 6.0,
 'tausq': 0.12,
 })

sampler_config = edict({'name': 'tds',
 'resample_strategy': 'systematic',
 'ess_threshold': 1.0,
 't_truncate_percent': 0,
 })

# %%
# ### Massive production for many samples 
image_shape = (3, 9, 9)
diffusion_RPM_tds = create_gaussian_diffusion(
    steps=1000, #model_cfg["diffusion_steps"], # 1000
    image_shape=image_shape, 
    **diffusion_config, 
)
# Create diffusion object
diffusion_RPM_tds.tausq_ = task_config.tausq 
diffusion_RPM_tds.task = task_config.name 
diffusion_RPM_tds.use_mean_pred = True  
diffusion_RPM_tds.t_truncate = int(diffusion_RPM_tds.T * sampler_config.t_truncate_percent)
operator = get_operator(device=device, name=task_config.operator)  

def model_fn(x, t, y=None, **kwargs):
    return model_DiT(x, t, y, **kwargs) 

stats_col = []
results_col = []
for rule_id in trange(40):
    for sample_id in trange(sample_id_offset, sample_id_offset+sample_id_num):
        # get normalized ref_img 
        ref_img = (train_sample_img[rule_id, sample_id].cuda() - dataset_Xmean[0]) / dataset_Xstd[0]
        # build mask 
        mask = th.ones((3,9,9), dtype=th.bool, device=device)
        mask[:, 6:, 6:] = False
        recon_prob_fn = ConditioningMethod(operator=operator).recon_prob
        measurement = operator(ref_img, mask=mask) # returns a one-dimensional tensor 
        assert mask.shape == ref_img.shape 
        recon_prob_fn = partial(recon_prob_fn, measurement=measurement, mask=mask)
        # resetting 
        diffusion_RPM_tds.mask = mask 
        diffusion_RPM_tds.set_measurement(ref_img*mask) 
        diffusion_RPM_tds.recon_prob_fn = recon_prob_fn 
        diffusion_RPM_tds.clear_cache() 

        M = partial(diffusion_RPM_tds.M, model=model_fn, device=device, 
                    pred_xstart_var_type=task_config.pred_xstart_var_type) 
        G = partial(diffusion_RPM_tds.G, model=model_fn, 
                    debug_plot=debug_plot, debug_statistics=debug_statistics, debug_info={}, 
                    pred_xstart_var_type=task_config.pred_xstart_var_type)

        y = th.zeros(tds_batch_size, dtype=th.int, device=device)
        model_kwargs = dict(y=y)
        # Sampling 
        final_sample, log_w, normalized_w, resample_indices_trace, ess_trace, log_w_trace, xt_trace  = \
            smc_FK(M=M, G=G, 
                resample_strategy=sampler_config.resample_strategy, 
                ess_threshold=sampler_config.ess_threshold, 
                T=diffusion_RPM_tds.T, 
                P=tds_batch_size,  # sampler_config.num_particles
                verbose=False, 
                log_xt_trace=False, 
                extra_vals={"model_kwargs": model_kwargs, 
                            "batch_p": tds_batch_size}) 
        
        final_sample = ((final_sample.detach() * dataset_Xstd) + dataset_Xmean).cpu()
        r3_list, r2_list, rule_col = infer_rule_from_sample_batch(final_sample)
        C3_count, C2_count, anyvalid_count, total = compute_rule_statistics(r3_list, r2_list, rule_col, verbose=False)
        inpaintrow_validcount = anyvalid_count - total * 2
        stats_col.append({"rule_id": rule_id, "sample_id": sample_id, "inpaint_acc": C3_count/total, "valid_acc": inpaintrow_validcount/total,})
        results_col.append({"rule_id": rule_id, "sample_id": sample_id, "final_sample": final_sample, "r3_list": r3_list, "r2_list": r2_list, "rule_col": rule_col,
                            "C3_count": C3_count, "C2_count": C2_count, "valid_count": inpaintrow_validcount, "total": total,})

inpaint_stats_df = pd.DataFrame(stats_col)
savestr = f"ep{epoch}_{sample_id_offset}_{sample_id_offset+sample_id_num}_batch{tds_batch_size}" + (f"_step{tds_step}" if tds_step != 100 else "")
inpaint_stats_df.to_csv(join(outdir, f"inpaint_stats_{savestr}.csv"), index=False)
inpaint_stats_df.to_pickle(join(outdir, f"inpaint_stats_{savestr}.pkl"))
pkl.dump(results_col, open(join(outdir, f"inpaint_results_{savestr}.pkl"), "wb"))

# post hoc synopsys and analysis 
inpaint_acc_tab = inpaint_stats_df.groupby("rule_id").agg({"inpaint_acc": "mean", "valid_acc": "mean", }) #"anyvalid_acc": "mean"

heldout_msk = np.zeros(40, dtype=bool)
heldout_msk[heldout_rules] = 1
print("Overall inpaint accuracy %.3f" % inpaint_acc_tab["inpaint_acc"].mean())
print("- Heldout rule inpaint accuracy %.3f" % inpaint_acc_tab["inpaint_acc"].values[heldout_msk].mean())
print("- Trained rule inpaint accuracy %.3f" % inpaint_acc_tab["inpaint_acc"].values[~heldout_msk].mean())

print("Overall inpaint validity %.3f" % inpaint_acc_tab["valid_acc"].mean())
print("- Heldout rule valid accuracy %.3f" % inpaint_acc_tab["valid_acc"].values[heldout_msk].mean())
print("- Trained rule valid accuracy %.3f" % inpaint_acc_tab["valid_acc"].values[~heldout_msk].mean())

mean_acc = inpaint_acc_tab["inpaint_acc"].mean()
trained_acc = inpaint_acc_tab["inpaint_acc"].values[~heldout_msk].mean()
heldout_acc = inpaint_acc_tab["inpaint_acc"].values[heldout_msk].mean()
plt.figure(figsize=(10, 5))
sns.heatmap(inpaint_acc_tab["inpaint_acc"].values.reshape(4,10), 
            cmap="viridis", annot=True, fmt=".2f")
plt.axis("image")
add_rectangles(heldout_rules)
plt.xticks(np.arange(10)+0.5, [relation_dict[i] for i in range(10)], rotation=45)
plt.yticks(np.arange(4)+0.5, [attribute_dict[i] for i in range(4)], rotation=0)
plt.title(f"Inpainting C3 accuracy Twisted Sampler | Mean {mean_acc:.3f} train {trained_acc:.3f} heldout {heldout_acc:.3f}\n{expname} ep{epoch}")
saveallforms([outdir, figexpdir], f"inpainting_C3_acc_rule_heatmap_{savestr}")
plt.show()

mean_acc = inpaint_acc_tab["valid_acc"].mean()
trained_acc = inpaint_acc_tab["valid_acc"].values[~heldout_msk].mean()
heldout_acc = inpaint_acc_tab["valid_acc"].values[heldout_msk].mean()
plt.figure(figsize=(10, 5))
sns.heatmap(inpaint_acc_tab["valid_acc"].values.reshape(4,10), 
            cmap="viridis", annot=True, fmt=".2f")
plt.axis("image")
add_rectangles(heldout_rules)
plt.xticks(np.arange(10)+0.5, [relation_dict[i] for i in range(10)], rotation=45)
plt.yticks(np.arange(4)+0.5, [attribute_dict[i] for i in range(4)], rotation=0)
plt.title(f"Inpainting row Validity Twisted Sampler | Mean {mean_acc:.3f} train {trained_acc:.3f} heldout {heldout_acc:.3f}\n{expname} ep{epoch}")
saveallforms([outdir, figexpdir], f"inpainting_valid_acc_rule_heatmap_{savestr}")
plt.show()


