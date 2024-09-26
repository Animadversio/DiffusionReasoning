# %%

import os
import re
import json
import pickle as pkl
from os.path import join
from tqdm.auto import trange, tqdm
from os.path import join
from easydict import EasyDict as edict
import numpy as np
import torch as th
import einops
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from os.path import join
import sys
sys.path.append('/n/home12/binxuwang/Github/DiffusionReasoning')
from edm_utils import parse_train_logfile
from dataset_utils import onehot2attr_tsr
from stats_plot_utils import estimate_CI, shaded_error, saveallforms, shaded_error, add_rectangles
# from rule_utils import get_rule_list, get_obj_list, get_rule_img, check_consistent
from rule_new_utils import get_rule_annot
from GPT_models.GPT_RAVEN_model_lib import MultiIdxGPT2Model, completion_eval, seqtsr2imgtsr
from eval_memorization_utils import eval_memorization_all_level_sample_tsr, get_RAVEN_dataset, extract_row_mat_set, extract_panel_mat_set, extract_attr_row_mat_set, extract_attr_panel_mat_set, \
    compute_memorization_tab_through_training, visualize_memorization_dynamics, compute_memorization_binary_joint, extract_training_set_row_panel_sets
from posthoc_analysis_utils import sweep_collect_sample, sweep_collect_eval_data, extract_rule_list_from_eval_col, format_rule_list_to_mat, extract_and_convert


# %%
heldout_ids = [1, 16, 20, 34, 37]

# %%
tabdir = "/n/home12/binxuwang/Github/DiffusionReasoning/Tables"

GPT_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/GPT2_raven"
DiT_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
SSM_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/Mamba_raven"
EDM_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"

figdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning"
GPTfigdir = join(figdir, "GPT2_raven")
EDMfigdir = join(figdir, "EDM_raven")
DiTfigdir = join(figdir, "DiT_raven")
SSMfigdir = join(figdir, "SSM_raven")
figroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/memorization_dynamics"

# %% [markdown]
# %%
syndf_GPT = pd.read_csv(join(tabdir, "GPT_raven_tensorboard_data_last10.csv"), index_col=0)
tb_data_col = pkl.load(open(join(tabdir, "GPT2_raven_tensorboard_raw_data.pkl"), "rb"))

# %% [markdown]
# ### Mass produce

# %%
success_syndf = syndf_GPT.query("step > 900000")
# expfullname = 'GPT2_medium_RAVEN_uncond_heldout0_stream0_16M-20240820-024019/tensorboard_logs'
for expfullname in success_syndf.full_name.values:
    print(expfullname)
    expname = expfullname.split("/tensorboard_logs")[0]
    figexpdir = join(figroot, expname)
    os.makedirs(figexpdir, exist_ok=True)
    
    prefix = "eval_step" if "stream" in expname else "eval_epoch"
    eval_col = sweep_collect_eval_data(expname, GPT_exproot, prefix=prefix)
    sample_col = {epoch: seqtsr2imgtsr(epoch_stats['eval_complete_abinit'], h=3, w=3, p=3, R=3) for epoch, epoch_stats in eval_col.items()}
    dataset_size = extract_and_convert(expname)
    examples_per_rule = int(dataset_size * 1E6 / 40)
    print(f"examples_per_rule: {examples_per_rule}")
    # epoch_list, rule_list_all, consistency_all = extract_rule_list_from_eval_col(eval_col, is_abinit=True) # note set is_abinit to True
    # rule_cnt_mat, cons3_rule_cnt_mat, cons2_rule_cnt_mat = format_rule_list_to_mat(rule_list_all, consistency_all)

    train_tsr_X, train_tsr_y = get_RAVEN_dataset(n_classes=40, cmb_per_class=examples_per_rule, heldout_ids=(), cmb_offset=0, cache=True)
    train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col = extract_training_set_row_panel_sets(train_tsr_X)
    mem_stats_df = compute_memorization_tab_through_training(sample_col, eval_col, train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col, abinit=True)
    mem_stats_df.to_csv(join(figexpdir, "memorization_stats_train_set.csv"))
    print(mem_stats_df.tail(5))
    figh = visualize_memorization_dynamics(mem_stats_df, expname=expname)
    figh.savefig(join(figexpdir, "memorization_dynamics_train_set.png"))
    
    try:
        ctrl_tsr_X, ctrl_tsr_y = get_RAVEN_dataset(n_classes=40, cmb_per_class=examples_per_rule, heldout_ids=heldout_ids, cmb_offset=20000, cache=True)
        ctrl_X_sample_set, ctrl_X_row_set, ctrl_X_panel_set, ctrl_X_row_set_attr_col, ctrl_X_panel_set_attr_col = extract_training_set_row_panel_sets(ctrl_tsr_X)
    except Exception as e:
        print(e)
        print("Not enough control set, skip")
        print("samples in control set:", examples_per_rule)   
        continue
    mem_stats_ctrl_df = compute_memorization_tab_through_training(sample_col, eval_col, ctrl_X_sample_set, ctrl_X_row_set, ctrl_X_panel_set, ctrl_X_row_set_attr_col, ctrl_X_panel_set_attr_col, abinit=True)
    mem_stats_ctrl_df.to_csv(join(figexpdir, "memorization_stats_ctrl_set.csv"))
    print(mem_stats_ctrl_df.tail(5))
    figh2 = visualize_memorization_dynamics(mem_stats_ctrl_df, expname=expname+' Control set')
    saveallforms(figexpdir, "memorization_dynamics_ctrl_set", figh2)

