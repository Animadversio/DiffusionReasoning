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
    compute_memorization_tab_through_training, visualize_memorization_dynamics, visualize_memorization_with_ctrl_dynamics, compute_memorization_binary_joint, extract_training_set_row_panel_sets
from posthoc_analysis_utils import sweep_collect_sample, sweep_collect_eval_data, extract_rule_list_from_eval_col, format_rule_list_to_mat, extract_and_convert

# %%
tabdir = "/n/home12/binxuwang/Github/DiffusionReasoning/Tables"

GPT_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/GPT2_raven"
DiT_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
SiT_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/SiT/results"
SSM_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/Mamba_raven"
EDM_exproot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"

figdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning"
GPTfigdir = join(figdir, "GPT2_raven")
EDMfigdir = join(figdir, "EDM_raven")
DiTfigdir = join(figdir, "DiT_raven")
SiTfigdir = join(figdir, "SiT_raven")
SSMfigdir = join(figdir, "SSM_raven")
figroot = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Figures/DiffusionReasoning/memorization_dynamics"

# %%
# syndf_DiT = pd.read_csv(join(tabdir, "DiT_raven_tensorboard_data_last10.csv"), index_col=0)
# tb_data_col_DiT = pkl.load(open(join(tabdir, "DiT_raven_tensorboard_raw_data.pkl"), "rb"))

# use the non show backend of plt
plt.switch_backend('agg')

syndf_diff = pd.read_csv(join(tabdir, "diffusion_family_comb_tensorboard_data_last10.csv"), index_col=0)    
# %% [markdown]
# ### Mass produce
exproot_dict = {"GPT2": GPT_exproot, "SSM": SSM_exproot, "DiT": DiT_exproot, "EDM": EDM_exproot, "SiT": SiT_exproot}
partial_syndf = syndf_diff.query("step > 900000 and dataset_size == 0.16 and heldout == False and model_class in ['SiT','EDM','DiT']") # 
partial_syndf = syndf_diff.query("step > 900000 and dataset_size in [0.016, 1.6]  and model_class in ['SiT','EDM','DiT']") # and heldout == False
# expfullname = 'GPT2_medium_RAVEN_uncond_heldout0_stream0_16M-20240820-024019/tensorboard_logs'
for _, exprow in partial_syndf.iterrows():
    #expfullname in partial_syndf.full_name.values:
    try:
        expfullname = exprow.full_name
        print(expfullname)
        expname = expfullname.split("/tensorboard_logs")[0]
        figexpdir = join(figroot, expname)
        os.makedirs(figexpdir, exist_ok=True)
        if exprow.model_class in exproot_dict:
            exproot = exproot_dict[exprow.model_class]
        else:
            raise ValueError("Unknown model class")
        if "heldout0" in expfullname:
            heldout_ids = [1, 16, 20, 34, 37]
        else:
            heldout_ids = []
        
        prefix = "sample_rule_eval_" #"eval_step" if "stream" in expname else "eval_epoch"
        eval_col = sweep_collect_eval_data(expname, exproot, prefix=prefix)
        if exprow.model_class == "EDM":
            sample_col = sweep_collect_sample(expname, exproot, prefix="tensor_", non_prefix=prefix, format='tensor_%d.pt')
        else:
            sample_col = sweep_collect_sample(expname, exproot, prefix=None, non_prefix=prefix, format='%07d.pt')
        dataset_size = extract_and_convert(expname)
        examples_per_rule = int(dataset_size * 1E6 / 40)
        print(f"examples_per_rule: {examples_per_rule}")
        if examples_per_rule > 45000:
            continue
        
        train_tsr_X, train_tsr_y = get_RAVEN_dataset(n_classes=40, cmb_per_class=examples_per_rule, heldout_ids=heldout_ids, cmb_offset=0, cache=True)
        train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col = extract_training_set_row_panel_sets(train_tsr_X)
        mem_stats_df = compute_memorization_tab_through_training(sample_col, eval_col, train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col, abinit=True)
        mem_stats_df.to_csv(join(figexpdir, "memorization_stats_train_set.csv"))
        print(mem_stats_df.tail(5))
        figh = visualize_memorization_dynamics(mem_stats_df, expname=expname)
        saveallforms(figexpdir, "memorization_dynamics_train_set", figh)
        
        ctrl_tsr_X, ctrl_tsr_y = get_RAVEN_dataset(n_classes=40, cmb_per_class=examples_per_rule, heldout_ids=heldout_ids, cmb_offset=45000, cache=True)
        ctrl_X_sample_set, ctrl_X_row_set, ctrl_X_panel_set, ctrl_X_row_set_attr_col, ctrl_X_panel_set_attr_col = extract_training_set_row_panel_sets(ctrl_tsr_X)
        mem_stats_ctrl_df = compute_memorization_tab_through_training(sample_col, eval_col, ctrl_X_sample_set, ctrl_X_row_set, ctrl_X_panel_set, ctrl_X_row_set_attr_col, ctrl_X_panel_set_attr_col, abinit=True)
        mem_stats_ctrl_df.to_csv(join(figexpdir, "memorization_stats_ctrl_set.csv"))
        print(mem_stats_ctrl_df.tail(5))
        figh2 = visualize_memorization_dynamics(mem_stats_ctrl_df, expname=expname+' Control set')
        saveallforms(figexpdir, "memorization_dynamics_ctrl_set", figh2)

        figh3 = visualize_memorization_with_ctrl_dynamics(mem_stats_df, mem_stats_ctrl_df, expname=expname)
        saveallforms(figexpdir, "memorization_dynamics_train_ctrl_set_combined", figh3)
        plt.close('all')
    except Exception as e:
        print(e)
        print("Error processing ")
        # print("samples in control set:", examples_per_rule)   
        continue

