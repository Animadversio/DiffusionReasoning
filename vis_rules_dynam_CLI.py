import sys
# sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")
import os
from tqdm import trange
from os.path import join
import pickle as pkl
import torch
from easydict import EasyDict as edict
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import einops
import json
from copy import deepcopy
from stats_plot_utils import saveallforms
row_table = {
0: "Constant",
1: "Progression neg 2",
2: "Progression neg 1",
3: "Progression pos 1",
4: "Progression pos 2",
5: "Arithmetic pos",
6: "Arithmetic neg",
7: "XOR",
8: "OR",
9: "AND"
}
col_table =  {0: "Shape", 1: "Size", 2: "Color", 3: "Number", 4: "Position"}
entry_table = {}
for i in range(40):
    if i < 37:
        entry_table[i] = f"{row_table[i%10]}-{col_table[i//10]}"
    else:
        entry_table[i] = f"{row_table[i%10]}-{col_table[i//10+1]}"
# print(entry_table)

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

def visualize_indiv_rule_dynam(rule_mat, conv_wid=10, heldout_id=[1, 16, 20, 34, 37],
                               titlestr="Valid rule count separated by rule type", ylabel="Count", axs=None):
    # remove top and right spines from plot with plt
    plt.rcParams.update({'font.size': 12})
    if axs is None:
        figh, axs = plt.subplots(4, 10, figsize=(30, 12.5), sharex=True, sharey=True)
    else:
        figh = axs[0,0].get_figure()
    axs_f = axs.flatten()
    for i in range(40):
        ax = axs_f[i]
        # smooth the curve
        smooth_rule_cnt = np.convolve(rule_mat[:,i], np.ones(conv_wid)/conv_wid, mode='same')
        ax.plot(data['epoch_list'], smooth_rule_cnt, alpha=0.7, )
        ax.set_title(f"R{i}: {entry_table[i]}")
        # change the font color of title to red
        if i in heldout_id:
            ax.title.set_color('red')
        if i >= 30:
            ax.set_xlabel("generation")
        if i % 10 == 0:
            ax.set_ylabel(ylabel)
    figh.suptitle(titlestr, fontsize=20)
    figh.tight_layout()
    figh.show()
    return figh, axs

import argparse
parser = argparse.ArgumentParser(description="Process and visualize EDM rules.")
parser.add_argument("--expname", type=str, required=True, help="Name of the experiment.")
parser.add_argument("--exproot", type=str, default="/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/mini_edm/exps", help="Root directory for the experiment.")
args = parser.parse_args()
exproot = args.exproot
expname = args.expname

expdir = join(exproot, expname)
if os.path.exists(join(expdir, "args.json")):
    with open(join(expdir, "args.json"), 'r') as f:
        args = edict(json.load(f))
    train_data_fn = args['train_attr_fn']
    heldout_id = heldout_id_dict[train_data_fn]
    print(f"Found args.json file. Using heldout_id: {heldout_id} from {train_data_fn}.")
else:
    print("No args.json file found. Using default values.")
    heldout_id = [1, 16, 20, 34, 37]
# expdir = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps/BigBlnrX3_new_RAVEN10_abstract_20240315-1328"

data = np.load(join(expdir, "samples_inferred_rule_consistency_new.npz"), allow_pickle=True)
list(data.keys())
rule_list_all = deepcopy(data['rule_list_all'])
consistency_all = deepcopy(data['consistency_all'])

# each element in the rule rule_pool_all is a list of rules for all the samples in the corresponding generation
rule_pool_all = []
for i in trange(len(data['epoch_list'])):
    rule_pool = np.concatenate(list(rule_list_all[i,:,:].flatten())).astype(int)
    rule_pool_all.append(rule_pool)
# plot the number of rules == rule_i for each generation
rule_cnt_mat = np.zeros((len(data['epoch_list']), 40))
for i in trange(len(data['epoch_list'])):
    rule_pool = rule_pool_all[i]
    rule_uniq, counts = np.unique(rule_pool, return_counts=True)
    rule_cnt_mat[i, rule_uniq] = counts

cons3_rule_pool_all = []
cons2_rule_pool_all = []
for i in trange(len(data['epoch_list'])):
    rule_pool = np.concatenate(list(consistency_all[i,0,:].flatten())).astype(int)
    cons3_rule_pool_all.append(rule_pool)
    rule_pool = np.concatenate(list(consistency_all[i,1,:].flatten())).astype(int)
    cons2_rule_pool_all.append(rule_pool)

cons3_rule_cnt_mat = np.zeros((len(data['epoch_list']), 40))
cons2_rule_cnt_mat = np.zeros((len(data['epoch_list']), 40))
for i in trange(len(data['epoch_list'])):
    rule_pool = cons3_rule_pool_all[i]
    rule_uniq, counts = np.unique(rule_pool, return_counts=True)
    cons3_rule_cnt_mat[i, rule_uniq] = counts
    rule_pool = cons2_rule_pool_all[i]
    rule_uniq, counts = np.unique(rule_pool, return_counts=True)
    cons2_rule_cnt_mat[i, rule_uniq] = counts
    
figdir = "/n/home12/binxuwang/Github/DiffusionReasoning/Figures_newrule"
figh, axs = visualize_indiv_rule_dynam(rule_cnt_mat, conv_wid=10, heldout_id=heldout_id,
                        titlestr=f"{expname}\nValid rule count separated by rule type")
saveallforms(figdir, f"{expname}_indiv_rule_validity", figh)


figh, axs = visualize_indiv_rule_dynam(cons3_rule_cnt_mat, conv_wid=10, heldout_id=heldout_id,
                                       titlestr=f"{expname}\nConsistency 3 (blue) and 2 (orange) rule count", )
figh, axs = visualize_indiv_rule_dynam(cons2_rule_cnt_mat, conv_wid=10, heldout_id=heldout_id,
                                       titlestr=f"{expname}\nConsistency 3 (blue) and 2 (orange) rule count", axs=axs)
saveallforms(figdir, f"{expname}_indiv_rule_consistency", figh)

