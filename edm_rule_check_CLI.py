
from functools import partial
import os 
from os.path import join
from tqdm import trange, tqdm
import glob
import time
import copy
import random
from PIL import Image, ImageOps, ImageFilter
import re
import pandas as pd
import einops
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from edm_utils import parse_train_logfile
from dataset_utils import onehot2attr_tsr
from stats_plot_utils import estimate_CI, shaded_error
from rule_utils import get_rule_list, get_obj_list, get_rule_img, check_consistent

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cpu')

def batch_load_samples_infer_rules(samples_dir, epoch_list, encoding="onehot", fmt="tensor_%s.pt"):
    rules_all = []
    for epoch in tqdm(epoch_list): 
        if not os.path.exists(join(samples_dir, fmt % epoch)):
            print(epoch, "not exist")
            break
        samples = torch.load(join(samples_dir, fmt % epoch)) # (batch, 27, 9, 9)
        if encoding == "onehot":
            attr_tsr_list = onehot2attr_tsr(samples, threshold=0.5)
        elif encoding == "digit":
            attr_tsr_list = torch.round(samples).int() # (batch, 3, 9, 9)
        else:
            raise ValueError("encoding should be onehot or digit")
        rules_list = []
        for i, attr_tsr in enumerate(attr_tsr_list): 
            rule_img = get_rule_img(attr_tsr) # (3, 9, 9) -> (3,)
            rules_list.append(rule_img)
        rules_all.append(rules_list)
    rules_all = np.asarray(rules_all) # (201, 25, 3)
    epoch_list = epoch_list[:len(rules_all)]
    
    consistent_mat = []
    for epoch_i in trange(len(rules_all)): 
        consistent_all = [check_consistent(rules) 
                          for rules in rules_all[epoch_i]]
        consistent_mat.append(consistent_all)
    consistent_mat = np.asarray(consistent_mat)
    return rules_all, consistent_mat, epoch_list


def visualize_consistency(epoch_list, consistent_mat, title_str="Wide Dep x3 Blnr", figname="RAVEN10_abstract_BigBlnr", figdir="../Figures", ):
    sample_size = consistent_mat.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5))
    ax.spines[['right', 'top']].set_visible(False)
    CI_low, CI_high = estimate_CI((consistent_mat == 1).sum(axis=1), sample_size, alpha=0.05)
    shaded_error(plt.gca(), epoch_list, (consistent_mat == 1).mean(axis=1),
                    CI_low, CI_high, label="Same in 3 rows", color="C0")
    CI_low, CI_high = estimate_CI((consistent_mat == 2).sum(axis=1), sample_size, alpha=0.05)
    shaded_error(plt.gca(), epoch_list, (consistent_mat==2).mean(axis=1),
                    CI_low, CI_high, label="Same in 2 rows", color="C1")
    ax.set_ylabel('frac of consitent rule\n across rows', fontsize=14)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_title(f"{title_str}, uncond Diffusion")
    ax.legend()
    fig.savefig(join(figdir,f"{figname}_rule_consistency.pdf"), dpi=300, )#bbox_inches='tight')
    fig.savefig(join(figdir,f"{figname}_rule_consistency.png"), dpi=300, )#bbox_inches='tight')
    return fig
    
    
def visualize_rule_validity(epoch_list, rules_all, title_str="Wide Dep x3 Blnr", figname="RAVEN10_abstract_BigBlnr", figdir="../Figures", ):
    sample_size = rules_all.shape[1]
    row_num = rules_all.shape[2]
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5))
    ax.spines[['right', 'top']].set_visible(False)
    CI_low, CI_high = estimate_CI((rules_all != -1).sum(axis=(1,2)), sample_size * row_num, alpha=0.05)
    shaded_error(plt.gca(), epoch_list, (rules_all != -1).mean(axis=(1,2)),
                    CI_low, CI_high, label="valid row", color="C2")
    ax.set_ylabel('frac of valid rule\n among all rows', fontsize=14)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_title(f"{title_str}, uncond Diffusion")
    ax.legend()
    fig.savefig(join(figdir,f"{figname}_rule_valid.pdf"), dpi=300, )#bbox_inches='tight')
    fig.savefig(join(figdir,f"{figname}_rule_valid.png"), dpi=300, )#bbox_inches='tight')
    return fig



def rule_summary_table(rules_all, consistent_mat, epoch_list):
    rule_summary_df = []
    assert rules_all.shape[0] == consistent_mat.shape[0] == len(epoch_list)
    for rule_ep, consistent_ep, epoch in zip(rules_all, consistent_mat, epoch_list): 
        rule_valid = (rule_ep != -1).mean()
        rule_valid_cnt = (rule_ep != -1).sum()
        rule_consistent_1 = (consistent_ep == 1).mean()
        rule_consistent_1_cnt = (consistent_ep == 1).sum()
        rule_consistent_2 = (consistent_ep == 2).mean()
        rule_consistent_2_cnt = (consistent_ep == 2).sum()
        rule_consistent_3 = (consistent_ep == 3).mean()
        rule_consistent_3_cnt = (consistent_ep == 3).sum()
        rule_summary_df.append({"epoch": epoch,
                                "valid": rule_valid.mean(),
                                "valid_cnt": rule_valid_cnt,
                                "cst_1": rule_consistent_1.mean(),
                                "cst_1_cnt": rule_consistent_1_cnt,
                                "cst_2": rule_consistent_2.mean(),
                                "cst_2_cnt": rule_consistent_2_cnt,
                                "cst_3": rule_consistent_3.mean(),
                                "cst_3_cnt": rule_consistent_3_cnt,
                                })
    rule_summary_df = pd.DataFrame(rule_summary_df)
    print(rule_summary_df.tail())
    return rule_summary_df


if __name__ == "__main__":  
    import argparse
    parser = argparse.ArgumentParser(description="Process and visualize EDM rules.")

    # Add arguments
    parser.add_argument("--ep_start", type=int, default=0, help="Start of the epoch range.")
    parser.add_argument("--ep_stop", type=int, default=1000000, help="End of the epoch range.")
    parser.add_argument("--ep_step", type=int, default=5000, help="Step size for the epoch range.")
    parser.add_argument("--expname", type=str, required=True, help="Name of the experiment.")
    parser.add_argument("--exproot", type=str, default="/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/mini_edm/exps", help="Root directory for the experiment.")
    parser.add_argument("--encoding", type=str, choices=["onehot", "digit"], default="digit", help="Encoding method for processing samples.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Compute device to use.")
    parser.add_argument("--figname", type=str, default="", help="Name of the figure to save.")
    parser.add_argument("--figdir", type=str, default="Figures", help="Directory to save the figure.")
    parser.add_argument("--title_str", type=str, default="", help="Title of the figure.")
    parser.add_argument("--fmt", type=str, default="tensor_%s.pt", help="Format of the sample files.")
    parser.add_argument("--update", action="store_true", help="Update the existing inferred rules.")
                        

    # Parse arguments
    args = parser.parse_args()
    # exproot = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/mini_edm/exps"
    # expname = "WideBlnr_RAVEN10_abstract_onehot_20240211-1743"
    # epoch_list = list(np.arange(0, 1000000, 5000))
    exproot = args.exproot
    expname = args.expname
    epoch_list = list(np.arange(args.ep_start, args.ep_stop, args.ep_step))
    encoding = args.encoding
    figdir = args.figdir
    figname = expname if args.figname == "" else args.figname
    title_str = expname.replace("_"," ") if args.title_str == "" else args.title_str
    
    expdir = join(exproot, expname)
    print("Analyzing", expname)
    if os.path.exists(join(expdir, "std.log")):
        df = parse_train_logfile(join(expdir, "std.log"))
        df.plot(x="step", y=["average_loss","batch_loss"], alpha=0.75)
        plt.savefig(join(expdir, "loss_curve.png"))
    
    if os.path.exists(join(expdir, "samples_inferred_rule_consistency.npz")) and args.update:
        print("Inferred rules already exist. Loading from file.")
        npzfile = np.load(join(expdir, "samples_inferred_rule_consistency.npz"))
        rules_all_load = npzfile["rules_all"]
        consistent_mat_load = npzfile["consistent_mat"]
        epoch_list_load = npzfile["epoch_list"]
        epoch_rest = [epoch for epoch in epoch_list if epoch not in epoch_list_load]
        print(rules_all_load.shape, consistent_mat_load.shape, len(epoch_rest))
        print("Updating inferred rules, starting from epoch: ", epoch_rest[0])
        rules_all_rest, consistent_mat_rest, epoch_rest = batch_load_samples_infer_rules(
            join(expdir, "samples"), epoch_rest, encoding=encoding, fmt=args.fmt)
        if len(epoch_rest) == 0 and len(rules_all_rest) == 0:
            rules_all_wide, consistent_mat_wide, epoch_list = rules_all_load, consistent_mat_load, epoch_list_load
        else:
            rules_all_wide = np.concatenate([rules_all_load, rules_all_rest], axis=0)
            consistent_mat_wide = np.concatenate([consistent_mat_load, consistent_mat_rest], axis=0)
            epoch_list = np.concatenate([epoch_list_load, epoch_rest], axis=0)
            print("Inferred rules updated.")
        print(rules_all_wide.shape, consistent_mat_wide.shape, len(epoch_list))
    else:
        rules_all_wide, consistent_mat_wide, epoch_list = batch_load_samples_infer_rules(
            join(expdir, "samples"), epoch_list, encoding=encoding, fmt=args.fmt)
    
    np.savez(join(expdir, "samples_inferred_rule_consistency.npz"), 
            consistent_mat=consistent_mat_wide, 
            rules_all=rules_all_wide, epoch_list=epoch_list)
    
    rule_summary_df = rule_summary_table(rules_all_wide, consistent_mat_wide, epoch_list)
    
    visualize_consistency(epoch_list, consistent_mat_wide, 
                        title_str=title_str, figname=figname, figdir=figdir,)

    visualize_rule_validity(epoch_list, rules_all_wide, 
                            title_str=title_str, figname=figname, figdir=figdir,);
