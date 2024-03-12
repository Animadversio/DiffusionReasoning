
import sys
# sys.path.append("../")
from functools import partial
import os 
from os.path import join
from tqdm import trange, tqdm
import re
import glob
import time
import copy
import random
from PIL import Image, ImageOps, ImageFilter
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
from collections import defaultdict

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cpu')
from dataset_utils import onehot2attr_tsr
from stats_plot_utils import estimate_CI, shaded_error
from rule_new_utils import check_r3_r2_batch
from edm_utils import parse_train_logfile
from rule_utils import get_rule_list, get_obj_list, get_rule_img, check_consistent


def batch_load_samples(samples_dir, epoch_list, encoding="onehot", fmt="tensor%s.pt"):
    sample_all = []
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
        sample_all.append(attr_tsr_list)
    epoch_list = epoch_list[:len(sample_all)]
    return sample_all, epoch_list


def samples_infer_rule_consistency(sample_all, epoch_list, ):
    stats_col = []
    rule_list_all = []
    consistency_all = []
    pbar = tqdm(enumerate(epoch_list))
    for i, epoch in pbar: 
        # print(i, epoch)
        batchsize = sample_all[i].shape[0]
        sample_batch = sample_all[i] # [1024, 3, 9, 9]
        sample_batch = sample_batch.view(-1, 3, 3, 3, 9) 
        sample_batch = einops.rearrange(sample_batch, 
                "B attr row h (panel w) -> B row panel (h w) attr", 
                                    panel=3, w=3, h=3, attr=3)
        r3_list, r2_list, rule_col = check_r3_r2_batch(sample_batch)
        r3_count = sum([len(x) > 0 for x in r3_list])
        r2_count = sum([len(x) > 0 for x in r2_list])
        rule_flatten = np.array(rule_col, dtype=object).flatten() # [3 * 1024]
        anyvalid_count = sum([len(x) > 0 for x in rule_flatten])
        stats_col.append({"epoch": epoch, "r3_count": r3_count, "r2_count": r2_count, 
                        "valid_count": anyvalid_count, "sample_count": batchsize})
        rule_list_all.append((rule_col))
        consistency_all.append((r3_list, r2_list))
        pbar.set_postfix({"epoch": epoch, "r3_count": r3_count, "r2_count": r2_count})

    stats_df = pd.DataFrame(stats_col)
    return stats_df, rule_list_all, consistency_all


# def batch_load_samples_infer_rules(samples_dir, epoch_list, encoding="onehot", fmt="tensor_%s.pt"):
#     rules_all = []
#     for epoch in tqdm(epoch_list): 
#         if not os.path.exists(join(samples_dir, fmt % epoch)):
#             print(epoch, "not exist")
#             break
#         samples = torch.load(join(samples_dir, fmt % epoch)) # (batch, 27, 9, 9)
#         if encoding == "onehot":
#             attr_tsr_list = onehot2attr_tsr(samples, threshold=0.5)
#         elif encoding == "digit":
#             attr_tsr_list = torch.round(samples).int() # (batch, 3, 9, 9)
#         else:
#             raise ValueError("encoding should be onehot or digit")
#         rules_list = []
#         for i, attr_tsr in enumerate(attr_tsr_list): 
#             rule_img = get_rule_img(attr_tsr) # (3, 9, 9) -> (3,)
#             rules_list.append(rule_img)
#         rules_all.append(rules_list)
#     rules_all = np.asarray(rules_all) # (201, 25, 3)
#     epoch_list = epoch_list[:len(rules_all)]
    
#     consistent_mat = []
#     for epoch_i in trange(len(rules_all)): 
#         consistent_all = [check_consistent(rules) 
#                           for rules in rules_all[epoch_i]]
#         consistent_mat.append(consistent_all)
#     consistent_mat = np.asarray(consistent_mat)
#     return rules_all, consistent_mat, epoch_list


def visualize_consistency_new(stats_df, title_str="Wide Dep x3 Blnr", figname="RAVEN10_abstract_BigBlnr", figdir="../Figures", ):
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5))
    ax.spines[['right', 'top']].set_visible(False)
    CI_low, CI_high = estimate_CI(stats_df.r3_count, stats_df.sample_count, alpha=0.05)
    shaded_error(plt.gca(), stats_df.epoch, stats_df.r3_count / stats_df.sample_count,
                    CI_low, CI_high, label="Same in 3 rows", color="C0")
    CI_low, CI_high = estimate_CI(stats_df.r2_count, stats_df.sample_count, alpha=0.05)
    shaded_error(plt.gca(), stats_df.epoch, stats_df.r2_count / stats_df.sample_count,
                    CI_low, CI_high, label="Same in 2 rows", color="C1")
    ax.set_ylabel('frac of consitent rule\n across rows', fontsize=14)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_title(f"{title_str}, uncond Diffusion")
    ax.legend()
    if figdir is not None:
        fig.savefig(join(figdir, f"{figname}_new_rule_consistency.pdf"), dpi=300, )#bbox_inches='tight')
        fig.savefig(join(figdir, f"{figname}_new_rule_consistency.png"), dpi=300, )#bbox_inches='tight')
    return fig
    
    
def visualize_rule_validity_new(stats_df, title_str="Wide Dep x3 Blnr", figname="RAVEN10_abstract_BigBlnr", figdir="../Figures", ):
    row_num = 3
    fig, ax = plt.subplots(1, 1, figsize=(6,4.5))
    ax.spines[['right', 'top']].set_visible(False)
    CI_low, CI_high = estimate_CI(stats_df.valid_count, 
                                  stats_df.sample_count * row_num, alpha=0.05)
    shaded_error(plt.gca(), stats_df.epoch, stats_df.valid_count / stats_df.sample_count / row_num,
                    CI_low, CI_high, label="valid row", color="C2")
    ax.set_ylabel('frac of valid rule\n among all rows', fontsize=14)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_title(f"{title_str}, uncond Diffusion")
    ax.legend()
    if figdir is not None:
        fig.savefig(join(figdir,f"{figname}_new_rule_valid.pdf"), dpi=300, )#bbox_inches='tight')
        fig.savefig(join(figdir,f"{figname}_new_rule_valid.png"), dpi=300, )#bbox_inches='tight')
    return fig


# def rule_summary_table(rules_all, consistent_mat, epoch_list):
#     rule_summary_df = []
#     assert rules_all.shape[0] == consistent_mat.shape[0] == len(epoch_list)
#     for rule_ep, consistent_ep, epoch in zip(rules_all, consistent_mat, epoch_list): 
#         rule_valid = (rule_ep != -1).mean()
#         rule_valid_cnt = (rule_ep != -1).sum()
#         rule_consistent_1 = (consistent_ep == 1).mean()
#         rule_consistent_1_cnt = (consistent_ep == 1).sum()
#         rule_consistent_2 = (consistent_ep == 2).mean()
#         rule_consistent_2_cnt = (consistent_ep == 2).sum()
#         rule_consistent_3 = (consistent_ep == 3).mean()
#         rule_consistent_3_cnt = (consistent_ep == 3).sum()
#         rule_summary_df.append({"epoch": epoch,
#                                 "valid": rule_valid.mean(),
#                                 "valid_cnt": rule_valid_cnt,
#                                 "cst_1": rule_consistent_1.mean(),
#                                 "cst_1_cnt": rule_consistent_1_cnt,
#                                 "cst_2": rule_consistent_2.mean(),
#                                 "cst_2_cnt": rule_consistent_2_cnt,
#                                 "cst_3": rule_consistent_3.mean(),
#                                 "cst_3_cnt": rule_consistent_3_cnt,
#                                 })
#     rule_summary_df = pd.DataFrame(rule_summary_df)
#     print(rule_summary_df.tail())
#     return rule_summary_df


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
    parser.add_argument("--figdir", type=str, default="Figures_newrule", help="Directory to save the figure.")
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
    # expdir = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results/045-RAVEN10_abstract-uncond-DiT_S_1_20240311-1256"
    
    if not (os.path.exists(join(expdir, "samples_inferred_rule_consistency_new.npz"))
            and args.update):# :)
        print("Inferred rules not found. Inferring from samples.")
        sample_all, epoch_list = batch_load_samples(join(expdir, "samples"), epoch_list, 
                                        encoding=encoding, fmt=args.fmt)
        stats_df, rule_list_all, consistency_all = samples_infer_rule_consistency(
                                    sample_all, epoch_list)
    else:
        print("Inferred rules already exist. Loading from file.")
        npzfile = np.load(join(expdir, "samples_inferred_rule_consistency_new.npz"), allow_pickle=True)
        stats_df_load = pd.read_csv(join(expdir, "consistency_new_stats_df.csv"),)
        consistency_all_load = npzfile["consistency_all"]
        rule_list_all_load = npzfile["rule_list_all"]
        epoch_list_load = npzfile["epoch_list"]
        epoch_rest = [epoch for epoch in epoch_list if epoch not in epoch_list_load]
        print(rule_list_all_load.shape, consistency_all_load.shape, len(epoch_rest), stats_df_load.shape)
        print("Updating inferred rules, starting from epoch: ", epoch_rest[0])
        sample_all, epoch_rest = batch_load_samples(join(expdir, "samples"), 
                            epoch_rest, encoding=encoding, fmt=args.fmt) #encoding="digit", fmt="%07d.pt")
        stats_df_rest, rule_list_all_rest, consistency_all_rest = samples_infer_rule_consistency(
                            sample_all, epoch_rest)

        # rules_all_rest, consistent_mat_rest, epoch_rest = batch_load_samples_infer_rules(
        #     join(expdir, "samples"), epoch_rest, encoding=encoding, fmt=args.fmt)
        if len(epoch_rest) == 0 and len(sample_all) == 0:
            consistency_all, rule_list_all, epoch_list = consistency_all_load, rule_list_all_load, epoch_list_load
            stats_df = stats_df_load
        else:
            rule_list_all_rest = np.array(rule_list_all_rest, dtype=object)
            consistency_all_rest = np.array(consistency_all_rest, dtype=object)
            rule_list_all = np.concatenate([rule_list_all_load, rule_list_all_rest], axis=0)
            consistency_all = np.concatenate([consistency_all_load, consistency_all_rest], axis=0)
            epoch_list = np.concatenate([epoch_list_load, epoch_rest], axis=0)
            stats_df = pd.concat([stats_df_load, stats_df_rest], axis=0)
            print("Inferred rules updated.")
        print(rule_list_all.shape, consistency_all.shape, len(epoch_list), stats_df.shape)
    
    stats_df.to_csv(join(expdir, "consistency_new_stats_df.csv"))
    np.savez(join(expdir, "samples_inferred_rule_consistency_new.npz"), 
                consistency_all=consistency_all, 
                rule_list_all=rule_list_all, epoch_list=epoch_list)
    visualize_consistency_new(stats_df, title_str=title_str, 
                                figname=figname, figdir=figdir,)
    visualize_rule_validity_new(stats_df, title_str=title_str, 
                                figname=figname, figdir=figdir,)
    # else:
    #     rules_all_wide, consistent_mat_wide, epoch_list = batch_load_samples_infer_rules(
    #         join(expdir, "samples"), epoch_list, encoding=encoding, fmt=args.fmt)
    
    # np.savez(join(expdir, "samples_inferred_rule_consistency.npz"), 
    #         consistent_mat=consistent_mat_wide, 
    #         rules_all=rules_all_wide, epoch_list=epoch_list)
    
    # rule_summary_df = rule_summary_table(rules_all_wide, consistent_mat_wide, epoch_list)
    
    # visualize_consistency(epoch_list, consistent_mat_wide, 
    #                     title_str=title_str, figname=figname, figdir=figdir,)

    # visualize_rule_validity(epoch_list, rules_all_wide, 
    #                         title_str=title_str, figname=figname, figdir=figdir,);
