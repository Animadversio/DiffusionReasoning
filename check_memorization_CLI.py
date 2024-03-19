#%%
import torch
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import einops
from os.path import join
from tqdm import trange, tqdm
from os.path import join
from edm_utils import parse_train_logfile
from dataset_utils import onehot2attr_tsr
from stats_plot_utils import estimate_CI, shaded_error, saveallforms
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


import argparse
parser = argparse.ArgumentParser(description="Process and visualize EDM rules.")

# Add arguments
parser.add_argument("--ep_start", type=int, default=0, help="Start of the epoch range.")
parser.add_argument("--ep_stop", type=int, default=1000000, help="End of the epoch range.")
parser.add_argument("--ep_step", type=int, default=50000, help="Step size for the epoch range.")
parser.add_argument("--expname", type=str, required=True, help="Name of the experiment.")
parser.add_argument("--exproot", type=str, default="/n/holylabs/LABS/kempner_fellows/Users/binxuwang/DL_Projects/mini_edm/exps", help="Root directory for the experiment.")
parser.add_argument("--encoding", type=str, choices=["onehot", "digit"], default="digit", help="Encoding method for processing samples.")
# parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu", help="Compute device to use.")
parser.add_argument("--figname", type=str, default="", help="Name of the figure to save.")
parser.add_argument("--figdir", type=str, default="Figures_newrule", help="Directory to save the figure.")
parser.add_argument("--title_str", type=str, default="", help="Title of the figure.")
parser.add_argument("--fmt", type=str, default='%07d.pt', help="Format of the sample files.")
# parser.add_argument("--update", action="store_true", help="Update the existing inferred rules.")
                    
# Parse arguments
args = parser.parse_args()
exproot = args.exproot
expname = args.expname
epoch_list = list(range(args.ep_start, args.ep_stop, args.ep_step))
encoding = args.encoding
figdir = args.figdir
figname = expname if args.figname == "" else args.figname
title_str = expname.replace("_"," ") if args.title_str == "" else args.title_str


"""Measure L1 distance to nearest neighbor in the training set"""
DiTroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"
EDMroot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
train_attrs = torch.load('/n/home12/binxuwang/Github/DiffusionReasoning/train_inputs_new.pt')
# shape [35, 12000, 3, 9, 3]
# figdir = "/n/home12/binxuwang/Github/DiffusionReasoning/Figures_newrule"
# sample_dir = DiTroot + "/042-RAVEN10_abstract-uncond-DiT_B_1_20240309-1718/samples"
# expname = "045-RAVEN10_abstract-uncond-DiT_S_1_20240311-1256"
# epoch_list = range(1000,1000001,10000)
visualize = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
expdir = join(exproot, expname)
sample_dir = join(expdir, "samples")
sample_all, epoch_list = batch_load_samples(sample_dir, epoch_list, encoding=args.encoding, fmt=args.fmt)
train_attrs = train_attrs.to(device)
#%%
# Check row level memorization
batchsize = 32
minL1dist_col = {}
for epochi in range(len(sample_all)):
    attr_panels = einops.rearrange(sample_all[epochi], 
            'b attr (row h) (p w) -> (b row) p (h w) attr', row=3, p=3)
    attr_panels.shape # (batch*3, 3, 9, 3)

    minL1dist_mat = []
    minimum_vec = []
    for i in trange(0, len(attr_panels), batchsize):
        L1dist = (train_attrs[None,] - attr_panels[i:i+batchsize].to(device)[:,None,None]).abs().sum(dim=(-1,-2,-3))
        minL1dist = L1dist.min(dim=-1).values
        minimum = minL1dist.min(dim=-1).values
        minL1dist_mat.append(minL1dist)
        minimum_vec.append(minimum)
    torch.cuda.empty_cache()

    minL1dist_mat = torch.concatenate(minL1dist_mat, dim=0).cpu() # (3 * sampleN, 35)
    minimum_vec = torch.concatenate(minimum_vec, dim=0).cpu() # (3 * sampleN,)
    # print basic statistics of minimum_vec using pandas
    minL1dist_col[epoch_list[epochi]] = minL1dist_mat
    if visualize:
        print(epoch_list[epochi])
        print(pd.Series(minimum_vec.numpy()).describe())
        plt.hist(minimum_vec, bins=100)
        plt.xlabel('Minimum Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of minimum_vec (epoch {epoch_list[epochi]})')
        plt.show()
    
torch.save(minL1dist_col, join(expdir, "row_minL1dist_dict.pt"))
#%%
# Check panel level memorization
train_attrs_one_panel = einops.rearrange(train_attrs,
        'class B p pos attr -> class (B p) pos attr', pos=9, attr=3, p=3)
print(train_attrs_one_panel.shape)
# train_attrs_one_panel.shape # (class, batch * 3, 9, 3)
batchsize_row = 16 # cannot be larger, otherwise OOM
minL1dist_col_panel = {}
for epochi in range(len(sample_all)):
    attr_one_panel = einops.rearrange(sample_all[epochi], 
            'b attr (row h) (p w) -> (b row p) (h w) attr', row=3, p=3)
    # print(attr_one_panel.shape) # (batch * 3 * 3, 9, 3)
    minL1dist_mat_panel = []
    minimum_vec_panel = []
    for i in trange(0,len(attr_one_panel),batchsize_row):
        L1dist = (train_attrs_one_panel[None,] - 
                  attr_one_panel[i:i+batchsize_row].to(device)[:,None,None]).abs().sum(dim=(-1,-2))
        minL1dist_panel = L1dist.min(dim=-1).values # minimum distance to each class (min acros training samples)
        minimum_panel = minL1dist_panel.min(dim=-1).values # minimum distance to any class (min across rules)
        minL1dist_mat_panel.append(minL1dist_panel)
        minimum_vec_panel.append(minimum_panel)
    torch.cuda.empty_cache()

    minL1dist_mat_panel = torch.concatenate(minL1dist_mat_panel, dim=0).cpu() # (3 * sampleN, 35)
    minimum_vec_panel = torch.concatenate(minimum_vec_panel, dim=0).cpu() # (3 * sampleN,)
    minL1dist_col_panel[epoch_list[epochi]] = minL1dist_mat_panel
    # print basic statistics of minimum_vec using pandas
    if visualize:
        print(epoch_list[epochi])
        print(pd.Series(minimum_vec_panel.numpy()).describe())
        plt.hist(minimum_vec_panel, bins=100)
        plt.xlabel('Minimum Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of minimum_vec (epoch {epoch_list[epochi]})')
        plt.show()
    
torch.save(minL1dist_col_panel, join(expdir, "panel_minL1dist_dict.pt"))
# %%
# formatting array to dataframe of stats
mem_tab = []
for epoch in minL1dist_col:
    minL1, minclass = minL1dist_col[epoch].min(dim=1)
    memfrac = (minL1 == 0).float().mean()
    if memfrac > 0:
        # print(minclass[minL1 == 0].unique())
        row_memclass, row_memcnts = torch.unique(
            minclass[minL1 == 0], return_counts=True)
        row_memcnts = row_memcnts.tolist()
        row_memclass = row_memclass.tolist()
    else:
        row_memclass = []
        row_memcnts = []
    minL1_panel, minclass_panel = minL1dist_col_panel[epoch].min(dim=1)
    memfrac_panel = (minL1_panel == 0).float().mean()
    if memfrac_panel > 0:
        # print(minclass_panel[minL1_panel == 0].unique())
        panel_memclass, panel_memcnts = torch.unique(
            minclass_panel[minL1_panel == 0], return_counts=True)
        panel_memclass = panel_memclass.tolist()
        panel_memcnts = panel_memcnts.tolist()
    else:
        panel_memclass = []
        panel_memcnts = []
    mem_tab.append({"epoch":epoch, 
                    "row_memfrac":memfrac.item(),
                    "row_memclass": row_memclass,
                    "row_memcnts": row_memcnts,
                    "panel_memfrac":memfrac_panel.item(),
                    "panel_memclass": panel_memclass,
                    "panel_memcnts": panel_memcnts,
                    })
mem_tab = pd.DataFrame(mem_tab)
mem_tab.to_csv(join(expdir, "memorization_stats.csv"))
print(mem_tab.tail())
# %%
mem_tab.plot(x="epoch", y=["row_memfrac", "panel_memfrac"])
plt.ylabel("Fraction of samples memorized")
plt.xlabel("Epoch")
plt.title(f"Memorization of training samples\n{expname}")
saveallforms(figdir, f"{figname}_memorization", plt.gcf())
# %%
# minL1dist_mat_panel.reshape(1024,3,3,-1).min(dim=-1).values.float().mean(dim=0)
# (minL1dist_mat_panel.reshape(1024,3,3,-1).min(dim=-1).values==0).float().mean(dim=0)