
import sys
sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")
sys.path.append("/n/home12/binxuwang/Github/DiT")
import os
import json
from os.path import join
import pickle as pkl
import torch
import torch as th
from tqdm import tqdm
import numpy as np
import einops
import seaborn as sns
import matplotlib.pyplot as plt
from easydict import EasyDict as edict
plt.rcParams['figure.dpi'] = 72
plt.rcParams['figure.figsize'] = [6.0, 4.0]
plt.rcParams['figure.edgecolor'] = (1, 1, 1, 0)
plt.rcParams['figure.facecolor'] = (1, 1, 1, 0)
# vector graphics type
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from train_edm import create_model, edm_sampler, EDM
from edm_utils import edm_sampler_inpaint, create_edm, get_default_config, create_edm_new
# from rule_utils import get_rule_img, get_obj_list, get_rule_list
# from rule_utils import check_consistent
from rule_new_utils import check_r3_r2_batch
from dataset_utils import train_data2attr_tsr,load_raw_data,load_PGM_abstract
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, TensorDataset
from stats_plot_utils import saveallforms

import circuit_toolkit
print(circuit_toolkit.__file__)
from circuit_toolkit.layer_hook_utils import print_specific_layer, get_module_name_shapes, featureFetcher_module

def infer_rule_from_sample_batch(sample_batch):
    # if not int convert to int
    sample_batch = sample_batch.round().int()
    sample_batch = sample_batch.view(-1, 3, 3, 3, 9) 
    sample_batch = einops.rearrange(sample_batch, 
        "B attr row h (panel w) -> B row panel (h w) attr", 
        panel=3, w=3, h=3, attr=3)
    r3_list, r2_list, rule_col = check_r3_r2_batch(sample_batch)
    return r3_list, r2_list, rule_col

DiT_configs = {
}
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

#%%
import time
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from diffusion import create_diffusion
from models import DiT

device = "cuda"

# Define the linear classifier model
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


# Define the training loop
def train_model(model, train_loader, num_epochs, learning_rate, print_every=50,
                eval_every=500, eval_func=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_record = []
    test_record = []
    for epoch in range(num_epochs):
        acc_total = 0
        cnt_total = 0
        for inputs, labels in train_loader:
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc_cnt = (outputs.argmax(dim=1) == labels).sum().item()
            acc_total += acc_cnt
            cnt_total += len(labels)
        accuracy = acc_total / cnt_total
        if (epoch + 1) % print_every == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        train_record.append((epoch, loss.item(), accuracy))
        if ((epoch + 1) % eval_every == 0 or epoch == num_epochs - 1) and eval_func is not None:
            test_acc, _ = eval_func(model)
            test_record.append((epoch, test_acc))
    train_record = pd.DataFrame(train_record, columns=["epoch", "loss", "accuracy"])
    test_record = pd.DataFrame(test_record, columns=["epoch", "accuracy"])
    return train_record, test_record


def test_model(model, test_loader):
    acc_total = 0
    cnt_total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        with th.no_grad():
            outputs = model(inputs)
        pred_cls = outputs.argmax(dim=1)
        acc_cnt = (pred_cls == labels).sum().item()
        acc_total += acc_cnt
        cnt_total += len(labels)
    accuracy = acc_total / cnt_total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy, pred_cls


def fit_SGD_linear_classifier(train_X, train_y, test_X=None, test_y=None, 
                              num_classes=40, 
                              batch_size=1024, num_epochs=100, 
                              learning_rate = 0.001, print_every=100, eval_every=500,):
    # Define the linear classifier model
    input_size = train_X.shape[1]
    model = LinearClassifier(input_size, num_classes).to("cuda")
    if batch_size is None:
        feat_loader = [(train_X.to("cuda"), train_y.to("cuda"))]
    else:
        feat_dataset = TensorDataset(train_X.to("cuda"), train_y.to("cuda")) # .to("cuda")
        feat_loader = DataLoader(feat_dataset, batch_size=batch_size, shuffle=True,
                             drop_last=True) # pin_memory=True, num_workers=
    
    if test_X is not None and test_y is not None:
        if batch_size is None:
            test_feat_loader = [(test_X.to("cuda"), test_y.to("cuda"))]
        else:
            test_dataset = TensorDataset(test_X.to("cuda"), test_y.to("cuda")) # .to("cuda")
            test_feat_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Define the training loop
    train_record, test_record = train_model(model, feat_loader, num_epochs, learning_rate, print_every=print_every, eval_every=eval_every,
                eval_func=lambda model: test_model(model, test_feat_loader) if test_feat_loader is not None else None)
    # Define the testing loop
    test_acc, pred_cls = test_model(model, test_feat_loader)
    results = edict()
    results.feature_dim = input_size
    results.train_record = train_record
    results.test_record = test_record
    results.test_acc = test_acc
    results.pred_cls = pred_cls
    return model, results

import argparse
parser = argparse.ArgumentParser(description="Load and evaluate a DiT model.")
parser.add_argument("--expname", type=str, default="WideBlnrX3_new_RAVEN10_abstract_20240315-1327", 
                    help="The name of the experiment.")
parser.add_argument("--epoch", type=int, default=1000000, 
                    help="The epoch of the model to load.")
# parser.add_argument("--use_ema", action="store_true", 
#                     help="Whether to use the EMA model.")
parser.add_argument("--t_scalars", type=float, default=[0.1], 
                    nargs="+", help="The time value to evaluate the score vectors.")
parser.add_argument("--noPCA", action="store_true", 
                    help="Whether to use PCA for the feature vectors.")
parser.add_argument("--PC_dim", type=int, default=512, 
                    help="The dimension of the PCA projection.")
parser.add_argument("--figdir", type=str, default="/n/home12/binxuwang/Github/DiffusionReasoning/Figures_repr_classify",
                    help="The directory to save the figures.")

# model_DiT
# exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/DiT/results"

args = parser.parse_args()
expname = args.expname
epoch = args.epoch
# use_ema = args.use_ema
t_scalars = args.t_scalars
noPCA = args.noPCA
PC_dim = args.PC_dim
figdir = args.figdir

# if DiTargs["cond"]:
#     num_classes = DiTargs.num_classes
#     class_dropout_prob = DiTargs["class_dropout_prob"]
# else:
#     num_classes = 0
#     class_dropout_prob = 1.0

    
DATASET="RAVEN10_abstract"
# heldout_rules = heldout_id_dict[DiTargs["train_attr_fn"]]


if DATASET == "RAVEN10_abstract":
    in_channels = 3
    dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to("cuda")
    dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to("cuda")
elif DATASET == "RAVEN10_abstract_onehot":
    in_channels = 27
    dataset_Xmean = th.tensor([0.5, ]).view(1, 1, 1, 1)
    dataset_Xstd = th.tensor([0.5, ]).view(1, 1, 1, 1)
    raise NotImplementedError("RAVEN10_abstract_onehot not implemented.")

# Prepare the dataset, training and testing
# train_data_fn = "train_inputs_new.pt"
# train_attrs = torch.load(f'/n/home12/binxuwang/Github/DiffusionReasoning/{train_data_fn}')
# train_attrs = train_attrs.to(int)
train_attrs = np.load("/n/home12/binxuwang/Github/DiffusionReasoning/attr_all.npy")
train_attrs = th.from_numpy(train_attrs).to(int)

if DATASET == "RAVEN10_abstract":
    train_row_img = einops.rearrange(train_attrs, 'c s pnl (H W) att -> c s att H (pnl W)', H=3, W=3, att=3, pnl=3)
    train_sample_img = einops.rearrange(train_row_img, 'c (S R) att H W -> c S att (R H) W', R=3,att=3, H=3, W=9)
elif DATASET == "RAVEN10_abstract_onehot":
    raise NotImplementedError("RAVEN10_abstract_onehot not implemented.")
    # use the one-hot encoding
labels_tsr = th.arange(len(train_sample_img)).to(int).view(-1,1).repeat(1, train_sample_img.shape[1])

X_train = train_sample_img[:, :3000]
y_train = labels_tsr[:, :3000]
X_test = train_sample_img[:, 3000:]
y_test = labels_tsr[:, 3000:]
X_train = X_train.reshape(-1, 3, 9, 9)
y_train = y_train.reshape(-1)
X_test = X_test.reshape(-1, 3, 9, 9)
y_test = y_test.reshape(-1)
print(X_train.shape, y_train.shape)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear

# epoch = 700000
config_mapping = {
    "WideBlnrX3_new_RAVEN10_abstract_20240315-1327": 
        # --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear
        dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "BigBlnrX3_new_RAVEN10_abstract_20240412-0143":
        # --layers_per_block 3 --model_channels 192 --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear
        dict(layers_per_block=3, model_channels=192, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
    "WideBlnrX3_new_noattn_RAVEN10_abstract_20240412-1254":
        # --layers_per_block 2 --model_channels 128 --channel_mult 1 2 4 --attn_resolutions 0   --train_batch_size 256 --spatial_matching bilinear
        dict(layers_per_block=2, model_channels=128, channel_mult=[1, 2, 4], attn_resolutions=[0], spatial_matching="bilinear"),
    "BaseBlnrX3_new_RAVEN10_abstract_20240313-1736": 
        # --layers_per_block 1 --model_channels 64  --channel_mult 1 2 4 --attn_resolutions 9 3 --train_batch_size 256 --spatial_matching bilinear
        dict(layers_per_block=1, model_channels=64, channel_mult=[1, 2, 4], attn_resolutions=[9, 3], spatial_matching="bilinear"),
}
# expname = "WideBlnrX3_new_RAVEN10_abstract_20240315-1327"
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/mini_edm/exps"
expdir = join(exproot, expname)
ckptdir = join(expdir, "checkpoints")
repr_expdir = join(expdir, "repr_classifier")
os.makedirs(repr_expdir, exist_ok=True)
device = "cuda"
# config_ft = get_default_config(DATASET, layers_per_block=2, 
#                                model_channels=128, 
#                                channel_mult=[1, 2, 4], 
#                                attn_resolutions=[9, 3], 
#                                spatial_matching="bilinear")
config_ft = get_default_config(DATASET, **config_mapping[expname])
# PC_dim = 1024
# # noPCA = False
# noPCA = True
use_ema = True
if epoch == -1:
    # use random initialization if epoch is -1
    edm, model_EDM = create_edm_new(None, config_ft, device) 
    ckpt_str = "ckptRNDINIT"
    print("Random initialization")
else:
    ckpt_path = join(ckptdir, f"ema_{epoch}.pth")
    edm, model_EDM = create_edm_new(ckpt_path, config_ft, device) 
    ckpt_str = f"ckpt{epoch:07d}EMA"
    print(f"Loaded {ckpt_str}: from {ckpt_path}, use_ema: {use_ema}")

model_EDM.to("cuda").eval();

if noPCA: 
    PC_dim = "FULL"
# if noPCA optimizing in the full space requires a smaller learning rate
learning_rate = 0.0005 if noPCA else 0.005
fetcher = featureFetcher_module()
record_module_list = ["input",
               'enc.9x9_conv',
                'enc.3x3_down',
                'enc.1x1_down',
                'dec.1x1_in0',
                'dec.1x1_in1',
                'dec.3x3_up',
                'dec.9x9_up',
                # 'dec.9x9_block2',
                "dec.9x9_aux_norm",
                "dec.9x9_aux_conv", ]
for blockname in list(model_EDM.enc):
    if f"enc.{blockname}" in record_module_list:
        fetcher.record_module(model_EDM.enc[blockname], target_name=f"enc.{blockname}")
for blockname in list(model_EDM.dec):
    if f"dec.{blockname}" in record_module_list:
        fetcher.record_module(model_EDM.dec[blockname], target_name=f"dec.{blockname}")

for t_scalar in args.t_scalars: # 0.3, 0.5, 0.7, 0.9, 1.0, 0.05, 0.02, 0.1
    t_str = str(t_scalar).replace('.', '_')
    t_beg = time.time()
    # t_scalar = 0.1
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False, drop_last=False)
    feature_col = defaultdict(list)
    for X_batch, y_batch in tqdm(train_loader):
        t_vec = th.ones(X_batch.shape[0], dtype=torch.float, device="cuda") * t_scalar
        X_batch_norm = (X_batch.cuda().float() - dataset_Xmean) / dataset_Xstd
        with th.no_grad():
            model_EDM(X_batch_norm, t_vec, None)
        feature_col["input"].append(X_batch.float())
        for key, activations in fetcher.activations.items():
            feature_col[key].append(activations)
    for key in feature_col.keys():
        feature_col[key] = th.cat(feature_col[key], dim=0)
        print(key, feature_col[key].shape)

    feature_col_test = defaultdict(list)
    for X_batch, y_batch in tqdm(test_loader):
        t_vec = th.ones(X_batch.shape[0], dtype=torch.float, device="cuda") * t_scalar
        X_batch_norm = (X_batch.cuda().float() - dataset_Xmean) / dataset_Xstd
        with th.no_grad():
            model_EDM.forward(X_batch_norm, t_vec, None)
        feature_col_test["input"].append(X_batch.float())
        for key, activations in fetcher.activations.items():
            feature_col_test[key].append(activations)
            
    for key in feature_col_test.keys():
        feature_col_test[key] = th.cat(feature_col_test[key], dim=0)
        print(key, feature_col_test[key].shape)
    th.cuda.empty_cache() 
    t_feat = time.time()
    print(f"Feature extraction time: {t_feat - t_beg:.2f}")

    model_PCA_col = {}
    PC_proj_col = {}
    results_col = {}
    for layerkey in record_module_list: # 
        t0 = time.time()
        featmat = feature_col[layerkey].reshape(len(train_dataset),-1)
        featmat_test = feature_col_test[layerkey].reshape(len(test_dataset),-1)
        featmean = featmat.mean(dim=0)
        featdim = featmat.shape[1]
        t1 = time.time()
        print(feature_col[layerkey].shape, featmat.shape, featdim)
        if noPCA or featdim <= PC_dim:
            feat_PCA = (featmat - featmean[None,]).cuda()
            feat_PCA_std = feat_PCA.std(dim=0)
            feat_PCA = feat_PCA / feat_PCA_std[None,]
            feat_PCA_test = (featmat_test - featmean[None,]).cuda() 
            feat_PCA_test = feat_PCA_test / feat_PCA_std[None,]
        else:
            U, S, V = torch.pca_lowrank((featmat - featmean[None,]).cuda(), q=PC_dim, center=False, niter=3)
            print(U.shape, S.shape, V.shape)
            del U, S
            torch.cuda.empty_cache()
            feat_PCA = (featmat - featmean[None,]).cuda() @ V
            feat_PCA_std = feat_PCA.std(dim=0)
            feat_PCA = feat_PCA / feat_PCA_std[None,]
            feat_PCA_test = (featmat_test - featmean[None,]).cuda() @ V
            feat_PCA_test = feat_PCA_test / feat_PCA_std[None,]
            torch.cuda.empty_cache()
            V = V.cpu()
        t2 = time.time()
        model, results_dict = fit_SGD_linear_classifier(feat_PCA, y_train, feat_PCA_test, y_test,
                                        num_classes=40, batch_size=None, num_epochs=5000, print_every=500, 
                                        eval_every=1000, learning_rate=learning_rate)
        t3 = time.time()
        print(f"Layer {layerkey} PCA time: {t1-t0:.2f}, PCA transform time: {t2-t1:.2f}, training time: {t3-t2:.2f}")
        model_PCA_col[layerkey] = model
        if noPCA or featdim <= PC_dim:
            PC_proj_col[layerkey] = (None, featmean.cpu(), feat_PCA_std.cpu())
        else:
            PC_proj_col[layerkey] = (V.cpu(), featmean.cpu(), feat_PCA_std.cpu())
        results_col[layerkey] = results_dict
        del feat_PCA, feat_PCA_test
        th.cuda.empty_cache()
        
    pkl.dump(results_col, open(join(repr_expdir,f"results_col_t{t_str}_PC{PC_dim}_{ckpt_str}.pkl"), "wb"))
    pkl.dump(PC_proj_col, open(join(repr_expdir,f"PC_proj_col_t{t_str}_PC{PC_dim}_{ckpt_str}.pkl"), "wb"))
    pkl.dump(model_PCA_col, open(join(repr_expdir,f"model_PCA_col_t{t_str}_PC{PC_dim}_{ckpt_str}.pkl"), "wb"))

    test_acc_synopsis = {key: results_col[key].test_record.accuracy.max() for key in results_col.keys()}
    train_acc_synopsis = {key: results_col[key].train_record.accuracy.max() for key in results_col.keys()}
    syn_df = pd.DataFrame([test_acc_synopsis, train_acc_synopsis]).T
    syn_df.columns = ["Test Accuracy", "Train Accuracy"]

    plt.figure(figsize=(6, 4))
    sns.lineplot(data=syn_df, markers=True)
    plt.title(f"t_scalar: {t_scalar} Accuracy\n{expname}\n{ckpt_str} PC{PC_dim}")
    saveallforms([repr_expdir, figdir],f"{expname}_t{t_str}_PC{PC_dim}_{ckpt_str}_accuracy")
    plt.show()
    
    del feature_col
    del feature_col_test
    del PC_proj_col

del fetcher






