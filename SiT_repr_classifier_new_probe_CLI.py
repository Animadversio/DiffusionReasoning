
import os
import json
from os.path import join
import pickle as pkl
import time
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.auto import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
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


import sys
# sys.path.append("/n/home12/binxuwang/Github/mini_edm")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")
sys.path.append("/n/home12/binxuwang/Github/SiT")
# from train_edm import create_model, edm_sampler, EDM
# from edm_utils import edm_sampler_inpaint, create_edm, get_default_config
# from diffusion import create_diffusion
# from models import DiT
from models import SiT_models, SiT
from transport import create_transport, Sampler

# from rule_utils import get_rule_img, get_obj_list, get_rule_list
# from rule_utils import check_consistent
from rule_new_utils import check_r3_r2_batch, infer_rule_from_sample_batch, compute_rule_statistics
from dataset_utils import train_data2attr_tsr,load_raw_data,load_PGM_abstract
from collections import defaultdict
from stats_plot_utils import saveallforms

import circuit_toolkit
from circuit_toolkit.layer_hook_utils import print_specific_layer, get_module_name_shapes, featureFetcher_module

SiT_configs = {
    "SiT_XL_1": {"depth": 28, "hidden_size": 1152, "patch_size": 1, "num_heads": 16},
    "SiT_XL_3": {"depth": 28, "hidden_size": 1152, "patch_size": 3, "num_heads": 16},
    "SiT_L_1": {"depth": 24, "hidden_size": 1024, "patch_size": 1, "num_heads": 16},
    "SiT_L_3": {"depth": 24, "hidden_size": 1024, "patch_size": 3, "num_heads": 16},
    "SiT_B_1": {"depth": 12, "hidden_size": 768, "patch_size": 1, "num_heads": 12},
    "SiT_B_3": {"depth": 12, "hidden_size": 768, "patch_size": 3, "num_heads": 12},
    "SiT_S_1": {"depth": 12, "hidden_size": 384, "patch_size": 1, "num_heads": 6},
    "SiT_S_3": {"depth": 12, "hidden_size": 384, "patch_size": 3, "num_heads": 6},
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

device = "cuda" if torch.cuda.is_available() else "cpu"

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


def extract_features_SiT(
    model,
    fetcher,
    data_loader,
    dataset_Xmean,
    dataset_Xstd,
    t_scalar,
    device='cuda',
    progress_bar=True
):
    """
    Extracts features from specified layers of the model for the given dataset.

    Args:
        model (torch.nn.Module): The neural network model.
        fetcher (FeatureFetcher): An instance of the featureFetcher_module.
        data_loader (DataLoader): DataLoader for the dataset.
        dataset_mean (torch.Tensor): Mean for input normalization.
        dataset_std (torch.Tensor): Standard deviation for input normalization.
        t_scalar (float): Scalar value to create the time vector.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.
        progress_bar (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        dict: A dictionary with layer keys and concatenated activation tensors.
    """
    feature_col = defaultdict(list)
    loader = tqdm(data_loader) if progress_bar else data_loader
    for X_batch, _ in loader:
        # Prepare model inputs
        model_kwargs = {'y': th.zeros(X_batch.size(0), dtype=th.int, device=device)}
        t_vec = th.ones(X_batch.size(0), dtype=th.float, device=device) * t_scalar
        # Normalize the batch
        X_batch_norm = (X_batch.cuda().float() - dataset_Xmean) / dataset_Xstd
        # Forward pass with no gradient computation
        with th.no_grad():
            model.forward(X_batch_norm, t_vec, **model_kwargs)
        # Collect activations
        for key, activations in fetcher.activations.items():
            feature_col[key].append(activations.cpu())

    # Concatenate all activations for each layer
    for key in feature_col:
        feature_col[key] = th.cat(feature_col[key], dim=0)
        print(f"{key}: {feature_col[key].shape}")
    return feature_col


def train_pca_sgd_classifiers(
    feature_col,
    feature_col_test,
    y_train,
    y_test,
    PC_dim=1024,
    noPCA=False,
    num_classes=40,
    batch_size=None,
    num_epochs=5000,
    print_every=250,
    eval_every=1000,
    learning_rate=0.005,
    device='cuda'  # Specify 'cuda' or 'cpu'
):
    """
    Trains SGD linear classifiers on PCA-transformed features for each layer.

    Args:
        feature_col (dict): Training features for each layer.
        feature_col_test (dict): Test features for each layer.
        y_train (torch.Tensor or np.ndarray): Training labels.
        y_test (torch.Tensor or np.ndarray): Test labels.
        PC_dim (int, optional): Number of principal components. Defaults to 1024.
        noPCA (bool, optional): Whether to skip PCA. Defaults to False.
        num_classes (int, optional): Number of target classes. Defaults to 40.
        batch_size (int, optional): Batch size for SGD. Defaults to None.
        num_epochs (int, optional): Number of training epochs. Defaults to 5000.
        print_every (int, optional): Frequency of printing progress. Defaults to 250.
        eval_every (int, optional): Frequency of evaluating on test set. Defaults to 1000.
        learning_rate (float, optional): Learning rate for SGD. Defaults to 0.005.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda'.

    Returns:
        model_PCA_col (dict): Trained models for each layer.
        PC_proj_col (dict): PCA projection parameters for each layer.
        results_col (dict): Training and evaluation results for each layer.
    """
    model_PCA_col = {}
    PC_proj_col = {}
    results_col = {}

    for layerkey in feature_col.keys():
        print(f"Processing layer: {layerkey}")
        t0 = time.time()

        # Reshape feature matrices
        featmat = feature_col[layerkey].view(len(feature_col[layerkey]), -1)
        featmat_test = feature_col_test[layerkey].view(len(feature_col_test[layerkey]), -1)

        # Compute mean of training features
        featmean = featmat.mean(dim=0)
        t1 = time.time()

        if noPCA:
            # Center and normalize features without PCA
            feat_PCA = (featmat - featmean[None, :]).to(device)
            feat_PCA_std = feat_PCA.std(dim=0)
            feat_PCA = feat_PCA / feat_PCA_std[None, :]
            feat_PCA_test = (featmat_test - featmean[None, :]).to(device)
            feat_PCA_test = feat_PCA_test / feat_PCA_std[None, :]
        else:
            # Perform PCA
            centered_feat = (featmat - featmean[None, :]).to(device)
            U, S, V = torch.pca_lowrank(centered_feat, q=PC_dim, center=False, niter=3)
            print(f"PCA components for layer {layerkey}: U shape {U.shape}, S shape {S.shape}, V shape {V.shape}")
            # Clean up unnecessary variables
            del U, S
            torch.cuda.empty_cache()
            # Project training and test features
            feat_PCA = centered_feat @ V
            feat_PCA_std = feat_PCA.std(dim=0)
            feat_PCA = feat_PCA / feat_PCA_std[None, :]
            feat_PCA_test = (featmat_test - featmean[None, :]).to(device) @ V
            feat_PCA_test = feat_PCA_test / feat_PCA_std[None, :]
            torch.cuda.empty_cache()
            V = V.cpu()

        t2 = time.time()

        # Train the SGD linear classifier
        model, results_dict = fit_SGD_linear_classifier(
            feat_PCA, y_train, feat_PCA_test, y_test,
            num_classes=num_classes,
            batch_size=batch_size,
            num_epochs=num_epochs,
            print_every=print_every,
            eval_every=eval_every,
            learning_rate=learning_rate
        )

        t3 = time.time()

        print(f"Layer {layerkey} - PCA time: {t1 - t0:.2f}s, "
              f"PCA transform time: {t2 - t1:.2f}s, "
              f"Training time: {t3 - t2:.2f}s")

        # Store the trained model and PCA projection parameters
        model_PCA_col[layerkey] = model
        PC_proj_col[layerkey] = {
            'V': V.cpu() if not noPCA else None,  # PCA components
            'mean': featmean.cpu(),
            'std': feat_PCA_std.cpu()
        }
        results_col[layerkey] = results_dict
        del feat_PCA, feat_PCA_test

    return model_PCA_col, PC_proj_col, results_col


def train_dimred_sgd_classifiers(
    feature_col,
    feature_col_test,
    y_train,
    y_test,
    dimred_str="pca",
    num_classes=40,
    batch_size=None,
    num_epochs=5000,
    print_every=250,
    eval_every=1000,
    learning_rate=0.005,
    device='cuda'  # Specify 'cuda' or 'cpu'
):
    if dimred_str == "avgtoken":
        feature_red_col = {k: v.mean(dim=1) for k, v in feature_col.items()}
        feature_red_col_test = {k: v.mean(dim=1) for k, v in feature_col_test.items()}
        noPCA = True
        PC_dim = None
    elif dimred_str == "lasttoken":
        feature_red_col = {k: v[:, -1] for k, v in feature_col.items()}
        feature_red_col_test = {k: v[:, -1] for k, v in feature_col_test.items()}
        noPCA = True
        PC_dim = None
    elif dimred_str == "none":
        noPCA = True
        PC_dim = None
        feature_red_col = feature_col
        feature_red_col_test = feature_col_test
    elif dimred_str.startswith("pca"):
        noPCA = False
        PC_dim = int(dimred_str[3:])
        feature_red_col = feature_col
        feature_red_col_test = feature_col_test
    else:
        raise ValueError(f"Invalid dimensionality reduction method: {dimred_str}")
    
    model_PCA_col, PC_proj_col, results_col = train_pca_sgd_classifiers(
        feature_red_col, feature_red_col_test, y_train, y_test,
        PC_dim=PC_dim, noPCA=noPCA, num_classes=num_classes,
        batch_size=batch_size, num_epochs=num_epochs, print_every=print_every,
        eval_every=eval_every, learning_rate=learning_rate, device=device
    )
    return model_PCA_col, PC_proj_col, results_col


import re
def validate_dim_red(value):
    pattern = re.compile(r'^(pca\d+|none|avgtoken|lasttoken)$')
    if not pattern.match(value):
        raise argparse.ArgumentTypeError(
            f"Invalid choice: '{value}'. Must match one of: pca, none, avgtoken, lasttoken."
        )
    return value

import argparse
parser = argparse.ArgumentParser(description="Load and evaluate a SiT model.")
# expname = r"045-RAVEN10_abstract-uncond-SiT_S_1_20240311-1256"
# expname = r"062-RAVEN10_abstract-uncond-SiT_S_1_20240330-0111"
parser.add_argument("--expname", type=str, default="045-RAVEN10_abstract-uncond-SiT_S_1_20240311-1256", 
                    help="The name of the experiment.")
parser.add_argument("--epoch", type=int, default=1000000, 
                    help="The epoch of the model to load.")
parser.add_argument("--use_ema", action="store_true", 
                    help="Whether to use the EMA model.")
parser.add_argument("--t_scalars", type=float, default=[0, 1, 10, 25, 50, 100, 250, 500, 1000], 
                    nargs="+", help="The time value to evaluate the score vectors.")
parser.add_argument("--layers", type=int, default=[0, 2, 5, 8, 11],
                    nargs="+", help="The layers to evaluate.")
parser.add_argument("--dim_red_method", type=validate_dim_red, 
                    default=["pca512"], #choices=["pca", "none", "avgtoken", "lasttoken"],
                    nargs="+", help="The dimensionality reduction method.")
# parser.add_argument("--noPCA", action="store_true", 
#                     help="Whether to use PCA for the feature vectors.")
# parser.add_argument("--PC_dim", type=int, default=1024, 
#                     help="The dimension of the PCA projection.")
parser.add_argument("--figdir", type=str, default="Figures_newrule",
                    help="The directory to save the figures.")

# model_SiT
exproot = r"/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects/SiT/results"
args = parser.parse_args()
expname = args.expname
epoch = args.epoch
use_ema = args.use_ema
layers = args.layers
t_scalars = args.t_scalars
# PC_dim = args.PC_dim
# dim_red = args.dim_red_method
figdir = args.figdir

expdir = join(exproot, expname)
ckptdir = join(expdir, "checkpoints")
repr_expdir = join(expdir, "repr_classifier")
os.makedirs(repr_expdir, exist_ok=True)
SiTargs = json.load(open(join(expdir, "args.json")))

if SiTargs["cond"]:
    num_classes = SiTargs.num_classes
    class_dropout_prob = SiTargs["class_dropout_prob"]
else:
    num_classes = 0
    class_dropout_prob = 1.0

if SiTargs["dataset"] == "RAVEN10_abstract":
    in_channels = 3
    dataset_Xmean = th.tensor([1.5, 2.5, 2.5]).view(1, 3, 1, 1).to("cuda")
    dataset_Xstd = th.tensor([2.5, 3.5, 3.5]).view(1, 3, 1, 1).to("cuda")
elif SiTargs["dataset"] == "RAVEN10_abstract_onehot":
    in_channels = 27
    dataset_Xmean = th.tensor([0.5, ]).view(1, 1, 1, 1)
    dataset_Xstd = th.tensor([0.5, ]).view(1, 1, 1, 1)
    raise NotImplementedError("RAVEN10_abstract_onehot not implemented.")
    
model_cfg = SiT_configs[SiTargs["model"]]
model_SiT = SiT(input_size=9,
            in_channels=in_channels, **model_cfg,
            class_dropout_prob=class_dropout_prob,
            num_classes=num_classes,
            learn_sigma=True,)

if "heldout_ids" in SiTargs:
    heldout_rules = SiTargs["heldout_ids"]
else:
    heldout_rules = heldout_id_dict[SiTargs["train_attr_fn"]]

# train_data_fn = "train_inputs_new.pt"
# train_attrs = torch.load(f'/n/home12/binxuwang/Github/DiffusionReasoning/{train_data_fn}')
# train_attrs = train_attrs.to(int)
# prepare the dataset, training and testing
train_attrs = np.load("/n/home12/binxuwang/Github/DiffusionReasoning/attr_all.npy")
train_attrs = th.from_numpy(train_attrs).to(int)

if SiTargs["dataset"] == "RAVEN10_abstract":
    train_row_img = einops.rearrange(train_attrs, 'c s pnl (H W) att -> c s att H (pnl W)', H=3, W=3, att=3, pnl=3)
    train_sample_img = einops.rearrange(train_row_img, 'c (S R) att H W -> c S att (R H) W', R=3,att=3, H=3, W=9)
elif SiTargs["dataset"] == "RAVEN10_abstract_onehot":
    raise NotImplementedError("RAVEN10_abstract_onehot not implemented.")
    # use the one-hot encoding
labels_tsr = th.arange(40).to(int).view(-1,1).repeat(1, train_sample_img.shape[1]) # len(train_sample_img)

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

# abstract RAVEN dataset
if epoch == -1:
    # use random initialization if epoch is -1
    ckpt_str = "ckptRNDINIT"
    print("Random initialization")
else:
    ckpt_path = join(ckptdir, f"{epoch:07d}.pt")
    state_dict = th.load(ckpt_path, )
    if use_ema:
        model_SiT.load_state_dict(state_dict["ema"])
    else:
        model_SiT.load_state_dict(state_dict["model"])
    ckpt_str = f"ckpt{epoch:07d}EMA" if use_ema else f"ckpt{epoch:07d}"
    print(f"Loaded {ckpt_str}: from {ckpt_path}, use_ema: {use_ema}")

model_SiT.to("cuda").eval();
    
fetcher = featureFetcher_module()
for i in args.layers:
    fetcher.record_module(model_SiT.blocks[i], target_name=f"blocks.{i}")
train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
for t_scalar in args.t_scalars: 
    print(f"Processing t_scalar: {t_scalar}")
    t_beg = time.time()
    t_str = str(t_scalar).replace('.', '_')
    feature_col = extract_features_SiT(model_SiT, fetcher, train_loader, dataset_Xmean, dataset_Xstd, t_scalar)
    feature_col_test = extract_features_SiT(model_SiT, fetcher, test_loader, dataset_Xmean, dataset_Xstd, t_scalar)
    t_feat = time.time()
    print(f"Feature extraction time: {t_feat - t_beg:.2f}")
    for dimred_str in args.dim_red_method:
        # if noPCA optimizing in the full space requires a smaller learning rate
        learning_rate = 0.0005 if dimred_str == 'none' else 0.005
        model_PCA_col, PC_proj_col, results_col = train_dimred_sgd_classifiers(
                feature_col, feature_col_test, y_train, y_test, dimred_str=dimred_str,
                num_classes=40, batch_size=None,
                num_epochs=5000, print_every=250, eval_every=1000, learning_rate=learning_rate,
                device='cuda'
            )
        pkl.dump(results_col, open(join(repr_expdir,f"results_col_t{t_str}_{dimred_str}_{ckpt_str}.pkl"), "wb"))
        pkl.dump(PC_proj_col, open(join(repr_expdir,f"PC_proj_col_t{t_str}_{dimred_str}_{ckpt_str}.pkl"), "wb"))
        pkl.dump(model_PCA_col, open(join(repr_expdir,f"model_PCA_col_t{t_str}_{dimred_str}_{ckpt_str}.pkl"), "wb"))

        test_acc_synopsis = {key: results_col[key].test_record.accuracy.max() for key in results_col.keys()}
        train_acc_synopsis = {key: results_col[key].train_record.accuracy.max() for key in results_col.keys()}
        syn_df = pd.DataFrame([test_acc_synopsis, train_acc_synopsis]).T
        syn_df.columns = ["Test Accuracy", "Train Accuracy"]
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=syn_df, markers=True)
        plt.title(f"t_scalar: {t_scalar} Accuracy\n{expname}\n{ckpt_str} {dimred_str}")
        saveallforms([repr_expdir, figdir],f"t{t_str}_{dimred_str}_{ckpt_str}_accuracy.png")
        plt.show()
    del feature_col
    del feature_col_test
    del PC_proj_col

del fetcher






