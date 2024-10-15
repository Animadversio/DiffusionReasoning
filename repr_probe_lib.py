import time
import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from easydict import EasyDict as edict
from collections import defaultdict
from tqdm.auto import tqdm

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
    results.train_record = train_record
    results.test_record = test_record
    results.test_acc = test_acc
    results.pred_cls = pred_cls
    return model, results


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
        # featmat = feature_col[layerkey].view(len(feature_col[layerkey]), -1)
        # featmat_test = feature_col_test[layerkey].view(len(feature_col_test[layerkey]), -1)
        featmat = feature_col[layerkey].reshape(len(feature_col[layerkey]), -1)
        featmat_test = feature_col_test[layerkey].reshape(len(feature_col_test[layerkey]), -1)

        # Compute mean of training features
        featmean = featmat.mean(dim=0)
        featdim = featmat.shape[1]
        t1 = time.time()

        if noPCA or (featdim <= PC_dim):
            # note the special case where PCdim is higher than the feature dimension, then no need. 
            # Center and normalize features without PCA
            feat_PCA = (featmat - featmean[None, :]).to(device)
            feat_PCA_std = feat_PCA.std(dim=0)
            feat_PCA = feat_PCA / feat_PCA_std[None, :]
            feat_PCA_test = (featmat_test - featmean[None, :]).to(device)
            feat_PCA_test = feat_PCA_test / feat_PCA_std[None, :]
            V = None
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
            'V': V,  # PCA components
            'mean': featmean.cpu(),
            'std': feat_PCA_std.cpu()
        }
        results_col[layerkey] = results_dict
        del feat_PCA, feat_PCA_test

    return model_PCA_col, PC_proj_col, results_col


def extract_features_DiT(
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


def extract_features_GPT(
    model,
    fetcher,
    data_loader,
    device='cuda',
    cond = False,
    progress_bar=True
):
    """
    Extracts features from specified layers of the model for the given dataset.
    Note, for GPT, some module output is tuple, so we need to take the first element. (e.g. GPT2Block) this is not thorougly tested. !! 

    Args:
        model (torch.nn.Module): The neural network model.
        fetcher (FeatureFetcher): An instance of the featureFetcher_module.
        cond (bool, optional): Whether to condition on the labels. Defaults to False.
        data_loader (DataLoader): DataLoader for the dataset.
        device (str, optional): Device to perform computations on. Defaults to 'cuda'.
        progress_bar (bool, optional): Whether to display a progress bar. Defaults to True.

    Returns:
        dict: A dictionary with layer keys and concatenated activation tensors.
    """
    feature_col = defaultdict(list)
    loader = tqdm(data_loader) if progress_bar else data_loader
    for X_batch, y_batch in loader:
        # Prepare model inputs
        if cond:
            model_kwargs = {'y': y_batch.to(device)}
        else:
            model_kwargs = {'y': th.zeros(X_batch.size(0), dtype=th.long, device=device)}
        # Forward pass with no gradient computation
        with th.no_grad():
            model.forward(X_batch.to(device), **model_kwargs)
        # Collect activations
        for key, activations in fetcher.activations.items():
            # for GPT2Block, the output is a tuple of length 2, the first is the hidden state, the second is sth. else like attentions.
            if isinstance(activations, list) or isinstance(activations, tuple):
                activations = activations[0].detach().cpu()
            feature_col[key].append(activations)

    # Concatenate all activations for each layer
    for key in feature_col:
        feature_col[key] = th.cat(feature_col[key], dim=0)
        print(f"{key}: {feature_col[key].shape}")
    return feature_col


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
    elif dimred_str == "avgspace":
        feature_red_col = {k: v.mean(dim=(2,3)) for k, v in feature_col.items()}
        feature_red_col_test = {k: v.mean(dim=(2,3)) for k, v in feature_col_test.items()}
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

