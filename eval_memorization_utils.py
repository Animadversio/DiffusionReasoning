import einops
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

train_attrs_cache = {}
attr_img_tsr_cache = {}
# train_attr_fn = "attr_all_1000k.pt" 
# data_root = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/RPM_dataset/RPM1000k"
def get_RAVEN_dataset(data_root="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/RPM_dataset/RPM1000k", 
                      train_attr_fn="attr_all_1000k.pt", n_classes=40, cmb_per_class=4000, heldout_ids=(), cmb_offset=0, cache=True):
    if not cache:
        train_attrs = torch.load(f'{data_root}/{train_attr_fn}') # [35, 10000, 3, 9, 3]
        attr_img_tsr = einops.rearrange(train_attrs,  'cls (B R) p (H W) attr -> cls B attr (R H) (p W)', H=3,W=3,p=3,R=3,attr=3,)
    else:
        if train_attr_fn in attr_img_tsr_cache:
            attr_img_tsr = attr_img_tsr_cache[train_attr_fn]
        else:
            train_attrs = torch.load(f'{data_root}/{train_attr_fn}')
            attr_img_tsr = einops.rearrange(train_attrs,  'cls (B R) p (H W) attr -> cls B attr (R H) (p W)', H=3,W=3,p=3,R=3,attr=3,)
            attr_img_tsr_cache[train_attr_fn] = attr_img_tsr
    
    max_default_cmb = attr_img_tsr.shape[1]
    if cmb_per_class > max_default_cmb:
        raise ValueError(f'cmb_per_class should be less than {max_default_cmb}')
    train_cls_msk = torch.ones(n_classes, dtype=bool)
    for heldout_id in heldout_ids:
        train_cls_msk[heldout_id] = False
    X = attr_img_tsr[:, cmb_offset:cmb_offset + cmb_per_class]
    y = torch.arange(0, n_classes).unsqueeze(1).expand(n_classes, cmb_per_class).to(int)
    X = einops.rearrange(X[train_cls_msk, :], 'cls B attr H W -> (cls B) attr H W')
    y = einops.rearrange(y[train_cls_msk, :], 'cls B -> (cls B)')
    return X, y


def check_fraction_train_set(gen_sample_rows_list, train_X_row_set):
    cnt = 0
    for i, row in enumerate(gen_sample_rows_list):
        if row in train_X_row_set:
            cnt += 1
            # print(i, row)
    total = len(gen_sample_rows_list)
    frac = cnt / total
    return cnt, frac


def check_fraction_train_set_idxlist(gen_sample_rows_list, train_X_row_set):
    cnt = 0
    idx_list = []
    for i, row in enumerate(gen_sample_rows_list):
        if row in train_X_row_set:
            cnt += 1
            idx_list.append(i)
            # print(i, row)
    total = len(gen_sample_rows_list)
    frac = cnt / total
    return cnt, frac, idx_list


def extract_sample_mat_set(sample_tsr, outtype=set):
    """Expecting 
    sample_tsr: torch.Tensor, shape=(B, attr, H, W), usually (B, 3, 9, 9)
    """
    sample_tsr = sample_tsr.round().int()
    sample_rows = einops.rearrange(sample_tsr, 'B attr Rh Cw -> (B) (attr Rh Cw)', attr=3, Rh=9, Cw=9).numpy().astype(int)
    assert sample_rows.shape[1] == 3 * 81, sample_rows.shape
    sample_row_set = outtype([tuple(sample_rows[i]) for i in range(sample_rows.shape[0])])
    return sample_rows, sample_row_set


def extract_row_mat_set(sample_tsr, outtype=set):
    """Expecting 
    sample_tsr: torch.Tensor, shape=(B, attr, H, W), usually (B, 3, 9, 9)
    """
    sample_tsr = sample_tsr.round().int()
    sample_rows = einops.rearrange(sample_tsr, 'B attr (R h) W -> (B R) (attr h W)', R=3, h=3, W=9).numpy().astype(int)
    assert sample_rows.shape[1] == 81, sample_rows.shape
    sample_row_set = outtype([tuple(sample_rows[i]) for i in range(sample_rows.shape[0])])
    return sample_rows, sample_row_set


def extract_panel_mat_set(sample_tsr, outtype=set):
    """Expecting 
    sample_tsr: torch.Tensor, shape=(B, attr, H, W), usually (B, 3, 9, 9)
    """
    sample_tsr = sample_tsr.round().int()
    sample_panels = einops.rearrange(sample_tsr, 'B attr (R h) (P w) -> (B R P) (attr h w)', R=3, h=3, P=3, w=3).numpy().astype(int)
    assert sample_panels.shape[1] == 27, sample_panels.shape
    sample_panel_set = outtype([tuple(sample_panels[i]) for i in range(sample_panels.shape[0])])
    return sample_panels, sample_panel_set


def extract_attr_row_mat_set(sample_tsr, attr_id, outtype=set):
    """Expecting 
    sample_tsr: torch.Tensor, shape=(B, attr, H, W), usually (B, 3, 9, 9)
    """
    sample_tsr = sample_tsr.round().int()
    sample_rows = einops.rearrange(sample_tsr[:, attr_id:attr_id+1], 'B attr (R h) W -> (B R) (attr h W)', R=3, h=3, W=9).numpy().astype(int)
    assert sample_rows.shape[1] == 27, sample_rows.shape
    sample_row_set = outtype([tuple(sample_rows[i]) for i in range(sample_rows.shape[0])])
    return sample_rows, sample_row_set


def extract_attr_panel_mat_set(sample_tsr, attr_id, outtype=set):
    """Expecting 
    sample_tsr: torch.Tensor, shape=(B, attr, H, W), usually (B, 3, 9, 9)
    """
    sample_tsr = sample_tsr.round().int()
    sample_panels = einops.rearrange(sample_tsr[:, attr_id:attr_id+1], 'B attr (R h) (P w) -> (B R P) (attr h w)', R=3, h=3, P=3, w=3).numpy().astype(int)
    assert sample_panels.shape[1] == 9, sample_panels.shape
    sample_panel_set = outtype([tuple(sample_panels[i]) for i in range(sample_panels.shape[0])])
    return sample_panels, sample_panel_set


def extract_training_set_row_panel_sets(train_tsr_X, return_set=True):
    train_tsr_X_samples, train_X_sample_set = extract_sample_mat_set(train_tsr_X)
    train_tsr_X_rows, train_X_row_set = extract_row_mat_set(train_tsr_X)
    train_tsr_X_panels, train_X_panel_set = extract_panel_mat_set(train_tsr_X)
    train_X_row_set_attr_col = {}
    train_X_panel_set_attr_col = {}
    for attr_id in range(3):
        train_tsr_X_rows_attr, train_X_row_set_attr = extract_attr_row_mat_set(train_tsr_X, attr_id)
        train_tsr_X_panels_attr, train_X_panel_set_attr = extract_attr_panel_mat_set(train_tsr_X, attr_id)
        train_X_row_set_attr_col[attr_id] = train_X_row_set_attr
        train_X_panel_set_attr_col[attr_id] = train_X_panel_set_attr
    if return_set:
        return train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col
    else:
        return train_tsr_X_samples, train_tsr_X_rows, train_tsr_X_panels, train_tsr_X_rows_attr, train_tsr_X_panels_attr


def eval_memorization_all_level_sample_tsr(sample_tsr, train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col):
    """
    Process the sample_col[epoch] and update statistics.

    Parameters:
    sample_col_epoch: The sample column for the given epoch.
    ctrl_X_row_set: Control set for rows.
    ctrl_X_panel_set: Control set for panels.
    ctrl_X_row_set_attr_col: Control set for row attributes.
    ctrl_X_panel_set_attr_col: Control set for panel attributes.

    Returns:
    stats_entry: A dictionary containing the statistics.
    """
    stats_entry = {}

    _, gen_sample_samples_list = extract_sample_mat_set(sample_tsr, outtype=list)
    _, gen_sample_rows_list = extract_row_mat_set(sample_tsr, outtype=list)
    _, gen_sample_panels_list = extract_panel_mat_set(sample_tsr, outtype=list)
    samplecnt, samplefrac = check_fraction_train_set(gen_sample_samples_list, train_X_sample_set)
    rowcnt, rowfrac = check_fraction_train_set(gen_sample_rows_list, train_X_row_set)
    pancnt, panfrac = check_fraction_train_set(gen_sample_panels_list, train_X_panel_set)
    stats_entry.update({"mem_samplecnt": samplecnt, "mem_samplefrac": samplefrac,
        "mem_rowcnt": rowcnt, "mem_rowfrac": rowfrac, "mem_pancnt": pancnt, "mem_panfrac": panfrac})

    for attri in range(3):
        _, gen_sample_panels_list_attr = extract_attr_panel_mat_set(sample_tsr, attri, outtype=list)
        _, gen_sample_rows_list_attr = extract_attr_row_mat_set(sample_tsr, attri, outtype=list)
        attr_rowcnt, attr_rowfrac = check_fraction_train_set(gen_sample_rows_list_attr, train_X_row_set_attr_col[attri])
        attr_pancnt, attr_panfrac = check_fraction_train_set(gen_sample_panels_list_attr, train_X_panel_set_attr_col[attri])
        stats_entry.update({f"mem_attr{attri}_rowcnt": attr_rowcnt, f"mem_attr{attri}_rowfrac": attr_rowfrac,
                            f"mem_attr{attri}_pancnt": attr_pancnt, f"mem_attr{attri}_panfrac": attr_panfrac})

    return stats_entry


def compute_memorization_tab_through_training(sample_col, eval_col, train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col, abinit=False):
    """
    abinit, flag for GPT type model, specify whether use the abinit generation statistics.
    """
    mem_stats_col = []
    for epoch in tqdm(sample_col.keys()):
        stats_entry = {"epoch": epoch}
        if "c3_cnt" in eval_col[epoch]:
            C3_cnt, C2_cnt, valid_cnt, total_cnt = eval_col[epoch]['c3_cnt'], eval_col[epoch]['c2_cnt'], eval_col[epoch]['anyvalid_cnt'], eval_col[epoch]['total']
        else:
            stats_dict = eval_col[epoch]['stats_abinit'] if abinit else eval_col[epoch]['stats']
            C3_cnt, C2_cnt, valid_cnt, total_cnt = stats_dict['C3'], stats_dict['C2'], stats_dict['anyvalid'], stats_dict['total']
        stats_entry.update({"C3": C3_cnt, "C2": C2_cnt, "valid": valid_cnt, "total": total_cnt})
        stats_entry.update(eval_memorization_all_level_sample_tsr(sample_col[epoch], train_X_sample_set, train_X_row_set, train_X_panel_set, train_X_row_set_attr_col, train_X_panel_set_attr_col))
        mem_stats_col.append(stats_entry)
        
    mem_stats_df = pd.DataFrame(mem_stats_col)
    return mem_stats_df


def visualize_memorization_dynamics(mem_stats_df, expname=''):
    figh, axs = plt.subplots(1, 2, figsize=(10, 6))
    plt.sca(axs[0])
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_rowfrac", label="full row")   
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr0_rowfrac", label="row attr0")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr1_rowfrac", label="row attr1")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr2_rowfrac", label="row attr2")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_samplefrac", label="full sample")
    plt.title("Row memorization fraction")
    plt.ylabel("row fraction")
    plt.xlabel("step")
    plt.legend()
    plt.sca(axs[1])
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_panfrac", label="full panel")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr0_panfrac", label="panel attr0")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr1_panfrac", label="panel attr1")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr2_panfrac", label="panel attr2")
    plt.title("Panel memorization fraction")
    plt.ylabel("panel fraction")
    plt.xlabel("step")
    plt.legend()
    plt.suptitle(f"Memorization of training set at various level\n{expname}")
    # saveallforms(figexpdir, "memorization_levels_dynamics", figh)
    plt.show()
    return figh


def visualize_memorization_with_ctrl_dynamics(mem_stats_df, mem_stats_ctrl_df, expname=''):
    figh, axs = plt.subplots(1, 2, figsize=(10, 6))
    plt.sca(axs[0])
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_rowfrac", label="full row")   
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr0_rowfrac", label="row attr0")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr1_rowfrac", label="row attr1")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr2_rowfrac", label="row attr2")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_samplefrac", label="full sample")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_rowfrac", label="full row (ctrl)", linestyle=':', color="C0")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_attr0_rowfrac", label="row attr0 (ctrl)", linestyle=':', color="C1")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_attr1_rowfrac", label="row attr1 (ctrl)", linestyle=':', color="C2")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_attr2_rowfrac", label="row attr2 (ctrl)", linestyle=':', color="C3")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_samplefrac", label="ctrl sample (ctrl)", linestyle=':', color="C4")
    plt.title("Row memorization fraction")
    plt.ylabel("row fraction")
    plt.xlabel("step")
    plt.legend()
    plt.sca(axs[1])
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_panfrac", label="full panel")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr0_panfrac", label="panel attr0")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr1_panfrac", label="panel attr1")
    sns.lineplot(data=mem_stats_df, x="epoch", y="mem_attr2_panfrac", label="panel attr2")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_panfrac", label="full panel (ctrl)", linestyle=':', color="C0")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_attr0_panfrac", label="panel attr0 (ctrl)", linestyle=':', color="C1")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_attr1_panfrac", label="panel attr1 (ctrl)", linestyle=':', color="C2")
    sns.lineplot(data=mem_stats_ctrl_df, x="epoch", y="mem_attr2_panfrac", label="panel attr2 (ctrl)", linestyle=':', color="C3")
    plt.title("Panel memorization fraction")
    plt.ylabel("panel fraction")
    plt.xlabel("step")
    plt.legend()
    plt.suptitle(f"Memorization of training set at various level\n{expname}")
    # saveallforms(figexpdir, "memorization_levels_dynamics", figh)
    plt.show()
    return figh


def compute_memorization_binary_joint(sample_tsr, train_X_panel_set):
    _, gen_sample_panels_list = extract_panel_mat_set(sample_tsr, outtype=list)
    pancnt, panfrac, idxlist = check_fraction_train_set_idxlist(gen_sample_panels_list, train_X_panel_set)
    sample_total = len(sample_tsr)
    row_total = sample_total * 3
    panel_total = len(gen_sample_panels_list)
    assert panel_total == sample_total * 9 
    panel_mem_binary_vec = np.zeros((panel_total, ), dtype=bool)
    panel_mem_binary_vec[idxlist] = True
    panel_mem_bin_row_mat = panel_mem_binary_vec.reshape((row_total, 3))
    panel_mem_bin_sample_mat = panel_mem_binary_vec.reshape((sample_total, 9))
    return panel_mem_bin_row_mat, panel_mem_bin_sample_mat


# Visualize joint density of binary variables. 
def plot_binary_patterns_with_connections(counts, binary_patterns=None, text="count", ax=None):
    # Example usage
    # counts = np.array([3305, 253, 103, 289, 59, 196, 370, 1569])
    # plot_binary_patterns_with_connections(counts);
    # Generate all possible combinations of 3D binary patterns
    binary_patterns_all = np.array(list(itertools.product([False, True], repeat=3)))
    if binary_patterns is None:
        binary_patterns = np.array(list(itertools.product([False, True], repeat=3)))

    # Define new coordinates for each of the 8 binary patterns to avoid overlap
    new_coords = {
        (False, False, False): (0, 0),
        (False, False, True): (1, 0),
        (False, True, False): (0, 1),
        (False, True, True): (1, 1),
        (True, False, False): (0.5, 0.5),
        (True, False, True): (1.5, 0.5),
        (True, True, False): (0.5, 1.5),
        (True, True, True): (1.5, 1.5)
    }

    # Extract new x and y coordinates based on the mapping
    x_coords = np.array([new_coords[tuple(pattern)][0] for pattern in binary_patterns])
    y_coords = np.array([new_coords[tuple(pattern)][1] for pattern in binary_patterns])
    x_coords_all = np.array([new_coords[tuple(pattern)][0] for pattern in binary_patterns_all])
    y_coords_all = np.array([new_coords[tuple(pattern)][1] for pattern in binary_patterns_all])
    # Define adjacency pairs based on binary patterns
    adjacency_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 3), # connections for False, False, False
        (4, 5), (4, 6), (5, 7), (6, 7), # connections for True, False, False
        (0, 4), (1, 5), (2, 6), (3, 7)  # connections between True and False in the first dimension
    ]
    # Plot the scatter plot with circle markers and connecting lines
    if ax is None:
        figh = plt.figure(figsize=(6, 6))
        ax = figh.add_subplot(111)
    else:
        figh = ax.figure
        plt.sca(ax)
    # Plot lines connecting adjacent points
    for pair in adjacency_pairs:
        plt.plot([x_coords_all[pair[0]], x_coords_all[pair[1]]],
                 [y_coords_all[pair[0]], y_coords_all[pair[1]]], 
                 color='gray', linestyle='-', linewidth=1)
    # Plot the points
    if text == "count":
        plt.scatter(x_coords, y_coords, s=counts, alpha=0.6, color='blue', marker='o')
    elif text == "freq":
        plt.scatter(x_coords, y_coords, s=counts / counts.sum() * 10000, alpha=0.6, color='blue', marker='o')
    # Adding labels
    for i in range(len(counts)):
        pattern_str = ''.join(['T' if x else 'F' for x in binary_patterns[i]])
        if text == "count":
            plt.text(x_coords[i], y_coords[i], f"{pattern_str}\n{counts[i]}", fontsize=16, ha='right')
        elif text == "freq":
            plt.text(x_coords[i], y_coords[i], f"{pattern_str}\n{counts[i] / counts.sum():.3f}", fontsize=16, ha='right')
        
    plt.title('Scatter Plot of Binary Patterns with Connections')
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.xlim(-0.5, 2.)
    plt.ylim(-0.5, 2.)
    # plt.show()
    # plt.xlabel('Redesigned X Coordinate')
    # plt.ylabel('Redesigned Y Coordinate')
    # plt.grid(True)
    return figh

# statistical test for independence of binary variables
def build_contingency_table(data):
    """
    Builds a 3D contingency table for three binary variables using numpy.unique.
    
    :param data: NumPy array of shape (n_samples, 3) with boolean values.
    :return: 3D NumPy array of shape (2, 2, 2) with counts.
    """
    # Ensure the data is a NumPy array of integers (0 and 1)
    data_int = data.astype(int)
    # Compute unique rows and their counts
    unique_rows, counts = np.unique(data_int, axis=0, return_counts=True)
    # Initialize a 3D contingency table with zeros
    contingency_table = np.zeros((2, 2, 2), dtype=int)
    # Populate the contingency table
    for row, count in zip(unique_rows, counts):
        a, b, c = row
        contingency_table[a, b, c] = count
    
    return contingency_table


def calculate_cramers_v(chi2, n, min_dim):
    """
    Calculates Cramér's V statistic.
    
    :param chi2: Chi-square statistic from the test.
    :param n: Total number of observations.
    :param min_dim: Minimum dimension (min(k_i - 1) for each variable i).
    :return: Cramér's V value.
    """
    return np.sqrt(chi2 / (n * min_dim))

