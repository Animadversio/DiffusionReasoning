import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import join

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def saveallforms(figdirs, fignm, figh=None, fmts=("png", "pdf")):
    """Save all forms of a figure in an array of directories."""
    if type(figdirs) is str:
        figdirs = [figdirs]
    if figh is None:
        figh = plt.gcf()
    for figdir in figdirs:
        for sfx in fmts:
            figh.savefig(join(figdir, fignm+"."+sfx), bbox_inches='tight')


def estimate_CI(count, total, alpha=0.05):
    """estimate the confidence interval of a ratio"""
    p = count / total
    n = total
    lower = stats.binom.ppf(alpha/2, n, p)
    upper = stats.binom.ppf(1-alpha/2, n, p)
    return lower/n, upper/n


def shaded_error(ax, x, y, yerr_low, yerr_high, label=None, color=None, alpha=0.3):
    ax.plot(x, y, label=label, color=color)
    ax.fill_between(x, yerr_low, yerr_high, alpha=alpha, color=color)


def add_rectangles(heldout_ids, fill=False, edgecolor='red', lw=2, col_num=10, ax=None, ):
    from matplotlib.patches import Rectangle
    if ax is None:
        ax = plt.gca()
    for rule_id in heldout_ids:
        row = rule_id // col_num
        col = rule_id % col_num
        ax.add_patch(Rectangle((col, row), 1, 1, fill=fill, edgecolor=edgecolor, lw=lw))


import seaborn as sns
from rule_new_utils import relation_dict, attribute_dict
def plot_rule_heatmap(rule_mat, title="Rule Heatmap", 
                      heldout_rules=(1, 16, 20, 34, 37),
                      ax=None, cmap="viridis", fmt=".2f"):
    if ax is None:
        figh = plt.figure(figsize=(10, 5))
        ax = figh.add_subplot(111)
    plt.sca(ax)
    sns.heatmap(rule_mat, cmap=cmap, annot=True, fmt=fmt, ax=ax)
    plt.axis("image")
    add_rectangles(heldout_rules)
    plt.xticks(np.arange(10)+0.5, [relation_dict[i] for i in range(10)], rotation=45)
    plt.yticks(np.arange(4)+0.5, [attribute_dict[i] for i in range(4)], rotation=0)
    ax.set_title(title)
    return figh


def visualize_consistency(epoch_list, consistent_mat, title_str="Wide Dep x3 Blnr", figname="RAVEN10_abstract_BigBlnr", figdir="../Figures", savefig=True,):
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
    if savefig:
        fig.savefig(join(figdir,f"{figname}_rule_consistency.pdf"), dpi=300, )#bbox_inches='tight')
        fig.savefig(join(figdir,f"{figname}_rule_consistency.png"), dpi=300, )#bbox_inches='tight')
    return fig
    
    
def visualize_rule_validity(epoch_list, rules_all, title_str="Wide Dep x3 Blnr", figname="RAVEN10_abstract_BigBlnr", figdir="../Figures", savefig=True,):
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
    if savefig:
        fig.savefig(join(figdir,f"{figname}_rule_valid.pdf"), dpi=300, )#bbox_inches='tight')
        fig.savefig(join(figdir,f"{figname}_rule_valid.png"), dpi=300, )#bbox_inches='tight')
    return fig


from rule_new_utils import attribute_dict, relation_dict
# from sklearn.metrics import confusion_matrix
def visualize_cm(cm, heldout_rules, titlestr=""):
    figh1 = plt.figure(figsize=(10, 4))
    sns.heatmap(np.diag(cm).reshape(4, 10) / 1000 * 100, cmap="Blues", annot=True, fmt=".1f", cbar=False)
    add_rectangles(heldout_rules)
    plt.xticks(np.arange(10)+0.5, [relation_dict[i] for i in range(10)], rotation=45)
    plt.yticks(np.arange(4)+0.5, [attribute_dict[i] for i in range(4)], rotation=0)
    plt.title(f"Accuracy Percentage (Diagonal confusion matrix)\n{titlestr}")
    plt.show()

    figh2, ax = plt.subplots(1,1,figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix {titlestr}")
    plt.ylabel('True Rule')
    plt.xlabel('Predicted Rule')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if int(label.get_text()) in heldout_rules:
            label.set_color('red')
    plt.show()
    return figh1, figh2
    
    
def visualize_cm_default(cm, heldout_rules, titlestr=""):
    plt.figure(figsize=(10, 4))
    sns.heatmap(np.diag(cm).reshape(1,-1) / cm.sum(axis=1)[None,] * 100, cmap="Blues", annot=True, fmt=".1f", cbar=False)
    # for rule_id in heldout_rules:
    #     row = rule_id // 10
    #     col = rule_id % 10
    #     plt.gca().add_patch(Rectangle((col, row), 1, 1, fill=False, edgecolor='red', lw=2))
    plt.title(f"Accuracy Percentage (Diagonal confusion matrix)\n{titlestr}")
    plt.show()

    figh, ax = plt.subplots(1,1,figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix {titlestr}")
    plt.ylabel('True Rule')
    plt.xlabel('Predicted Rule')
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     if int(label.get_text()) in heldout_rules:
    #         label.set_color('red')
    plt.show()