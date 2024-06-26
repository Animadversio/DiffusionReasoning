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

