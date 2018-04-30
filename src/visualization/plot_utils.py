import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
import numpy as np
from IPython.core.display import display, HTML
import pandas as pd
import os
import subprocess
import itertools
import matplotlib as mpl
from collections import OrderedDict
def glance_dict(d,n=3):
    return dict(itertools.islice(d.items(), n))

def summary_pooling_table(stats):
    p_df = pd.DataFrame(stats)
    def dist(x):
        return '{:.3f} ({:.3f})'.format(np.mean(x),np.std(x))
    f = {'pid':['count'], 'r-tracks':{'mean (std)':dist},'r-artist':{'mean (std)':dist}}
    return p_df.groupby(['k','strategy','n']).agg(f)

def pooling_plots(stats):
    re_stats=[]
    for i in stats:
        re_stats.append(OrderedDict([('pid',i['pid']),('k',i['k']),('strategy',i['strategy']),('n',i['n']),('metric','r-tracks'),('value',i['r-tracks'])]))
        re_stats.append(OrderedDict([('pid',i['pid']),('k',i['k']),('strategy',i['strategy']),('n',i['n']),('metric','r-artist'),('value',i['r-artist'])]))
    for indx,grp in pd.DataFrame(re_stats).groupby('strategy'):

        sns.violinplot(x='k',y='value',hue='metric', data=grp,cut=0,split=True)
        plt.ylim([0,1])
        plt.legend(bbox_to_anchor=(0.5, 1),ncol=2)
        plt.title('Startegy = {}'.format(indx))
        plt.show()
        # distplots
    for indx,grp in pd.DataFrame(stats).groupby('strategy'):
        cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

        sns.distplot(grp['r-tracks'],label='Tracks',color=cols[0])
        dist_stats_box(grp['r-tracks'])
        plt.xlim([0,1])
        plt.xlabel('R-precision (Tracks)')
        plt.ylabel('Normalized Frequency')
        plt.title('Strategy = {}'.format(indx))
        plt.show()
        sns.distplot(grp['r-artist'],label='Artist',color=cols[1])
        dist_stats_box(grp['r-artist'])
        plt.xlim([0,1])
        plt.xlabel('R-precision (Artist)')
        plt.ylabel('Normalized Frequency')
        plt.title('Strategy = {}'.format(indx))
        plt.show()
    return


def dist_stats_box(y):
    result = stats.describe(y)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ex_labels = []
    # ex_labels.append('$n=%d$' % (result.nobs))
    ex_labels.append('$\mu=%2.2f$' % (result.mean))
    ex_labels.append('$med=%2.2f$' % (np.median(y)))
    ex_labels.append('$\sigma=%2.2f$' % (np.sqrt(result.variance)))
    ex_labels.append('$\min=%2.2f$' % (result.minmax[0]))
    ex_labels.append('$\max=%2.2f$' % (result.minmax[1]))
    ex_handles = [mpatches.Patch(
        color='white', alpha=0.0, visible=False) for i in ex_labels]
    plt.legend(handles=handles + ex_handles, labels=labels + ex_labels,
               loc='best', frameon=False)

    return


def html_header(txt, lvl=1):
    return display(HTML('<h{:d}>{:s}</h{:d}>'.format(lvl, txt, lvl)))


LATEX_TABLE = r'''\documentclass{{standalone}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{graphicx}}
\usepackage{{xcolor,colortbl}}

\begin{{document}}

{}

\end{{document}}
'''


def pandas_settings():
    pd.set_option('precision', 2)
    pd.options.display.float_format = '{:,.2f}'.format
    return

def plot_settings():
    """Plot settings"""
    # wierdly have to plot stuff first to get
    # matplotlib to accept
    # plt.figure(figsize=(12, 6))
    # plt.show()
    # awesome plot options
    sns.set_style("white")
    sns.set_style('ticks')

    sns.set_context("paper", font_scale=2.25)
    # matplotlib stuff
    plt_params = {
        'figure.figsize': (10, 6.5),
        'lines.linewidth': 3,
        'axes.linewidth': 2.5,
        'savefig.dpi': 300,
        'xtick.major.width': 2.5,
        'ytick.major.width': 2.5,
        'xtick.minor.width': 1,
        'ytick.minor.width': 1,
    }

    mpl.rcParams.update(plt_params)

    return

def write_latex_table(df, name, adir='.', render=True,index=True):
    # table results
    tex_name = '{}.tex'.format(name)
    filename = os.path.join(adir, '{}.tex'.format(name))
    a_str = df.to_latex(multicolumn=True, multirow=True, escape=False,index=index)
    with open(filename, 'w') as a_file:
        a_file.write(LATEX_TABLE.format(a_str))
    if render:
        p = subprocess.Popen(['pdflatex', tex_name], cwd=adir)
        p.wait()
        pdf_file = tex_name.replace('.tex', '.pdf')
        png_file = tex_name.replace('.tex', '.png')
        p = subprocess.Popen(
            ['convert', '-density', '300', pdf_file, '-quality', '90', png_file], cwd=adir)
        p.wait()
        # remove latex stuff
        subprocess.Popen(['rm', '{}.aux'.format(name)], cwd=adir)
        subprocess.Popen(['rm', '{}.log'.format(name)], cwd=adir)
    return
