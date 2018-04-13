import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
import numpy as np
from IPython.core.display import display, HTML
import pandas as pd
import os
import subprocess


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


def write_latex_table(df, name, adir='', render=True):
    # table results
    tex_name = '{}.tex'.format(name)
    filename = os.path.join(adir, '{}.tex'.format(name))
    a_str = df.to_latex(multicolumn=True, multirow=True, escape=False,index=False)
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
