import pandas as pd
import numpy as np
# from numpy import exp, log
from matplotlib import pyplot as plt
# from scipy.spatial.transform import Rotation
# from math import pi
from io import StringIO
from glob import glob
# from operator import itemgetter as get
# from tqdm import tqdm
import pickle
from collections import defaultdict

import statistics as stats
import math


from proc import bounding_ellipse, proc as calc_metrics

COLOR_DEFAULT = '#4a7dba'
COLOR_N2 = '#34ca6d'
COLOR_AM = '#ca3434'
COLOR_CB = '#7434ca'

TIME_RANGE = [-5, 6] # inc exc

filenames = ['j.tsv', 'm.tsv']
filenames = ['2022_L1_training/' + name + '.tsv' for name in ['Peter', 'Stephanie', 'Zander']]

def hsv_cmap(num_steps, h, s, modulate_opacity=False):
    from matplotlib.colors import ListedColormap
    from colorsys import hsv_to_rgb
    return ListedColormap(np.array([[
        *hsv_to_rgb(h, s, i),
        i if modulate_opacity else 1
    # ] for i in np.linspace(0, 1, num_steps)]))
    # ] for i in np.logspace(-4, 0, num=num_steps, endpoint=True, base=1.3)]))
    # ] for i in np.logspace(-1.9, 0, num=num_steps, endpoint=True, base=1.8)]))
    ] for i in np.logspace(-1, 0, num=num_steps, endpoint=True, base=2)]),
    name=f"hsv_cmap({h}, {s}, {modulate_opacity})")

# cmaps = ['Blues', 'Greens', 'Reds']*4
# cmaps = ['summer', 'cool', 'winter']*4
cmaps = [hsv_cmap(256, x, 0.7) for x in np.linspace(0.4, 0.9, 4)] * 10
# cmaps = [hsv_cmap(256, 0.5908 + x/20, 0.6022) for x in range(-3, 4)]

contrast_colors = [ (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128) ] # https://sashamaps.net/docs/resources/20-colors/


def jankily_read_combined_data(filename):
    ret = []

    with open(filename, "r") as rf:
        line = rf.readline().strip().split('	')
        if len(line) == 2:
            scale = float(line[0])  # unsafe: panics
            author = line[1]
        else:
            print(f"WARNING!!!! no float scalar found, treating data as empty {filename}")
            print(line)
            rf.seek(0)

        stripped = '\n'.join(l.strip() for l in rf.read().split('\n'))
    dfs_str = list(filter(lambda s: len(s), stripped.split("\n\n")))
    for df_str in dfs_str:
        df = pd.read_csv(StringIO(df_str), sep="	")
        ret.append({ 'author': author, 'scale': scale, 'video': df['name'][0], 'time': df['time'][0], 'df': df })

    return ret

def jankly_show_data_distribution(dfds, row, col,
        title=None, xlabel=None, ylabel=None, strain=None):
    fig, ax = plt.subplots()
    ax.hist([dfd['df'][row][col] for dfd in dfds], color=COLOR_DEFAULT)

    ax.set_xlabel(xlabel or row)
    ax.set_ylabel(ylabel or "# of annotators")
    ax.set_title(title or f"Distribution of {row} at time {col}{' in ' + strain if strain else ''}")
    plt.show()

def jankily_stddev(dfds, blacklist=[]):
    """Takes the cellwise stdev of a list of CSVs."""
    dfs = [dfd['df'] for dfd in dfds]
    out = dfs[0].copy() # OPTM copy expensive

    for name in dfs[0].columns:
        if name in blacklist:
            print(name)
            continue
        for i, _x in enumerate(dfs[0][name]):
            points = [df[name][i] for df in dfs]
            if math.isnan(set(points).pop()):
                out[name][i] = float('nan')
            else:
                out[name][i] = stats.stdev(
                    list(filter(lambda x: not math.isnan(x), points))
                )
    return out

def jankily_mean(dfds):
    dfs = [dfd['df'] for dfd in dfds]
    out = dfs[0].copy() # OPTM

    for name in dfs[0].columns:
        for i, _x in enumerate(dfs[0][name]):
            points = [df[name][i] for df in dfs]
            if math.isnan(set(points).pop()):
                out[name][i] = float('nan')
            else:
                out[name][i] = stats.fmean(
                    list(filter(lambda x: not math.isnan(x), points))
                )
    return out

def jankily_collate_by_worm(dfdss):
    ret = []
    for dfds_per_strain in dfdss:
        annotators_by_worm = defaultdict(list)
        for dfd_per_author in dfds_per_strain:
            dfd = dfd_per_author
            print('dfd', dfd, '\n\n\n')
            annotators_by_worm[f"{dfd['video']}:{dfd['time']}"].append(dfd)

        # TODO: average standard deviation https://www.statology.org/averaging-standard-deviations/
        # print(list(annotators_by_worm.values()))
        dfds_per_strain = []
        for strain in annotators_by_worm.values():
            print('strain', strain, '\n\n\n')
            dfds_per_strain.append({ **strain[0],
                'scale': stats.fmean(dfd['scale'] for dfd in strain),
                'author': ', '.join(dfd['author'] for dfd in strain),
                'df': jankily_mean(strain) })
        ret.append(dfds_per_strain)
    print("\n\n\n\n")
    print(ret)
    print("\n\n", len(ret), '\n\n')
    print(*[len(x) for x in ret], '\n\n\n\n\n\n\n')
    return ret

def jankily_make_line_plot(dfdss, col,
        colors=[COLOR_N2, COLOR_AM, COLOR_CB],
        labels=['N2', 'AM725', 'CB1338'],
        title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    # print(dfdss[0])
    sizes = [len(dfds) for dfds in dfdss]
    means = [jankily_mean(dfds) for dfds in dfdss]
    stddevs = [jankily_stddev(dfds) for dfds in dfdss]

    for df in means: print(df); print('\n')
    for df in stddevs: print(df); print('\n')

    for n, mean, stdd, color, label in zip(sizes, means, stddevs, colors, labels):
        mean = np.array([x for x in mean[col] if not math.isnan(x)])
        stdd = np.array([x for x in stdd[col] if not math.isnan(x)])
        frames = list(range(*(TIME_RANGE if len(mean) == 11 else [-4, 6])))
        assert len(mean) == len(stdd)
        ax.errorbar(frames, mean, yerr=stdd,
                color=color, elinewidth=1,
                label=f"{label} (n={n})")
        ax.fill_between(frames, mean + stdd, mean - stdd, alpha=0.1, color=color)

    ax.set_title(title or f"{col} across strains")
    ax.set_xlabel(xlabel or 'frame')
    if ylabel: ax.set_ylabel(ylabel)
    ax.legend()
    # plt.show()

from matplotlib.patches import Ellipse
def plot_bounding_ellipse(ax, points, label, tolerance=1e-4):
    center, radii, rotation = bounding_ellipse(points, tolerance)
    print(center, radii, rotation)
    return ax.add_patch(Ellipse(center, width=radii[0]*2, height=radii[1]*2,
                                angle=rotation, facecolor='none', edgecolor='blue'))


def plot_skeleton_2d(ax, df, label):
    for i, row in df.iterrows():
        info = row[:3]
        pos  = row[3:]
        posx = pos[1::2]
        posy = pos[2::2]
        print(f'plotting for {label}')
        ax.scatter(posx, posy, label=label)

        # fit ellipse
        P = np.array([posx, posy], dtype='float64').T
        plot_bounding_ellipse(ax, P, label)

def plot_all_2d(filenames):
    dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    # print(len(dfs[0]))
    fig, axs = plt.subplots(1, len(dfs[0]))
    for (name, df), ax in zip(dfs, axs):
        plot_skeleton_2d(ax, df, name)
    plt.show()


def plot_3d_by_point(dfds):
    # dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    ax = plt.axes(projection='3d')

    scatters = []
    for dfd in dfds:
        labeller, df = dfd['author'], dfd['df']
        for (t, row), cmap in zip(list(df.iterrows())[-8:-3], cmaps):
            ydata = row[4::2]
            zdata = row[5::2]
            xdata = [t] * len(ydata)
            print(labeller, t, f'(color={cmap}) (lens)', len(ydata), len(zdata))
            scatters.append((labeller, ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap=cmap, label=labeller, alpha=1)))

    # create tooltips (https://towardsdatascience.com/tooltips-with-pythons-matplotlib-dcd8db758846)
    annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points")

    class TooltipManager:
        def __init__(self, scatters, tooltip):
            self.scatters = scatters
            self.tooltip = tooltip
            self.last_tip = ""

        def handle_hover(self, event):
            labels = set()
            for label, sc in self.scatters:
                cont, _ = sc.contains(event)
                if cont:
                    labels.add(label)

            if len(labels):
                self.tooltip.xy = (event.xdata, event.ydata)
                label_text = f"{', '.join(n for n in labels)}"
            else:
                label_text = ""

            if self.last_tip != label_text:
                annot.set_text(label_text)
                plt.gcf().canvas.draw()
                self.last_tip = label_text

    tooltip_mgr = TooltipManager(scatters, annot)
    plt.gcf().canvas.mpl_connect("motion_notify_event", tooltip_mgr.handle_hover)

    plt.show()



def plot_3d_by_point_split(dfds):
    # dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    # create tooltips (https://towardsdatascience.com/tooltips-with-pythons-matplotlib-dcd8db758846)

    class TooltipManager:
        def __init__(self, scatters, tooltip):
            self.scatters = scatters
            self.tooltip = tooltip
            self.last_tip = ""

        def handle_hover(self, event):
            labels = set()
            for label, sc in self.scatters:
                cont, _ = sc.contains(event)
                if cont:
                    labels.add(label)

            if len(labels):
                self.tooltip.xy = (event.xdata, event.ydata)
                label_text = f"{', '.join(n for n in labels)}"
            else:
                label_text = ""

            if self.last_tip != label_text:
                annot.set_text(label_text)
                plt.gcf().canvas.draw()
                self.last_tip = label_text


    # create scatters
    scatters = []
    for time in range(len(dfds[0]['df'])):
        fig, ax = plt.subplots()
        ax.set_title(f"time = {time}")
        annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points")
        for dfd in dfds:
            labeller, df = dfd['author'], dfd['df']
            for (t, row), cmap in zip(df.iterrows(), cmaps):
                if t != time: continue
                ydata = row[4::2]
                zdata = row[5::2]
                xdata = [t] * len(ydata)
                print("plotting one")
                print(labeller, t)
                # print(labeller, t, f'(color={cmap}) (lens)', len(ydata), len(zdata))
                scatters.append((labeller, ax.scatter(ydata, zdata, c=np.linspace(0, 1, 10), cmap=cmap, label=labeller, alpha=1)))

                ax.set_title(f"{dfd['author']} {dfd['video']} {dfd['time']} {t}")

        # more tooltip stuff
        tooltip_mgr = TooltipManager(scatters, annot)
        plt.gcf().canvas.mpl_connect("motion_notify_event", tooltip_mgr.handle_hover)
        plt.show()
        print("plt show")


    plt.show()


VIDEOS_BY_STRAIN = {
    # 'n2': [ 'ALS.22.3.8.#1.mp4', 'LA.ALS.3.25.22.#2.mp4' ],
    # 'am': [ 'ALS.22.3.8.#2.mp4', 'LA.ALS.3.25.22.#3.mp4' ],
    # 'cb': [ 'ALS.22.3.8.#3.mp4', 'LA.ALS.3.25.22.#1.mp4' ],
    'n2': [ 'LA.ALS.3.25.22.#2.mp4' ],
    'am': [ 'LA.ALS.3.25.22.#3.mp4' ],
    'cb': [ 'LA.ALS.3.25.22.#1.mp4' ],
    'n2oil': [ 'AL.ALS.3.25.22.#6.mp4' ],
    'cboil': [ 'LA.ALS.3.25.22.#7.mp4' ],
}
def dfd_filter(datas, author=None, notauthor=None, strain=None, worm=None):
    # assumptions: every video contains worms of one strain, every worm is uniquely identified by video name and beginning time stamp
    def pred(dfd):
        # print("checking", dfd['author'], dfd['video'], dfd['time'])
        if author is not None and dfd['author'] != author:
            # print("    author whitelist failed", dfd['author'])
            return False
        if notauthor is not None and dfd['author'] in notauthor:
            # print("    author blacklist triggered", dfd['author'])
            return False
        if strain is not None and dfd['video'] not in VIDEOS_BY_STRAIN[strain]:
            return False
        if worm is not None and f"{dfd['video']}:{str(dfd['time']) }" != worm:
            # print('    worm failed', dfd['author'], dfd['video'], dfd['time'])
            return False
        # print(' => returning true for dfd', dfd['author'], dfd['video'], dfd['time'])
        return True
    return [x for x in datas if pred(x)]

if __name__ == '__main__':
#     datas = [dfd for fname in glob('2022_L1_locomotion_assay/3-25-22/*.tsv') for dfd in jankily_read_combined_data(fname)]
#
# #
# #     for dfd in datas:
# #         print(f"processing {dfd['author']} {dfd['video']}")
# #         calc_metrics(dfd['df'], dfd['scale'])
# #
#     datas = [{ **dfd, 'df': calc_metrics(dfd['df'], dfd['scale']) } for dfd in tqdm(datas, desc="calculating metrics...")]
#
#     with open('all_data_procced.pickle', 'wb') as wf:
#         pickle.dump(datas, wf)

    with open('all_data_procced.pickle', 'rb') as rf:
        datas = pickle.load(rf)

    # POSTER TODO: realign headings; how to collate multiple worms of the same strain

    # worms = ['LA.ALS.3.25.22.#1.mp4:90', 'AL.ALS.3.25.22.#6.mp4:56', 'LA.ALS.3.25.22.#7.mp4:20']
    # datas_n2, datas_n2oil, datas_cboil = [dfd_filter(datas, notauthor=['zander'], worm=worm) for worm in worms]
    strains = ['n2', 'am', 'cb']
    datas_n2, datas_n2oil, datas_cboil = [dfd_filter(datas, notauthor=['zander'], strain=strain) for strain in strains]
    print(len(datas_n2), len(datas_n2oil), len(datas_cboil))
    data_in_strains = jankily_collate_by_worm([datas_n2, datas_n2oil, datas_cboil])
    jankily_make_line_plot(data_in_strains, 'heading', labels=strains)

    # print([(dfd['author'], dfd['df']) for dfd in datas_n2])
    # jankly_show_data_distribution(datas_n2, 'sum_angles', 2)


    # plot_all_2d(filenames)


    plt.savefig('out.png', dpi=300)
