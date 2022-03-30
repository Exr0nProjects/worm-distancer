import pandas as pd
import numpy as np
from numpy import exp, log
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from math import pi
from io import StringIO
from glob import glob
from operator import itemgetter as get
from tqdm import tqdm

from proc import bounding_ellipse, proc as calc_metrics

COLOR_DEFAULT = '#4a7dba'
COLOR_N2 = '#34ca6d'
COLOR_AM = '#ca3434'
COLOR_CB = '#7434ca'

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
cmaps = [hsv_cmap(256, x, 0.7) for x in np.linspace(0.4, 0.9, 4)] * 2

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

def jankily_collate_data(dfds):
    # iterrows = [dfd['df'].iterrows() for dfd in dfds]
    # for rows in zip(*iterrows):
    #     print(rows)
    #     # for row in rows:
    #     #     for x in row:
    #     #         print(x)
    #     print("aotehurockbroebk\n\n")

    # combined = pd.concat((dfd['df'] for dfd in dfds), keys=(dfd['author'] for dfd in dfds))
    # print(combined)

    for dfd in dfds:
        author, df = dfd['author'], dfd['df']
        print(author, df)
        raise NotImplementedError("how to get standard deviation?")

def jankly_show_data_distribution(dfds, row, col,
        title=None, xlabel=None, ylabel=None, strain=None):
    fig, ax = plt.subplots()
    ax.hist([dfd['df'][row][col] for dfd in dfds], color=COLOR_DEFAULT)

    ax.set_xlabel(xlabel or row)
    ax.set_ylabel(ylabel or "# of annotators")
    ax.set_title(title or f"Distribution of {row} at time {col}{' in ' + strain if strain else ''}")
    plt.show()

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


# def plot_3d_by_labeller(filenames):
#     dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
#     # 3d plotting from https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
#     ax = plt.axes(projection='3d')
#     cmaps = ['Blues', 'Greens', 'Reds']*2
#     for (labeller, df), cmap in zip(dfs, cmaps):
#         # print(labeller, '\n', df)
#         for t, row in df.iterrows():
#             ydata = row[4::2]
#             zdata = row[5::2]
#             xdata = [t] * len(ydata)
#             print(labeller, t, f'(color={cmap}) (lens)', len(ydata), len(zdata))
#             ax.scatter3D(xdata, ydata, zdata, s=500, c=zdata, cmap=cmap)
#     plt.show()

def plot_3d_by_point(filenames):
    dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    ax = plt.axes(projection='3d')

    scatters = []
    for labeller, df in dfs:
        for (t, row), cmap in zip(df.iterrows(), cmaps):
            ydata = row[4::2]
            zdata = row[5::2]
            xdata = [t] * len(ydata)
            print(labeller, t, f'(color={cmap}) (lens)', len(ydata), len(zdata))
            scatters.append((labeller, ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap=cmap, label=labeller)))

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


VIDEOS_BY_STRAIN = {
    'n2': [ 'ALS.22.3.8.#1.mp4' ],
    'am': [ 'ALS.22.3.8.#2.mp4' ],
    'cb': [ 'ALS.22.3.8.#3.mp4' ],
}
def dfd_filter(datas, author=None, strain=None, worm=None):
    # assumptions: every video contains worms of one strain, every worm is uniquely identified by video name and beginning time stamp
    def pred(dfd):
        if author is not None and dfd['author'] != author:
            return False
        if strain is not None and dfd['video'] not in VIDEOS_BY_STRAIN[strain]:
            return False
        if worm is not None and dfd['video'] + ':' + str(dfd['time']) != worm:
            return False
        return True
    return filter(pred, datas)

if __name__ == '__main__':
    datas = [dfd for fname in glob('2022_L1_locomotion_assay/*.tsv') for dfd in jankily_read_combined_data(fname)]

    datas = [{ **dfd, 'df': calc_metrics(dfd['df'], dfd['scale']) } for dfd in tqdm(datas, desc="calculating metrics...")]
    # for dfd in datas:
    #     dfd['df'].to_csv(f"out/{dfd['author']} {dfd['video']} {dfd['time']}")
    # jankily_collate_data(datas)
    jankly_show_data_distribution(dfd_filter(datas, worm='ALS.22.3.8.#1.mp4:107'), 'arclen', 5)
    jankly_show_data_distribution(dfd_filter(datas, worm='ALS.22.3.8.#1.mp4:107'), 'arclen', 4)
    jankly_show_data_distribution(dfd_filter(datas, worm='ALS.22.3.8.#1.mp4:107'), 'arclen', 3)
    jankly_show_data_distribution(dfd_filter(datas, worm='ALS.22.3.8.#1.mp4:107'), 'arclen', 2)


    # plot_all_2d(filenames)

    # plot_3d_by_point(filenames)

