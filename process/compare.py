import pandas as pd
import numpy as np
# from numpy import exp, log
from matplotlib import pyplot as plt
# from scipy.spatial.transform import Rotation
# from math import pi
from io import StringIO
from glob import glob
# from operator import itemgetter as get
from tqdm import tqdm
import pickle
from collections import defaultdict

import statistics as stats
import math


from proc import bounding_ellipse, proc as calc_metrics

COLOR_DEFAULT, COLOR_DEFAULT_HS = '#4a7dba', (0.5908, 0.6022)
COLOR_N2, COLOR_N2_HS = '#34ca6d', (0.3967, 0.7426)
COLOR_CB, COLOR_CB_HS = '#7434ca', (0.7378, 0.7426)
COLOR_AM, COLOR_AM_HS = '#ca3434', (0.0000, 0.7426)
COLOR_N2OIL, COLOR_N2OIL_HS = '#34cabe', (0.4866, 0.7419)
COLOR_CBOIL, COLOR_CBOIL_HS = '#9d34ca', (0.7833, 0.7426)
COLOR_AMOIL, COLOR_AMOIL_HS = '#ca7034', (0.0667, 0.7426)

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
cmaps = [hsv_cmap(256, x, 0.7) for x in np.linspace(0.4, 0.9, 3)]
# cmaps = [hsv_cmap(256, 0.5908 + x/20, 0.6022) for x in range(-3, 4)]

contrast_colors = [ (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128) ] # https://sashamaps.net/docs/resources/20-colors/


def jankily_read_combined_data(filename):
    ret = []

    with open(filename, "r") as rf:
        line = rf.readline().strip().split('	')
        if len(line) == 2:
            scale = float(line[0])*2  # unsafe: panics
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
    """plots the distribution of one metric for one worm at one timestamp across annotators.
    used to graphically check that the annotators are not wildly disagreeing."""
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
            # print('dfd', dfd, '\n\n\n')
            annotators_by_worm[f"{dfd['video']}:{dfd['time']}"].append(dfd)

        # TODO: average standard deviation https://www.statology.org/averaging-standard-deviations/
        dfds_per_strain = []
        for strain in annotators_by_worm.values():
            dfds_per_strain.append({ **strain[0],
                'scale': stats.fmean(dfd['scale'] for dfd in strain),
                'author': ', '.join(dfd['author'] for dfd in strain),
                'df': jankily_mean(strain) })
        ret.append(dfds_per_strain)
    return ret

def jankily_make_line_plot(dfdss, col,
        colors=[COLOR_N2, COLOR_AM, COLOR_CB],
        labels=['N2', 'AM725', 'CB1338'],
        title=None, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    sizes = [len(dfds) for dfds in dfdss]
    means = [jankily_mean(dfds) for dfds in dfdss]
    stddevs = [jankily_stddev(dfds) for dfds in dfdss]

    # for df in means: print(df); print('\n')
    # for df in stddevs: print(df); print('\n')

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



def plot_worms_per_timestep(dfds):
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
    # commented out data from 3-10-22 sheet
    # 'n2': [ 'ALS.22.3.8.#1.mp4', 'LA.ALS.3.25.22.#2.mp4' ],
    # 'am': [ 'ALS.22.3.8.#2.mp4', 'LA.ALS.3.25.22.#3.mp4' ],
    # 'cb': [ 'ALS.22.3.8.#3.mp4', 'LA.ALS.3.25.22.#1.mp4' ],
    'n2': [ 'LA.ALS.3.25.22.#2.mp4' ],
    'am': [ 'LA.ALS.3.25.22.#3.mp4' ],
    'cb': [ 'LA.ALS.3.25.22.#1.mp4' ],
    'n2+cbd': [ 'AL.ALS.3.25.22.#6.mp4' ],
    'am+cbd': [ 'LA.ALS.3.25.22.#7.mp4' ],
}
def dfd_filter(datas, author=None, notauthor=None, strain=None, worm=None):
    '''filter a list of dfd (dataframe descriptions) by author allow/denylist, worm strain, or worm id
    where worm_id is of the form "{video_filename}:{first_line_timestamp}"
    '''
    # assumptions: every video contains worms of one strain, every worm is uniquely identified by video name and beginning time stamp
    def pred(dfd): # predicate
        # print("checking", dfd['author'], dfd['video'], dfd['time'])
        if author is not None and dfd['author'] != author:
            # print("    author whitelist failed", dfd['author'])
            return False
        if notauthor is not None and dfd['author'] in notauthor:
            # print("    author blacklist triggered", dfd['author'])
            return False
        if strain is not None and dfd['video'] not in VIDEOS_BY_STRAIN[strain.lower()]:
            return False
        if worm is not None and f"{dfd['video']}:{str(dfd['time']) }" != worm:
            # print('    worm failed', dfd['author'], dfd['video'], dfd['time'])
            return False
        # print(' => returning true for dfd', dfd['author'], dfd['video'], dfd['time'])
        return True
    return [x for x in datas if pred(x)]

if __name__ == '__main__':
    # read all the files in the folder
    datas = [dfd for fname in glob('2022_L1_locomotion_assay/3-25-22/*.tsv') for dfd in jankily_read_combined_data(fname)]

    # visualize all of the points for a given worm. the plot_3d_by_point visualization fn takes in skeleton points not metrics, so this must happen before calc_metrics
    ##datas = dfd_filter(datas, worm='LA.ALS.3.25.22.#1.mp4:90')
    ##plot_3d_by_point(datas)


##  calculate the metrics for each file individually. This is useful for debugging (if calc_metrics fails on one of the files, you know that that file has corrupt data)
##     for dfd in datas:
##         print(f"processing {dfd['author']} {dfd['video']}")
##         calc_metrics(dfd['df'], dfd['scale'])
##

    # calculate metrics for each dataframe, in order to convert from skeletal points to metrics like { heading, velocity, ellipse ecentricity, etc }
    datas = [{ **dfd, 'df': calc_metrics(dfd['df'], dfd['scale']) } for dfd in tqdm(datas, desc="calculating metrics...")]

    # use these blocks to write the processed data to a file and read it back, if the data isn't changing. This is useful if calc_metrics is taking too long.
##    with open('all_data_procced.pickle', 'wb') as wf:
##        pickle.dump(datas, wf)

##    with open('all_data_procced.pickle', 'rb') as rf:
##        datas = pickle.load(rf)





    # an example of plotting all worms in the strain
    # each of the following double-commented lines show an example of selecting what strains and colors to use
    #strains, colors = ['N2', 'N2+CBD', 'CB'], [COLOR_N2, COLOR_AM, COLOR_CB]
    #strains, colors = ['N2', 'N2+CBD'], [COLOR_N2, COLOR_N2OIL]
    #strains, colors = ['N2', 'AM+CBD'], [COLOR_N2, COLOR_AMOIL]
    strains, colors = ['N2', 'AM', 'AM+CBD', 'CB'], [COLOR_N2, COLOR_AM, COLOR_AMOIL, COLOR_CB]

    data_in_strains = [dfd_filter(datas, notauthor=['zander'], strain=strain) for strain in strains]  # filters the data to select the strains that we want, so we can plot each strain seprately
    print([len(x) for x in data_in_strains])
    data_in_strains = jankily_collate_by_worm(data_in_strains)                                        # collate the data by worm, so that we only plot each worm once (average over annotators)
    # these next three lines set which charts we want to produce. They are zipped together, so the first element of each list is applied to the first chart and so on.
    metric_id = ['arclen', 'ellipse_ecentricity', 'heading', 'v0']
    metric_display = ['Body length', 'Ellipse eccentricity', 'Heading', 'Centroid speed']
    metric_units = ['mm', '(unitless, 0-1)', 'degrees', 'mm/frame']
    for id, display, units in zip(metric_id, metric_display, metric_units):
        jankily_make_line_plot(data_in_strains, id, labels=strains, title=f"{display} by strain", ylabel=units, colors=colors)
        plt.savefig(f"out/{','.join(strains)}-{id}.png", dpi=300)
        # input('press enter to continue')    # uncomment if you want to preview the chart and have the script wait for input before moving on





    # comparing indivudal worms, possibly across strains. The structure of this block of code is very similar to above, but with different filtering to create different charts
    # use one of the next three lines to select what set of worms you want to create a chart with

    #worms, labels, colors, strains = [f'AL.ALS.3.25.22.#6.mp4:{time}' for time in [56, 64, 99, 159, 224, 242, 332, 481, 607]], [f'N2+CBD:{time}' for time in [56, 64, 99, 159, 224, 242, 332, 481, 607]], hsv_cmap(9, *COLOR_N2OIL_HS).colors, ['N2+CBD']   # all of the N2+CBDs
    #worms, labels, colors, strains = [f'LA.ALS.3.25.22.#2.mp4:{time}' for time in [70, 222, 309]] + [f'AL.ALS.3.25.22.#6.mp4:{time}' for time in [56, 64, 99]], [f'N2:{time}' for time in [70, 22, 309]] + [f'N2+CBD:{time}' for time in [56, 64, 99]], list(hsv_cmap(3, *COLOR_N2_HS).colors) + list(hsv_cmap(3, *COLOR_N2OIL_HS).colors), ['N2', 'N2+CBD']  # N2 vs N2 + CBD individuals
    worms, labels, colors, strains = [f'LA.ALS.3.25.22.#3.mp4:{time}' for time in [51, 85, 200, 290]] + [f'LA.ALS.3.25.22.#7.mp4:{time}' for time in [20, 85, 245, 358]], [f'AM:{time}' for time in [51, 85, 200, 290]] + [f'AM+CBD:{time}' for time in [20, 85, 245, 358]], list(hsv_cmap(4, *COLOR_AM_HS).colors) + list(hsv_cmap(4, *COLOR_AMOIL_HS).colors), ['AM', 'AM+CBD']  # AM vs AM + CBD individuals

    worms_dfdss = [dfd_filter(datas, notauthor=['zander'], worm=worm) for worm in worms]
    print(len(worms_dfdss), [len(x) for x in worms_dfdss])

    metric_id = ['arclen', 'ellipse_ecentricity', 'heading', 'v0']
    metric_display = ['Body length', 'Ellipse eccentricity', 'Heading', 'Centroid speed']
    metric_units = ['mm', '(unitless, 0-1)', 'degrees', 'mm/frame']
    for id, display, units in zip(metric_id, metric_display, metric_units):
        jankily_make_line_plot(worms_dfdss, id, labels=labels, title=f"{display} ({' vs '.join(strains)})", ylabel=units, colors=colors)
        plt.savefig(f"out/{','.join(strains)}-{id}.png", dpi=300)
        # input('press enter to continue')





    # show data distribution to validate annotators
    # print([(dfd['author'], dfd['df']) for dfd in datas_n2])
    # jankly_show_data_distribution(datas_n2, 'sum_angles', 2)

