import pandas as pd
import numpy as np
from numpy import exp, log
from matplotlib import pyplot as plt
from ellipsoid import EllipsoidTool
from scipy.spatial.transform import Rotation
from math import pi

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

from matplotlib.patches import Ellipse
def plot_bounding_ellipse(ax, points, label, tolerance=1e-4):
    center, radii, rotation = EllipsoidTool().getMinVolEllipse(points, tolerance=tolerance)
    rotation = np.arctan2(rotation[1][0], rotation[0][0])   # convert rot mat to angle: https://math.stackexchange.com/a/301335
    print(center, radii, rotation)
    return ax.add_patch(Ellipse(center, width=radii[0]*2, height=radii[1]*2,
                                angle=rotation / pi * 180, facecolor='none', edgecolor='blue'))


def plot_skeleton_2d(ax, df, label):
    print(df)
    for i, row in df.iterrows():
        info = row[:3]
        pos  = row[3:]
        # print(pos)
        posx = pos[1::2]
        posy = pos[2::2]
        # print(posx, end='\n\n')
        print(posx, posy)
        print(f'plotting for {label}')
        ax.scatter(posx, posy, label=label)

        # fit ellipse
        if True:
            P = np.array([posx, posy], dtype='float64').T
            plot_bounding_ellipse(ax, P, label)

        break

def plot_all_2d(filenames):
    dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    print(len(dfs[0]))
    fig, axs = plt.subplots(1, len(dfs[0]))
    for (name, df), ax in zip(dfs, axs):
        plot_skeleton_2d(ax, df, name)
        break
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
                # annot.set_visible(True)
            else:
                label_text = ""
                # annot.set_visible(False)

            if self.last_tip != label_text:
                annot.set_text(label_text)
                plt.gcf().canvas.draw()
                self.last_tip = label_text

    tooltip_mgr = TooltipManager(scatters, annot)
    plt.gcf().canvas.mpl_connect("motion_notify_event", tooltip_mgr.handle_hover)

    plt.show()


if __name__ == '__main__':
    fig, ax = plt.subplots()
    points = np.array([[0, 0], [0.8, 4.2], [1, 0], [0, 1], [0, -1]])
    ax.scatter(points[:, 0], points[:, 1])
    plot_bounding_ellipse(ax, points, 'none')
    plt.show()
    # plot_all_2d(filenames)

    # plot_3d_by_point(filenames)

