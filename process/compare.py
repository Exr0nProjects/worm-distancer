import pandas as pd
from matplotlib import pyplot as plt

filenames = ['j.tsv', 'm.tsv']
# filenames = ['2022_L1_training/' + name for name in ['Peter', 'Stephanie', 'Leilani', 'Zander']]

contrast_colors = [ (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128) ] # https://sashamaps.net/docs/resources/20-colors/

def plot_skeleton_2d(ax, df, label):
    print(df)
    for i, row in df.iterrows():
        info = row[:3]
        pos  = row[3:]
        print(pos)
        posx = pos[1::2]
        posy = pos[1::2]
        print(posx, end='\n\n')
        ax.scatter(posx, posy, label=label)

def plot_all_2d(filenames):
    dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    print(len(dfs[0]))
    fig, axs = plt.subplots(1, len(dfs[0]))
    for (name, df), ax in zip(dfs, axs):
        plot_skeleton_2d(ax, df, name)
    plt.show()


def plot_all_3d(filenames):
    dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    # 3d plotting from https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    ax = plt.axes(projection='3d')
    cmaps = ['Purples', 'Blues', 'Greens', 'Reds']*2
    for (labeller, df), cmap in zip(dfs, cmaps):
        # print(labeller, '\n', df)
        for t, row in df.iterrows():
            # print(row)
            ydata = row[4::2]
            zdata = row[5::2]
            # ydata = list(row[:])
            # zdata = list(row[:])
            # ydata = ydata[4::2]
            # zdata = zdata[5::2]
            print(labeller, t, '(lens):', len(ydata), len(zdata))
            # print(ydata)
            # print(zdata)
            # print('\n\n')
            xdata = [t] * len(ydata)
            ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap=cmap)
            # print(t, row[3:])
    plt.show()

if __name__ == '__main__':
    # plot_all_2d(filenames)

    plot_all_3d(filenames)

