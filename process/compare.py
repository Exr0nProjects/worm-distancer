import pandas as pd
from matplotlib import pyplot as plt

filenames = ['j.tsv', 'm.tsv']

contrast_colors = [ (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128) ] # https://sashamaps.net/docs/resources/20-colors/

def plot_skeleton(ax, df, label):
    print(df)
    for i, row in df.iterrows():
        info = row[:3]
        pos  = row[3:]
        print(pos)
        posx = pos[1::2]
        posy = pos[1::2]
        print(posx, end='\n\n')
        ax.scatter(posx, posy, label=label)

if __name__ == '__main__':
    dfs = [(name, pd.read_csv(name, sep='	')) for name in filenames]
    print(len(dfs[0]))
    fig, axs = plt.subplots(1, len(dfs[0]))
    for (name, df), ax in zip(dfs, axs):
        plot_skeleton(ax, df, name)
