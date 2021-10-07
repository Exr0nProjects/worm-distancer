import pandas as pd
from math import sqrt
from itertools import islice

FILENAME = "worm_distances.tsv"
scale = None

while scale is None:
    # TODO: make this a feature in the distancer
    try:
        scale = float(input("What is the scale (px/mm)? "))
    except ValueError:
        print("Invalid floating point number, try again.")

if __name__ == '__main__':
    data = pd.read_csv(FILENAME, sep="	")
    data = data.set_index('time')

    info = data.iloc[:,0:1]
    skeleton = data.iloc[:,1:]
    skeleton = skeleton/scale
    skeleton = skeleton.diff();

    metrics = ['sumabs', 'sumsquareabs', 'abssum'] + ['c' + str(i) for i in range(0, len(skeleton.columns)//2)]

    skels = pd.DataFrame(columns=metrics)
    for idx, pos in islice(skeleton.iterrows(), 1, None):
        deltax = pos[::2]
        deltay = pos[1::2]
        dists = [sqrt(x**2+y**2) for x, y in zip(deltax, deltay)]
        sumabs = sum([abs(v) for v in dists[1:]])
        sumsquareabs = sqrt(sum([v**2 for v in dists[1:]]))
        abssum = abs(sum([v for v in dists[1:]]))
        to_app = { k: v for v, k in zip([sumabs, sumsquareabs, abssum] + dists, metrics) }
        skels = skels.append(to_app, ignore_index=True)

    print(skels)
