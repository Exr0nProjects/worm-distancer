import pandas as pd
from math import sqrt
from itertools import islice

# FILENAME = "worm_distances.tsv"
FILENAME = "assay_test.tsv"
scale = None

while scale is None:
    # TODO: make this a feature in the distancer
    try:
        scale = float(input("What is the scale (px/mm)? "))
    except ValueError:
        print("Invalid floating point number, try again.")

if __name__ == '__main__':
    data = pd.read_csv(FILENAME, sep="	")
    data = data.set_index(['name', 'time'])

    skeleton = data/scale

    # body_angles = calc_body_angles(skeleton)

    delta_skel = skeleton.diff();

    print(delta_skel)

    metrics = ['sumabs', 'sumsquareabs', 'abssum'] + ['c' + str(i) for i in range(0, len(delta_skel.columns)//2)]
    def generate_pos_metrics(pos):
        deltax = pos[::2]
        deltay = pos[1::2]
        dists = [sqrt(x**2+y**2) for x, y in zip(deltax, deltay)]
        sumabs = sum([abs(v) for v in dists[1:]])
        sumsquareabs = sqrt(sum([v**2 for v in dists[1:]]))
        abssum = abs(sum([v for v in dists[1:]]))
        ret = [ sumabs, sumsquareabs, abssum ] + dists
        return pd.Series(ret, index=metrics)
    pos_metrics = delta_skel.apply(generate_pos_metrics, axis=1)

    print(pos_metrics)
