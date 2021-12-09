import pandas as pd
from math import sqrt, atan2, pi
from itertools import islice
from more_itertools import windowed

# FILENAME = "worm_distances.tsv"
# FILENAME = "assay_test.tsv"
# FILENAME = "angle_assay_test.tsv"
FILENAME = "angle_test.tsv"
scale = None

while scale is None:
    # TODO: make this a feature in the distancer
    try:
        scale = float(input("What is the scale (px/mm)? "))
    except ValueError:
        print("Invalid floating point number, try again.")

def angle_to_center(ax, ay, cx, cy, bx, by):
    # return '|'.join(str(int(x)) for x in [ cx, cy, ax, ay, bx, by ])  # check that the window is working properly
    return atan2(ay-cy, ax-cx) - atan2(by-cy, bx-cx)

def generate_angle_metrics(pos):
    pos = pos[2:]
    angles = [ 180 + (angle_to_center(*points) * 180 / pi) for points in windowed(pos, n=6, step=2) ]
    angles = [ 0 if x != x else x for x in angles ]
    return pd.Series([sum(angles), sum(abs(a) for a in angles)] + angles,
            index=['sum_angles', 'abs_angles'] + ['a' + str(i) for i in range(1, len(pos)//2-1)])

def generate_pos_metrics(pos):
    metrics = ['sumabs', 'sumsquareabs', 'abssum'] + ['c' + str(i) for i in range(0, len(pos)//2)]
    deltax = pos[::2]
    deltay = pos[1::2]
    dists = [sqrt(x**2+y**2) for x, y in zip(deltax, deltay)]
    sumabs = sum([abs(v) for v in dists[1:]])
    sumsquareabs = sqrt(sum([v**2 for v in dists[1:]]))
    abssum = abs(sum([v for v in dists[1:]]))
    ret = [ sumabs, sumsquareabs, abssum ] + dists
    return pd.Series(ret, index=metrics)

if __name__ == '__main__':
    data = pd.read_csv(FILENAME, sep="	")
    data = data.set_index(['name', 'time'])
    skeleton = data/scale

    body_angles = skeleton.apply(generate_angle_metrics, axis=1)

    print(body_angles)

    delta_skel = skeleton.diff();
    pos_metrics = delta_skel.apply(generate_pos_metrics, axis=1)

    print(pos_metrics)
