from more_itertools import windowed
import numpy as np
import pandas as pd
from math import sqrt, atan2, pi
from itertools import islice

from ellipsoid import EllipsoidTool

from sys import argv

# FILENAME = "worm_distances.tsv"
# FILENAME = "assay_test.tsv"
# FILENAME = "angle_assay_test.tsv"
# FILENAME = "angle_test.tsv"
FILENAME = "input.tsv"
scale = float(argv[1]) if len(argv) > 1 else None

def bounding_ellipse(points, tolerance=1e-4):
    center, radii, rotation = EllipsoidTool().getMinVolEllipse(points, tolerance=tolerance)
    rotation = np.arctan2(rotation[1][0], rotation[0][0]) / pi * 180   # convert rot mat to angle: https://math.stackexchange.com/a/301335
    return center, radii, rotation

def angle_to_center(ax, ay, cx, cy, bx, by):
    # return '|'.join(str(int(x)) for x in [ cx, cy, ax, ay, bx, by ])  # check that the window is working properly
    return atan2(ay-cy, ax-cx) - atan2(by-cy, bx-cx)

def generate_angle_metrics(pos):
    pos = pos[2:]   # remove centroid points
    # print('pos after rem', pos, list(windowed([int(x) for x in pos], n=6, step=2)))
    angles = [ 180 + (angle_to_center(*points) * 180 / pi) for points in windowed(pos, n=6, step=2) ]
    angles = [ 0 if x != x else x for x in angles ]
    arclen = sum(sqrt((a-c)**2 + (b-d)**2) for a, b, c, d in windowed(pos, n=4, step=2))
    center, radii, rotation = bounding_ellipse(np.reshape(list(pos), (len(pos)//2, 2)))
    data = {
        'sum_angles': sum(angles),
        'abs_angles': sum(abs(a) for a in angles),
        'arclen': arclen,
        'ellipse_cx': center[0],
        'ellipse_cy': center[1],
        'ellipse_area': pi * radii[0] * radii[1],
        'ellipse_angle': rotation,
        'ellipse_ecentricity': sqrt(1-min(radii[0], radii[1])/max(radii[0], radii[1]))
    }
    return pd.Series(list(data.values()) + angles,
            index=list(data.keys()) + ['a' + str(i) for i in range(1, len(pos)//2-1)])

def generate_pos_metrics(pos):
    metrics = ['sumabs', 'sumsquareabs', 'abssum', 'heading']\
            + ['v' + str(i) for i in range(0, len(pos)//2)]
    deltax = pos[::2]
    deltay = pos[1::2]
    dists = [sqrt(x**2+y**2) for x, y in zip(deltax, deltay)]
    heading = [atan2(y, x) for x, y in zip(deltax, deltay)][0]/pi*180
    sumabs = sum([abs(v) for v in dists[1:]])
    sumsquareabs = sqrt(sum([v**2 for v in dists[1:]]))
    abssum = abs(sum([v for v in dists[1:]]))
    ret = [ sumabs, sumsquareabs, abssum, heading ] + dists
    return pd.Series(ret, index=metrics)

def proc(filename, scale):
    data = pd.read_csv(filename, sep="	")
    data = data.set_index(['name', 'time'])
    skeleton = data/scale

    body_angles = skeleton.apply(generate_angle_metrics, axis=1)

    delta_skel = skeleton.diff();
    pos_metrics = delta_skel.apply(generate_pos_metrics, axis=1)

    return body_angles[['sum_angles', 'abs_angles']].merge(pos_metrics[['heading', 'v0']], left_index=True, right_index=True)

if __name__ == '__main__':

    while scale is None:
        # TODO: make this a feature in the distancer
        try:
            scale = float(input("What is the scale (px/mm)? "))
        except ValueError:
            print("Invalid floating point number, try again.")


    pd.options.display.float_format = "{:.2f}".format

    export = proc(FILENAME, scale)
    export.to_csv('output.tsv', sep='	')
    print("see output.tsv for output")
