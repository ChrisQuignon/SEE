from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import csv
import glob
from math import sqrt, tan
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import chisquare, histogram, norm
from scipy import integrate



ahead = glob.glob('recordings/forward_exp/*/*')
left = glob.glob('recordings/left_exp/*/*')
right = glob.glob('recordings/rigt_exp/*/*')

xs = []
ys = []
rolls = []
cams = []

files=['right', 'ahead', 'left']

keys = ['time', 'cam', 'tag', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']

sets = []

for filenames in [right, ahead, left]:
    # print set
    set = []

    for file in filenames:
        with open(file, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ', fieldnames = keys)
            data = [x for x in reader]

            for m in data:
                m['time'] = int(m['time'])
                m['tag'] = int(m['tag'])
                m['x'] = float(m['x'])
                m['y'] = float(m['y'])
                m['z'] = float(m['z'])
                m['roll'] = float(m['roll'])
                m['pitch'] = float(m['pitch'])
                m['yaw'] = float(m['yaw'])

            #filter tags
            # data = [d for d in data if d['tag'] == '22203']

            #FILTER BY CAMERA
            # data = [d for d in data if d['cam'] == 'lifecam19']
            # ymax = max(data, lambda y : y['y'])
            # print ymax[0]
            # set.append(ymax[0])

            #NO FILTER
            set.extend(data)

    # set = sorted(set, key=lambda k: k['time'])
    sets.append(set)


#TRANSFORM
tf19 = np.asarray([  1.1924880638503051e-08, 9.9999999999999978e-01, -4.5000000536619623e+01, 9.9999999999999978e-01, -1.1924880638503051e-08, -4.4999999463380362e+01, 0., 0., 1. ])
tf19 = np.reshape(tf19, (3, 3))

tf20 = np.asarray([ 1.1924880638503051e-08, 9.9999999999999978e-01, -4.4999982875871396e+01, 9.9999999999999978e-01, -1.1924880638503051e-08, 1.4360000005366194e+03, 0., 0., 1. ])
tf20 = np.reshape(tf20, (3, 3))


# #INTRINSIC
# tf19 = np.asarray([1.4514377e+03, 0., 0.9520943e+03, 0., 1.45109189e+03, 0.5617813e+03, 0., 0., 1.])
# tf19 = np.reshape(tf19, (3, 3))
#
# tf20 = np.asarray([1.46918029377e+03, 0., 0.96358764231e+03, 0., 1.46870303635e+03, 0.56708041227e+03, 0., 0., 1.])
# tf20 = np.reshape(tf20, (3, 3))


# #WARP
# tf19 = np.asarray([-0.078682, 2.710943, -12.072611, -2.508042, 1.308308, 3227.267900, -0.000204, 0.000301, 0.802686])
# tf19 = np.reshape(tf19, (3, 3))
#
# tf20 = np.asarray([-0.654994, 1.777761, 881.728210, -2.143361, 0.260860, 2904.716553, -0.000242, 0.000109, 0.759868])
# tf20 = np.reshape(tf20, (3, 3))

# T_trans * T_rot

vs = []





for set in sets:
    ds = []
    for d in set:
        v = np.asarray([d['x'], d['y'], d['z']])
        c = 'g'
        if d['cam'] == "lifecam19":
            v = v.dot(tf19)
            c = 'b'
        if d['cam'] == "lifecam20":
            v = v.dot(tf20)
            c = 'r'
        # # v = v.T
        ds.append(v)
    vs.append(ds)

for v in vs:
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([x[0] for x in v], [x[1] for x in v], [x[2] for x in v], c = c,  linestyle = '-', marker = 'o')
    pyplot.show()
