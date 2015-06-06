from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import csv
import glob
from math import sqrt, acos
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import chisquare, histogram, norm
from scipy import integrate



ahead = glob.glob('recordings/ahead*')
left = glob.glob('recordings/left*')
right = glob.glob('recordings/right*')

files=['left', 'ahead', 'right']

keys = ['time', 'cam', 'x', 'y', 'z', ]

dirs = [left, ahead, right]

recordings = []
for d in dirs:
    direction = []
    for file in d:
        run = []
        with open(file, 'rb') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=' ', fieldnames = keys)
            run = [x for x in reader]

        for m in run:
            m['time'] = int(m['time'])
            m['x'] = float(m['x'])
            m['y'] = float(m['y'])
            m['z'] = float(m['z'])
        direction.append(run)
    recordings.append(direction)



# FILTER

#LEFT
for i, run in enumerate(recordings[0]):
    run  = [d for d in run if d['y'] > 1990]
    run  = [d for d in run if d['x'] > 1700]
    recordings[0][i] = run

#AHEAD
for i, run in enumerate(recordings[1]):
    run  = [d for d in run if d['y'] > 2000]
    recordings[1][i] = run

#RIGHT
for i, run in enumerate(recordings[2]):
    run  = [d for d in run if d['y'] < 3000]
    run  = [d for d in run if d['x'] > 1750]
    recordings[2][i] = run

#CUT OFF THE MARKER FLAPS
#Maybe with v or w, after completion

def complete(run):
    r = []
    x1 = run[0]['x']
    y1 = run[0]['y']
    t1 = run[0]['time']
    theta1 = 0.0

    for d in run[1:]:
        x2 = d['x']
        y2 = d['y']
        t2 = d['time']

        dt = t2-t1
        if (x2-x1) == 0:
            break
        dtheta = np.arctan((y2-y1)/(x2-x1))


        theta2 = theta1 + dtheta
        v = (y2 -y1) / dt
        w = (x1 - x2)/dt

        x1 = x2
        y1 = y2
        t1 = t2
        theta = theta2

        n = {
            'cam':d['cam'],
            'time':d['time'],
            'z':d['z'],
            'x':x2,
            'y':y2,
            'dt':dt,
            'v':v,
            'w':w,
            "theta":theta
        }
        r.append(n)
    return r

for i, direction in enumerate(recordings):
    for j, run in enumerate(direction):
        c = complete(run)
        recordings[i][j] = c

#CIRCLE FIT adopted from https://gist.github.com/lorenzoriano/6799568

import numpy as np
from scipy import optimize
# from matplotlib import pyplot as plt, cm, colors
from math import pi

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri     = calc_R(x, y, *center)
    R      = Ri.mean()
    residu = np.sum((Ri - R)**2)
    return xc, yc, R, residu

centers = []

for direction in [recordings[0], recordings[2]]:
    x = []
    y = []
    for run in direction:
        x.extend([d['x'] for d in run])
        y.extend([d['y'] for d in run])
    xc, yc, R, residu = leastsq_circle(x, y)
    centers.append((xc, yc))
