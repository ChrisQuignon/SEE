from matplotlib import pyplot
import csv
import glob
from math import sqrt, tan
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import chisquare, histogram, norm
from scipy import integrate



ahead = glob.glob('recordings/teahead*')
right = glob.glob('recordings/right*')
left = glob.glob('recordings/left*')

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
                # m['tag'] = int(m['tag'])
                m['x'] = float(m['x'])
                m['y'] = float(m['y'])
                m['z'] = float(m['z'])
                m['roll'] = float(m['roll'])
                m['pitch'] = float(m['pitch'])
                m['yaw'] = float(m['yaw'])

            #filter tags
            data = [d for d in data if d['tag'] == '37599']

            #FILTER BY CAMERA
            data = [d for d in data if d['cam'] == 'lifecam20']
            ymax = max(data, lambda y : y['y'])
            print ymax[0]
            set.append(ymax[0])

            #NO FILTER
            # set.extend(data)

    # set = sorted(set, key=lambda k: k['time'])
    sets.append(set)


for set in sets:
    x = [d['x'] for d in set]
    y = [d['y'] for d in set]
    roll = [d['roll'] for d in set]

    cam = []
    for d in set:
        if d['cam'] == 'lifecam19':
            cam.append('b')
        else:
            cam.append('r')

    xs.append(x)
    ys.append(y)
    rolls.append(roll)
    cams.append(cam)

#
##PLOT CARTESIAN COORDINATES
for i in range(len(xs)):

    for j in range(len(xs[i])):
        pyplot.scatter(xs[i][j], ys[i][j], color = cams[i][j], marker = (3, 0, rolls[i][j]*57.3), facecolor='None', s = 80)
    # pyplot.scatter(m_x, m_y, color = 'g', zorder = 1, s = 40, alpha = 0.5)

    pyplot.title('Cartesian Coordinates ' + files[i])
    pyplot.ylabel('y in mm')
    pyplot.xlabel('x in mm')
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.grid(True)
    # pyplot.locator_params(nbins=10)
    pyplot.savefig('img/' + files[i]+'.png')
    pyplot.clf()

##SUBSTRACT MEAN
x_norm = []
y_norm = []

for i in range(len(xs)):
    x_norm.append([x - np.mean(xs[i]) for x in xs[i]])
    y_norm.append([y - np.mean(ys[i]) for y in ys[i]])


#CONVERT TO POLAR COODRINATES
rs = []
ps = []

for i in range(len(xs)):
    r = []
    p = []
    for j in range(len(xs[i])):
        r.append(sqrt(xs[i][j]**2 + ys[i][j] ** 2))
        p.append(tan(xs[i][j]/ys[i][j]))#sawp X an Y
        # print r, p
    rs.append(r)
    ps.append(p)
    pyplot.title('Polar Coordinates ' + files[i])
    pyplot.grid(True)
    pyplot.locator_params(nbins=10)

    for j in range(len(r)):
        pyplot.scatter(p[j], r[j], color = cams[i][j], marker = (3, 0, rolls[i][j]*57.3), facecolor='None', s = 80)

    pyplot.ylabel('distance in mm')
    pyplot.xlabel('angle in radians')
    pyplot.savefig('img/' + files[i]+ '_pc.png')
    pyplot.clf()
    # pyplot.show()

##SUBSTRACT MEAN
p_norm = []
r_norm = []
for i in range(len(xs)):
    p_norm.append([p - np.mean(ps[i])for p in ps[i]])
    r_norm.append([r - np.mean(rs[i]) for r in rs[i]])

#BOXPLOTS
pyplot.title('Boxplot of the distance')
pyplot.grid(True)
pyplot.locator_params(nbins=10)
pyplot.boxplot(rs, sym='')
pyplot.ylabel('distance in cm')
pyplot.xticks([1, 2, 3], files, rotation='horizontal')
pyplot.savefig('img/BoxplotDistance.png',dpi=50)
# pyplot.show()
pyplot.clf()

pyplot.title('Normalized Angle')
pyplot.grid(True)
pyplot.locator_params(nbins=10)
pyplot.boxplot(p_norm, vert=False, sym='')
pyplot.yticks([1, 2, 3], files, rotation='vertical')
pyplot.xlabel('angle in radians')
pyplot.savefig('img/BoxplotAngleNorm.png',dpi=50)
# pyplot.show()
pyplot.clf()
