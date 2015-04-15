from matplotlib import pyplot
import csv
import glob
from math import sqrt, tan
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches

#NEEDS scipy 0.12 (anaconda)
# from scipy.stats import chi2_contingency

# files = glob.glob('*.csv')
files = ['data_left.csv', 'data_ahead.csv', 'data_right.csv']
#files = glob.glob('data_ahead.csv')

xs = []
ys = []
mx = []
my = []
names=[]

for file in files:
    data = {}
    names.append(file)


    with open(file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        data =  [x for x in reader]

    x = [float (d['x']) for d in data]
    y = [float (d['y']) for d in data]

    xs.append(x)
    ys.append(y)

    mx.append(sum(x) / float(len(x)))
    my.append(sum(y) / float(len(y)))

# #PLOT SINGLE RUNS
# for i in range(len(xs)):
#
#     pyplot.scatter(xs[i], ys[i], c = 'black', marker = 'x', zorder = 4)
#     # pyplot.scatter(m_x, m_y, color = 'g', zorder = 1, s = 40, alpha = 0.5)
#
#     # #PLOT FAILURE
#     # if len(xs[i]) > 20:
#     #     pyplot.scatter(x[20:], y[20:], c = 'red', marker = 'x', zorder = 4)
#
#     pyplot.title('Cartesian Coordinates ' + files[i][:-4])
#     pyplot.axes().set_aspect('equal', 'datalim')
#     pyplot.grid(True)
#     pyplot.locator_params(nbins=10)
#     pyplot.savefig('img/' + files[i][:-4]+'.png')
#     pyplot.clf()

#EXCLUDE BAD POINT
# for i in range(len(xs)):
#     xs[i] = xs[i][:20]
#     ys[i] = ys[i][:20]

##SUBSTRACT MEAN
x_norm = []
y_norm = []

for i in range(len(xs)):
    x_norm.append([x - mx[i] for x in xs[i]])
    y_norm.append([y - my[i] for y in ys[i]])

# ##show Names
# pyplot.title('X Axis')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(xs, vert=False)
# pyplot.yticks([1, 2, 3], names, rotation='vertical')
# pyplot.savefig('img/BoxplotX.png',dpi=50)
# # pyplot.show()
# pyplot.clf()
#
# pyplot.title('Normalized X Axis')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(x_norm)
# pyplot.xticks([1, 2, 3], names)
# pyplot.savefig('img/BoxplotXNorm.png',dpi=50)
# # pyplot.show()
# pyplot.clf()
#
# pyplot.title('Y Axis')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(ys)
# pyplot.xticks([1, 2, 3], names, rotation='horizontal')
# pyplot.savefig('img/BoxplotY.png',dpi=50)
# # pyplot.show()
# pyplot.clf()
#
# pyplot.title('Normalized Y Axis')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(y_norm)
# pyplot.xticks([1, 2, 3], names)
# pyplot.savefig('img/BoxplotYNorm.png',dpi=50)
# # pyplot.show()
# pyplot.clf()
#
# # #PLOT RUNS OVERLAYED
# for i in range(len(xs)):
#     x_norm = [x - mx[i] for x in xs[i]]
#     y_norm = [y - my[i] for y in ys[i]]
#
#     pyplot.scatter(x_norm, y_norm, c = pyplot.cm.spectral(i/float(len(xs))), s = 40.0, alpha = 0.8)
#
# pyplot.title('Cartesian Overlay')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.grid(True)
# # pyplot.locator_params(nbins=10)
#
# pyplot.savefig('img/overlay.png')
# # pyplot.show()
# pyplot.clf()

#CONVERT TO POLAR COODRINATES

rs = []
ps = []

mp = []
mr = []

for i in range(len(xs)):
    r = []
    p = []
    for j in range(len(xs[i])):
        # print xs[i][j], ys[i][j]
        r.append(sqrt(xs[i][j]**2 + ys[i][j] ** 2))
        p.append(tan(xs[i][j]/ys[i][j]))#sawp X an Y
        # print r, p
    rs.append(r)
    ps.append(p)
    mp.append(sum(p) / float(len(p)))
    mr.append(sum(r) / float(len(r)))
    # pyplot.title('Polar Coordinates ' + files[i][:-4])
    # pyplot.grid(True)
    # pyplot.locator_params(nbins=10)
    # pyplot.scatter(p, r, c = 'black', marker = 'x')
    # pyplot.savefig('img/' + files[i][:-4]+ '_pc.png')
    # pyplot.clf()
    # # pyplot.show()

##SUBSTRACT MEAN
p_norm = []
r_norm = []
for i in range(len(xs)):
    p_norm.append([p - mp[i] for p in ps[i]])
    r_norm.append([r - mr[i] for r in rs[i]])
#
# ##show Names
# pyplot.title('Angle')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(ps, vert=False)
# pyplot.yticks([1, 2, 3], names, rotation='vertical')
# pyplot.savefig('img/BoxplotAngle.png',dpi=50)
# # pyplot.show()
# pyplot.clf()
#
# pyplot.title('Distance')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(rs)
# pyplot.xticks([1, 2, 3], names, rotation='horizontal')
# pyplot.savefig('img/BoxplotDistance.png',dpi=50)
# # pyplot.show()
# pyplot.clf()
#
# pyplot.title('Normalized Angle')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(p_norm, vert=False)
# pyplot.yticks([1, 2, 3], names, rotation='vertical')
# pyplot.savefig('img/BoxplotAngleNorm.png',dpi=50)
# # pyplot.show()
# pyplot.clf()
#
# pyplot.title('Normalized Distance')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
# pyplot.boxplot(r_norm)
# pyplot.xticks([1, 2, 3], names, rotation='horizontal')
# pyplot.savefig('img/BoxplotDistanceNorm.png',dpi=50)
# # pyplot.show()
# pyplot.clf()


##HISTOGRAMS
setNames = ['Xs', 'Ys', 'Distances', 'Angles']
for idx, set in enumerate([xs, ys, rs, ps]):

    for jdx, vals in enumerate(set):
        name = setNames[idx] + ' ' + files[jdx][:-4]


        mu = np.mean(vals)
        sigma = np.std(vals)

        # the histogram of the data
        n, bins, patches = pyplot.hist(vals, 8, normed=1, alpha=0.75)

        # add a 'best fit' line
        y = mlab.normpdf( bins, mu, sigma)
        l = pyplot.plot(bins, y, 'r--', linewidth=1)

        mean_error = sqrt(sigma/len(vals))
        error_std_dev = sqrt(sigma **2 / 2 * len(vals))

        s = "Standard error of mean/deviation: {0}/{1}".format(round(mean_error,4), round(error_std_dev,4))
        pyplot.title('Histogram: ' + name)

        pyplot.xlabel(s)
        # pyplot.ylabel("Frequency")
        # pyplot.grid(True)
        # pyplot.show()
        pyplot.savefig('img/' + name + '.png',dpi=50)
        pyplot.clf()
