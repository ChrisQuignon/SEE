from matplotlib import pyplot
import csv
import glob
from math import sqrt, tan
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import chisquare, histogram, norm

files = ['data_left.csv', 'data_ahead.csv', 'data_right.csv']

xs = []
ys = []
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

##PLOT CARTESIAN COORDINATES
for i in range(len(xs)):

    pyplot.scatter(xs[i], ys[i], c = 'black', marker = 'x', zorder = 4)
    # pyplot.scatter(m_x, m_y, color = 'g', zorder = 1, s = 40, alpha = 0.5)

    pyplot.title('Cartesian Coordinates ' + files[i][:-4])
    pyplot.ylabel('y in cm')
    pyplot.xlabel('x in cm')
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.grid(True)
    pyplot.locator_params(nbins=10)
    pyplot.savefig('img/' + files[i][:-4]+'.png')
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
    pyplot.title('Polar Coordinates ' + files[i][:-4])
    pyplot.grid(True)
    pyplot.locator_params(nbins=10)
    pyplot.scatter(p, r, c = 'black', marker = 'x')

    pyplot.ylabel('distance in cm')
    pyplot.xlabel('angle in radians')
    pyplot.savefig('img/' + files[i][:-4]+ '_pc.png')
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
pyplot.boxplot(rs)
pyplot.ylabel('distance in cm')
pyplot.xticks([1, 2, 3], names, rotation='horizontal')
pyplot.savefig('img/BoxplotDistance.png',dpi=50)
# pyplot.show()
pyplot.clf()

pyplot.title('Normalized Angle')
pyplot.grid(True)
pyplot.locator_params(nbins=10)
pyplot.boxplot(p_norm, vert=False)
pyplot.yticks([1, 2, 3], names, rotation='vertical')
pyplot.xlabel('angle in radians')
pyplot.savefig('img/BoxplotAngleNorm.png',dpi=50)
# pyplot.show()
pyplot.clf()

##HISTOGRAMS
setNames = ['Distances', 'Angles']#'Xs', 'Ys',
for idx, set in enumerate([rs, ps]):#xs, ys,

    for jdx, vals in enumerate(set):
        name = setNames[idx] + '_' + files[jdx][:-4]

        #CALCULATING MU AND SIGMA
        mu = np.mean(vals)
        sigma = np.std(vals)

        # #ERRORS - not needed
        # mean_error = sqrt(sigma/len(vals))
        # error_std_dev = sqrt(sigma **2 / 2 * len(vals))


        # HISTOGRAM
        n, bins, patches = pyplot.hist(vals, bins=6)

        #TODO USE RANGES HERE
        # n, bins, patches = pyplot.hist(vals, range=[])

        #GAUSSIAN
        y = mlab.normpdf( bins, mu, sigma)
        pyplot.plot(bins, y, 'r--', linewidth=1)

        #CHI SQUARED TEST
        chisq, p = chisquare(bins, y)

        print 'Histogram: ' + name + ':'
        print 'mu: ' + str(mu)
        print 'sigma: ' + str(sigma)
        print "chisq:" + str(chisq)
        print "p: " + str(p)
        # print 'Standard error of mean: ' + str(mean_error)
        # print 'Standard erro of deviation: ' + str(error_std_dev)
        print ''

        s = 'Histogram: ' + files[jdx][:-4]
        pyplot.title(s)

        pyplot.ylabel("# of measurements")
        if idx == 0:
            pyplot.xlabel("distance in cm")
        else:
            pyplot.xlabel("angle in radians")
        pyplot.grid(True)
        # pyplot.show()
        pyplot.savefig('img/' + name + '.png',dpi=50)
        pyplot.clf()
