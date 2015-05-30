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
right = glob.glob('recordings/right_exp/*/*')

xs = []
ys = []
rolls = []
cams = []

files=['left', 'ahead', 'right']

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

            # FILTER OUTLIER
            data = [x for x in data if x['y'] < 680]

            final = data[0]


            #FILTER OUTLIER
            if final['y'] > -345:
                set.append(final)

    # set = sorted(set, key=lambda k: k['time'])
    sets.append(set)


for set in sets:
    #FILTER OUTLIER
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
    pyplot.savefig('img/' + files[i]+'_s.png')
    # pyplot.show()
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
    ax = pyplot.subplot(111, polar=True)
    ax.set_title('Polar Coordinates ' + files[i])
    ax.grid(True)
    # ax.locator_params(nbins=10)

    for j in range(len(r)):
        ax.plot(p[j], r[j], color = cams[i][j], marker = (3, 0, rolls[i][j]*57.3))

    # ax.ylabel('distance in mm')
    # ax.xlabel('angle in radians')
    pyplot.savefig('img/' + files[i]+ '_pc_s.png')
    # pyplot.show()
    pyplot.clf()

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
pyplot.savefig('img/BoxplotDistance_s.png',dpi=50)
# pyplot.show()
pyplot.clf()

pyplot.title('Normalized Angle')
pyplot.grid(True)
pyplot.locator_params(nbins=10)
pyplot.boxplot(p_norm, vert=False, sym='')
pyplot.yticks([1, 2, 3], files, rotation='vertical')
pyplot.xlabel('angle in radians')
pyplot.savefig('img/BoxplotAngleNorm_s.png',dpi=50)
# pyplot.show()
pyplot.clf()



##HISTOGRAMS
setNames = ['Distances', 'Angles']#'Xs', 'Ys',
for idx, set in enumerate([rs, ps]):#xs, ys,

    for jdx, vals in enumerate(set):
        name = setNames[idx] + '_' + files[jdx]

        #CALCULATING MU AND SIGMA
        mu = np.mean(vals)
        sigma = np.std(vals)


        # HISTOGRAM
        # ignore handcrafted bins
        bins = 5
        observed, bins_m, patches_m = pyplot.hist(vals, bins=bins, color = 'b', alpha = 0.6)

        #GAUSSIAN
        x = np.arange(bins_m[0], bins_m[-1], 0.001)
        y = mlab.normpdf( x, mu, sigma)
        pyplot.plot(x, y, 'r--', linewidth=1)


        #Measurements expected
        expected = []

        for i in range(len(bins_m)-1):
            #integration of the gaussian
            prob = integrate.quad(norm(mu, sigma).pdf, bins_m[i], bins_m[i+1])[0]
            expected.append(prob*float(len(vals)))

        #Plot expected
        ms = []
        widths = []

        for i in range(len(bins_m)-1):
            ms.append((bins_m[i]+bins_m[i+1])/2)
            widths.append((bins_m[i+1]-bins_m[i]))

        pyplot.bar(ms, expected, widths, align='center', color = 'g', alpha = 0.6)

        #CHI SQUARED TEST
        chisq, p = chisquare(np.asarray(observed), np.asarray(expected))
        # chisq, p = chisquare(bins, y*(len(vals)))

        print 'Histogram ' + name + ':'
        print 'mu: ', round(mu, 2)
        print 'sigma: ', round(sigma,2)
        print 'chisq:', round (chisq, 2)
        print 'p: ', round(p*100, 2), '%'
        print '--'
        print 'expected: ', map(lambda x : round(x, 2), expected)
        print 'observed: ',observed
        print 'bins: ', bins
        # print 'Standard error of mean: ' + str(mean_error)
        # print 'Standard erro of deviation: ' + str(error_std_dev)
        print ''

        s = 'Histogram: ' + files[jdx]
        pyplot.title(s)

        pyplot.ylabel("# of measurements")
        if idx == 0:
            pyplot.xlabel("distance in cm")
        else:
            pyplot.xlabel("angle in radians")
        pyplot.grid(True)
        # pyplot.show()
        pyplot.savefig('img/' + name + '_s.png',dpi=50)
        pyplot.clf()
