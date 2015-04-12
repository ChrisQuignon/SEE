from matplotlib import pyplot
import csv
import glob

#NEEDS scipy 0.12 (anaconda)
# from scipy.stats import chi2_contingency

files = glob.glob('*.csv')
#files = glob.glob('data_ahead.csv')

xs = []
ys = []
mx = []
my = []

for file in files:
    data = {}


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
#     pyplot.scatter(xs[i][:20], ys[i][:20], c = 'black', marker = 'x', zorder = 4)
#
#     #pyplot.scatter(mean_x, mean_y, color = 'g', zorder = 1, s = 40, alpha = 0.5)
#
#     #PLOT FAILURE
#     if len(xs[i]) > 20:
#         pyplot.scatter(x[20:], y[20:], c = 'red', marker = 'x', zorder = 4)
#
#     pyplot.axes().set_aspect('equal', 'datalim')
#     pyplot.grid(True)
#     pyplot.locator_params(nbins=10)
#     pyplot.savefig('img/' + files[i][:-4]+'.png')
#     pyplot.clf()

#EXCLUDE BAD POINT
for i in range(len(xs)):
    xs[i] = xs[i][:20]
    ys[i] = ys[i][:20]


# #PLOT RUNS OVERLAYED
# for i in range(len(xs)):
#     x_norm = [x - mx[i] for x in xs[i]]
#     y_norm = [y - my[i] for y in ys[i]]
#
#     pyplot.scatter(x_norm, y_norm, c = pyplot.cm.spectral(i/float(len(xs))), s = 40.0, alpha = 0.8)
#
# pyplot.axes().set_aspect('equal', 'datalim')
# pyplot.grid(True)
# pyplot.locator_params(nbins=10)
#
# pyplot.savefig('img/overlay.png')
#
# pyplot.show()
# pyplot.clf()

# #MAKE ALL POINT POSITIVE
# for i in range(len(xs)):
#     xs[i] = [x + 20 for x in xs[i]]
#     ys[i] = [y + 20 for y in ys[i]]
#
# #CHI2_Contingency
#NEEDS scipy 0.12 (anaconda)
# for i in range(len(xs)):
#     g, p, dof, expctd = chi2_contingency([xs[i],[1]*len(xs[i])], lambda_="log-likelihood")
#     # print g, p, dof, expctd
#     print expctd
#     # pyplot.scatter(expctd[0], expctd[1])
#     # pyplot.show()
