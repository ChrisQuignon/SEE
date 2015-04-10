from matplotlib import pyplot
import csv
import glob

# files = glob.glob('*.csv')
files = glob.glob('data_ahead.csv')

for file in files:
    data = {}
    with open(file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        data =  [x for x in reader]

    x =  [float (d['x']) for d in data]
    y =  [float(d['y']) for d in data]

    mean_x = sum(x) / float(len(x))
    mean_y = sum(y) / float(len(y))

    pyplot.scatter(x, y, c = 'black', marker = 'x', zorder = 4)

    pyplot.scatter(mean_x, mean_y, color = 'g', zorder = 1, s = 40, alpha = 0.5)
    pyplot.scatter(min(x), min(y), color = 'b', zorder = 3, s = 40, alpha = 0.5)
    pyplot.scatter(max(x), max(y), color = 'r', zorder = 2, s = 40, alpha = 0.5)

    pyplot.show()
