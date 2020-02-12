import argparse
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc3t_plot.py shows two overlapped x/y scatter graphs: the blue one is the dataset, the red one is the prediction')

    parser.add_argument('--ds',
                        type=str,
                        dest='dataset_filename',
                        required=True,
                        help='dataset file (csv format)')

    parser.add_argument('--prediction',
                        type=str,
                        dest='prediction_data_filename',
                        required=True,
                        help='prediction data file (csv format)')

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the chart is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ax.scatter(float(row[1]), float(row[2]), float(row[3]), color='blue', s=1, marker='.')

    with open(args.prediction_data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            ax.scatter(float(row[1]), float(row[2]),  float(row[3]), color='red', s=2, marker='.')

    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()

    print("#### Terminated {} ####".format(__file__));
