import argparse
import csv
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fx_plot.py shows two overlapped x/y scatter graphs: the blue one is the train dataset, the red one is the predicted one')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='train dataset file (csv format)')

    parser.add_argument('--predicted',
                        type=str,
                        dest='predicted_data_filename',
                        required=True,
                        help='predicted data file (csv format)')

    parser.add_argument('--savefig',
                        type=str,
                        dest='save_figure_filename',
                        required=False,
                        default='',
                        help='if present, the chart is saved on a file instead to be shown on screen')

    args = parser.parse_args()

    with open(args.train_dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            plt.scatter(float(row[0]), float(row[1]), color='blue', s=1, marker='.')

    with open(args.predicted_data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            plt.scatter(float(row[0]), float(row[1]), color='red', s=2, marker='.')

    if args.save_figure_filename:
        plt.savefig(args.save_figure_filename)
    else:
        plt.show()
