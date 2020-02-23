import argparse
import numpy as np
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fxy_gen.py generates a synthetic dataset file calling a two-variables real function on a rectangle')

    parser.add_argument('--dsout',
                        type=str,
                        dest='ds_output_filename',
                        required=True,
                        help='dataset output file (csv format)')

    parser.add_argument('--fxy',
                        type=str,
                        dest='func_xy_body',
                        required=True,
                        help='f(x, y) body (lamba format)')

    parser.add_argument('--rxbegin',
                        type=float,
                        dest='range_xbegin',
                        required=False,
                        default=-5.0,
                        help='begin x range (default:-5.0)')

    parser.add_argument('--rxend',
                        type=float,
                        dest='range_xend',
                        required=False,
                        default=+5.0,
                        help='end x range (default:+5.0)')

    parser.add_argument('--rybegin',
                        type=float,
                        dest='range_ybegin',
                        required=False,
                        default=-5.0,
                        help='begin y range (default:-5.0)')

    parser.add_argument('--ryend',
                        type=float,
                        dest='range_yend',
                        required=False,
                        default=+5.0,
                        help='end y range (default:+5.0)')

    parser.add_argument('--rstep',
                        type=float,
                        dest='range_step',
                        required=False,
                        default=0.01,
                        help='step range (default: 0.01)')

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    x_values = np.arange(args.range_xbegin, args.range_xend, args.range_step, dtype=float)
    y_values = np.arange(args.range_ybegin, args.range_yend, args.range_step, dtype=float)
    func_xy = eval('lambda x, y: ' + args.func_xy_body)
    csv_ds_output_file = open(args.ds_output_filename, 'w')
    with csv_ds_output_file:
        writer = csv.writer(csv_ds_output_file, delimiter=',')
        for i in range(0, x_values.size):
            for j in range(0, y_values.size):
                writer.writerow([x_values[i], y_values[j], func_xy(x_values[i], y_values[j])])

    print("#### Terminated {} ####".format(__file__));
