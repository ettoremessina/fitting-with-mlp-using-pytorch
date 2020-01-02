import argparse
import numpy as np
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fx_gen.py generates a synthetic dataset file calling a one-variable real function in an interval')

    parser.add_argument('--dsout',
                        type=str,
                        dest='ds_output_filename',
                        required=True,
                        help='dataset output file (csv format)')

    parser.add_argument('--fx',
                        type=str,
                        dest='func_x_body',
                        required=True,
                        help='f(x) body (lamba format)')

    parser.add_argument('--rbegin',
                        type=float,
                        dest='range_begin',
                        required=False,
                        default=-5.0,
                        help='begin range (default:-5.0)')

    parser.add_argument('--rend',
                        type=float,
                        dest='range_end',
                        required=False,
                        default=+5.0,
                        help='end range (default:+5.0)')

    parser.add_argument('--rstep',
                        type=float,
                        dest='range_step',
                        required=False,
                        default=0.01,
                        help='step range (default: 0.01)')

    args = parser.parse_args()

    x_values = np.arange(args.range_begin, args.range_end, args.range_step, dtype=float)
    func_x = eval('lambda x: ' + args.func_x_body)
    csv_ds_output_file = open(args.ds_output_filename, 'w')
    with csv_ds_output_file:
        writer = csv.writer(csv_ds_output_file, delimiter=',')
        for i in range(0, x_values.size):
            writer.writerow([x_values[i], func_x(x_values[i])])
