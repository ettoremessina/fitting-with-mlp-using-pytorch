import argparse
import numpy as np
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc2t_gen.py generates a synthetic dataset file that contains the points of a parametric curve on plan calling a couple of one-variable real functions in an interval')

    parser.add_argument('--dsout',
                        type=str,
                        dest='ds_output_filename',
                        required=True,
                        help='dataset output file (csv format)')

    parser.add_argument('--xt',
                        type=str,
                        dest='funcx_t_body',
                        required=True,
                        help='x=x(t) body (lamba format)')

    parser.add_argument('--yt',
                        type=str,
                        dest='funcy_t_body',
                        required=True,
                        help='y=y(t) body (lamba format)')


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

    print("#### Started {} {} ####".format(__file__, args));

    t_values = np.arange(args.range_begin, args.range_end, args.range_step, dtype=float)
    funcx_t = eval('lambda t: ' + args.funcx_t_body)
    funcy_t = eval('lambda t: ' + args.funcy_t_body)
    csv_ds_output_file = open(args.ds_output_filename, 'w')
    with csv_ds_output_file:
        writer = csv.writer(csv_ds_output_file, delimiter=',')
        for i in range(0, t_values.size):
            writer.writerow([t_values[i], funcx_t(t_values[i]), funcy_t(t_values[i])])

    print("#### Terminated {} ####".format(__file__));
