import argparse
import csv
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc2t_predict_twin.py makes prediction of couples of coordinates of a parametric curve on plan modeled with two pretrained twin multilayer perceptrons each of them with only one output neuron')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model file')

    parser.add_argument('--ds',
                        type=str,
                        dest='dataset_filename',
                        required=True,
                        help='dataset file (csv format); only t-values are used')

    parser.add_argument('--predictionout',
                        type=str,
                        dest='prediction_data_filename',
                        required=True,
                        help='prediction data file (csv format)')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        required=False,
                        default='cpu',
                        help='target device')

    args = parser.parse_args()

    print("#### Started {} {} ####".format(__file__, args));

    t_values = []
    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            t_values.append(float(row[0]))

    checkpoint = torch.load(args.model_path)

    model_x = checkpoint['model_x'].to(device=args.device)
    model_x.load_state_dict(checkpoint['state_dict_x'])
    for parameter_x in model_x.parameters():
        parameter_x.requires_grad = False

    model_y = checkpoint['model_y'].to(device=args.device)
    model_y.load_state_dict(checkpoint['state_dict_y'])
    for parameter_y in model_y.parameters():
        parameter_y.requires_grad = False

    model_x.eval()
    print(model_x)
    model_y.eval()
    print(model_y)

    t = torch.unsqueeze(torch.FloatTensor(t_values), dim=1).to(device=args.device)
    x_pred = model_x(t)
    y_pred = model_y(t)
    x_values = x_pred.cpu().numpy()
    y_values = y_pred.cpu().numpy()
    csv_output_file = open(args.prediction_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(t_values)):
            writer.writerow([t_values[i], x_values[i][0], y_values[i][0]])

    print("#### Terminated {} ####".format(__file__));
