import argparse
import csv
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fxy_predict.py makes prediction of the values of a two-variabled real function modeled with a pretrained multilayer perceptron')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model path')

    parser.add_argument('--ds',
                        type=str,
                        dest='dataset_filename',
                        required=True,
                        help='dataset file (csv format); only x-values are used')

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

    x_values = []
    y_values = []
    with open(args.dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x_values.append(float(row[0]))
            y_values.append(float(row[1]))

    checkpoint = torch.load(args.model_path)
    model = checkpoint['model'].to(device=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    print(model)

    x = torch.FloatTensor(x_values).to(device=args.device)
    y = torch.FloatTensor(y_values).to(device=args.device)
    xy = torch.stack((x.T, y.T)).T;
    z_pred = model(xy)
    z_values = z_pred.cpu().numpy()
    csv_output_file = open(args.prediction_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(x_values)):
            writer.writerow([x_values[i], y_values[i], z_values[i][0]])

    print("#### Terminated {} ####".format(__file__));
