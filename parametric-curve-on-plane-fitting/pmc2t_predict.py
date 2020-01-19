import argparse
import csv
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc2t_predict.py makes prediction of couples of coordinates of a parametric curve on plan modeled with a pretrained multilayer perceptron with two output neurons')

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
    model = checkpoint['model'].to(device=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    print(model)

    t = torch.unsqueeze(torch.FloatTensor(t_values), dim=1).to(device=args.device)
    xy_pred = model(t)
    xy_values = xy_pred.cpu().numpy()
    csv_output_file = open(args.prediction_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(t_values)):
            writer.writerow([t_values[i], xy_values[i][0], xy_values[i][1]])

    print("#### Terminated {} ####".format(__file__));
