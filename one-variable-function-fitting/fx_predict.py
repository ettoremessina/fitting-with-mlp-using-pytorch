import argparse
import csv
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fx_predict.py makes a prediction on a test dataset of a one-variable function modeled with a pretrained multilayer perceptron network')

    parser.add_argument('--model',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='model path')

    parser.add_argument('--testds',
                        type=str,
                        dest='test_dataset_filename',
                        required=True,
                        help='test dataset file (csv format)')

    parser.add_argument('--predictedout',
                        type=str,
                        dest='predicted_data_filename',
                        required=True,
                        help='predicted data file (csv format)')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        required=False,
                        default='cpu',
                        help='target device')

    args = parser.parse_args()

    x_test = []
    y_test = []
    with open(args.test_dataset_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x_test.append(float(row[0]))
            y_test.append(float(row[1]))

    checkpoint = torch.load(args.model_path)
    model = checkpoint['model'].to(device=args.device)
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    print(model)

    x = torch.unsqueeze(torch.FloatTensor(x_test), dim=1).to(device=args.device)
    y_pred = model(x)
    y = y_pred.cpu().numpy()
    csv_output_file = open(args.predicted_data_filename, 'w')
    with csv_output_file:
        writer = csv.writer(csv_output_file, delimiter=',')
        for i in range(0, len(x_test)):
            writer.writerow([x_test[i], y[i][0]])
