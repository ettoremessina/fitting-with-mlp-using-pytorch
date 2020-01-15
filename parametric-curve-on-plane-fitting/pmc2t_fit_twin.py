import argparse
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as to
import torch.utils.data as tud

class ParamCurveOnPlanTrainData(tud.Dataset):
    t_train = []
    x_train = []
    y_train = []

    def __init__(self, train_dataset_filename):
            with open(args.train_dataset_filename) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    self.t_train.append(float(row[0]))
                    self.x_train.append(float(row[1]))
                    self.y_train.append(float(row[2]))

    def __getitem__(self, index):
        return self.t_train[index], self.x_train[index], self.y_train[index]

    def __len__(self):
        return len(self.t_train)

def build_model():
    layers = []
    layers.append(nn.Linear(1, args.hidden_layers_layout[0]))
    num_of_layers = len(args.hidden_layers_layout)
    for h in range(num_of_layers - 1):
        af = args.activation_functions[h]
        if af.lower() != 'none':
            layers.append(build_activation_function(af))
        layers.append(nn.Linear(args.hidden_layers_layout[h], args.hidden_layers_layout[h+1]))
    af = args.activation_functions[num_of_layers-1]
    if af.lower() != 'none':
        layers.append(build_activation_function(af))
    layers.append(nn.Linear(args.hidden_layers_layout[num_of_layers-1], 1))
    return nn.Sequential(*layers)

def build_activation_function(af):
    exp_af = 'lambda _ : nn.' + af
    return eval(exp_af)(None)

def build_optimizer(model):
    opt_init = args.optimizer
    bracket_pos = opt_init.find('(')
    if bracket_pos == -1:
        raise Exception('Wrong optimizer syntax')
    opt_init_tail = opt_init[bracket_pos+1:].strip()
    if opt_init_tail != ')':
        opt_init_tail = ', ' + opt_init_tail
    if len(args.learning_rate.strip()) > 0:
        opt_init_tail = ', lr=' + args.learning_rate + opt_init_tail
    opt_init = opt_init[:bracket_pos+1] + 'm.parameters()' + opt_init_tail
    exp_po = 'lambda m : to.' + opt_init
    optimizer = eval(exp_po)(model)
    return optimizer

def build_loss():
    exp_loss = 'lambda _ : nn.' + args.loss
    return eval(exp_loss)(None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pmc2t_fit_twin.py fits a parametric curve on plan dataset using two configurable twin multilayer perceptrons each of them with only one output neuron')

    parser.add_argument('--trainds',
                        type=str,
                        dest='train_dataset_filename',
                        required=True,
                        help='train dataset file (csv format)')

    parser.add_argument('--modelout',
                        type=str,
                        dest='model_path',
                        required=True,
                        help='output model path')

    parser.add_argument('--epochs',
                        type=int,
                        dest='epochs',
                        required=False,
                        default=500,
                        help='number of epochs')

    parser.add_argument('--batch_size',
                        type=int,
                        dest='batch_size',
                        required=False,
                        default=50,
                        help='batch size')

    parser.add_argument('--learning_rate',
                        type=str,
                        dest='learning_rate',
                        required=False,
                        default='',
                        help='learning rate')

    parser.add_argument('--hlayers',
                        type=int,
                        nargs = '+',
                        dest='hidden_layers_layout',
                        required=False,
                        default=[100],
                        help='number of neurons for each hidden layers')

    parser.add_argument('--hactivations',
                        type=str,
                        nargs = '+',
                        dest='activation_functions',
                        required=False,
                        default=['ReLU()'],
                        help='activation functions between layers')

    parser.add_argument('--optimizer',
                        type=str,
                        dest='optimizer',
                        required=False,
                        default='Adam()',
                        help='optimizer algorithm object')

    parser.add_argument('--loss',
                        type=str,
                        dest='loss',
                        required=False,
                        default='MSELoss()',
                        help='loss function name')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        required=False,
                        default='cpu',
                        help='target device')

    args = parser.parse_args()

    if len(args.hidden_layers_layout) != len(args.activation_functions):
        raise Exception('Number of hidden layers and number of activation functions must be equals')

    print("#### Started {} {} ####".format(__file__, args));

    dataset = ParamCurveOnPlanTrainData(args.train_dataset_filename)
    dataloader = tud.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    model_x = build_model().to(device=args.device)
    model_y = build_model().to(device=args.device)
    print(model_x)

    optimizer_x = build_optimizer(model_x)
    loss_func_x = build_loss().to(device=args.device)

    optimizer_y = build_optimizer(model_y)
    loss_func_y = build_loss().to(device=args.device)

    start_time = time.time()
    for epoch in range(args.epochs):
        print('MLP #1, Epoch {}/{}'.format(epoch+1, args.epochs))

        print('[', end = '')
        for batch_num, batch_data in enumerate(dataloader):
            print('=', end = '')
            t_train = torch.unsqueeze(torch.FloatTensor(batch_data[0].float()), dim=1).to(device=args.device)
            x_train = torch.unsqueeze(torch.FloatTensor(batch_data[1].float()), dim=1).to(device=args.device);
            prediction_x = model_x(t_train)
            loss_x = loss_func_x(prediction_x, x_train)
            optimizer_x.zero_grad()
            loss_x.backward()
            optimizer_x.step()
        print (']/{} - loss: {}'.format(batch_num, loss_x.item()))

    for epoch in range(args.epochs):
        print('MLP #2, Epoch {}/{}'.format(epoch+1, args.epochs))

        print('[', end = '')
        for batch_num, batch_data in enumerate(dataloader):
            print('=', end = '')
            t_train = torch.unsqueeze(torch.FloatTensor(batch_data[0].float()), dim=1).to(device=args.device)
            y_train = torch.unsqueeze(torch.FloatTensor(batch_data[2].float()), dim=1).to(device=args.device);
            prediction_y = model_y(t_train)
            loss_y = loss_func_y(prediction_y, y_train)
            optimizer_y.zero_grad()
            loss_y.backward()
            optimizer_y.step()
        print (']/{} - loss: {}'.format(batch_num, loss_y.item()))

    elapsed_time = time.time() - start_time
    print ("Training time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    checkpoint = {
        'model_x': model_x,
        'state_dict_x': model_x.state_dict(),
        'optimizer_x' : optimizer_x.state_dict(),
        'model_y': model_y,
        'state_dict_y': model_y.state_dict(),
        'optimizer_y' : optimizer_y.state_dict(),
         }
    torch.save(checkpoint, args.model_path)

    print("#### Terminated {} ####".format(__file__));
