import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", 'hyperopt'])

import bz2
import json
import os
import pickle
import zipfile
from time import time
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt.pyll import scope

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ROOT = os.path.split(sys.argv[0])[0]


def get_path(sub_path):
    full_path = os.path.join(ROOT, sub_path)
    dirs, filename = os.path.split(full_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    return full_path


def get_causeme_data(name):
    """
    :return: (n_datasets, n_batches, n_var, T)
    """
    name = name.split('.zip')[0]  # remove .zip if present
    path = get_path(f'data/{name}.zip')
    with zipfile.ZipFile(path, "r") as f:
        # array with (n_exp, T, n_var)
        data = np.stack([np.loadtxt(f.open(name)) for name in sorted(f.namelist())])
    # (n_exp, T, n_var)
    data = torch.from_numpy(data).to(device=DEVICE)
    # standardize (n_exp, T, n_var)
    data = (data - data.mean(dim=1, keepdims=True)) / data.std(dim=1, keepdims=True)
    # to float32(n_exp, 1, n_var, T)
    return data.transpose(-1, -2).unsqueeze(dim=1).to(dtype=torch.float32)


def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def write_json(filename, data):
    with open(filename, "w") as f:
        return json.dump(data, f, indent=4)


def write_bz2(filename, data):
    with bz2.BZ2File(filename, 'w') as mybz2:
        mybz2.write(bytes(json.dumps(data, indent=4), encoding='latin1'))


def read_bz2(filename, default=None):
    if os.path.exists(filename):
        with bz2.BZ2File(filename, 'r') as f:
            return json.loads(f.read())
    return default


def write_trials(filename, _trials):
    with open(filename, "wb") as f:
        pickle.dump(_trials, f)


def read_trials(filename, default=None):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)  # load checkpoint
    return default


class TemporalBlock(nn.Module):
    """ copied from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.pad = nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, (1,)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """ copied from https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script """
    def __init__(self, num_inputs, channels, num_outputs, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else channels[i - 1]
            out_channels = channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.fc = nn.Linear(channels[-1], num_outputs)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(self.network(x).transpose(-1, -2)).transpose(-1, -2)


def receptive_field(kernel_size, layers):
    return (kernel_size - 1) * (2 ** layers - 1) + 1


class NAVARTCN(nn.Module):
    def __init__(self, num_nodes, kernel_size, n_channel_layers, channel_dim, dropout):
        """
        Neural Additive Vector AutoRegression (NAVAR) model
        Args:
            num_nodes: int
                The number of time series (N)
            kernel_size: int
                ..
            n_channel_layers: int
                ..
            channel_dim: int
                ..
            dropout: float
                Dropout probability of units in hidden layers
        """
        super(NAVARTCN, self).__init__()
        self.num_nodes = num_nodes

        channels = [channel_dim]
        for _ in range(n_channel_layers-1):
            channels.append(channels[-1] * 2)
        self.lags_k = receptive_field(kernel_size, len(channels))

        tcn_list = []
        for _ in range(num_nodes):
            tcn_list += [TemporalConvNet(1, channels, num_nodes, kernel_size, dropout)]

        self.tcn_list = nn.ModuleList(tcn_list)
        self.biases = nn.Parameter(torch.ones(num_nodes, 1) * 0.0001)

    def forward(self, x):
        # x: (bs, num_nodes, time_steps)
        # we split the input into the components
        # x: num_nodes x (bs, 1, time_steps)
        x = x.split(1, dim=1)
        # x: num_nodes x (bs, num_nodes, time_steps)
        x = [tcn(n) for n, tcn in zip(x, self.tcn_list)]

        # contributions: (bs, num_nodes, num_nodes, time_steps)
        contributions = torch.stack(x, dim=1)
        # predictions: (bs, num_nodes, time_steps)
        predictions = contributions.sum(dim=1) + self.biases
        return predictions, contributions


def train_tcn_model(data, epochs, val_proportion, learning_rate, lambda1, dropout, weight_decay,
                    kernel_size, n_channel_layers, channel_dim):
    # data: torch.Size([bs, num_nodes, T])
    bs, N, T = data.size()
    T_test = int(val_proportion * T)
    T_train = T - T_test

    model = NAVARTCN(num_nodes=N,
                     kernel_size=kernel_size,
                     n_channel_layers=n_channel_layers,
                     channel_dim=channel_dim,
                     dropout=dropout).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='mean')
    all_losses = []

    model.train()
    x, y = data[..., :T_train-1], data[..., 1:T_train]
    for _ in range(epochs):
        # predictions, contributions: (bs, num_nodes, time_steps), (bs, num_nodes, num_nodes, time_steps)
        predictions, contributions = model.forward(x)
        loss = criterion(predictions, y) + (lambda1 / N) * contributions.abs().sum(dim=(1, 2)).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        all_losses.append(loss.item())

    #plt.plot(all_losses)
    #plt.show()

    with torch.no_grad():
        # get contributions for complete dataset:
        _, all_contributions = model.forward(data)
        all_contributions = all_contributions.detach().std(dim=-1).flatten().cpu().tolist()

        # get loss of test dataset:
        loss = -1
        if T_test > 0:
            x, y = data[..., T_train:-1], data[..., T_train+1:]
            # predictions, contributions: (bs, num_nodes, 1), (bs, num_nodes, num_nodes, 1)
            predictions, contributions = model.forward(x)
            loss = (criterion(predictions, y) + (lambda1 / N) * contributions.abs().sum(dim=(1, 2)).mean()).item()

    return loss, all_contributions


counter = 0
trials = None


def eval_params(params, data, filename, max_evals, n_validation_sets):
    global counter, trials

    print(f"\t\tRun eval #{counter + 1}/{max_evals}")
    t0 = time()

    # checkpoint
    write_trials(filename, trials)

    train_data = data[counter % min(n_validation_sets, len(data))]
    counter += 1

    loss, _ = train_tcn_model(train_data, **params)

    print(f'\t\t[{round((time()-t0)/60, 1)} min]')

    return loss


def run_hyperopt(experiment, data, max_evals, n_validation_sets=1):
    global counter, trials
    counter = 0
    filename = get_path(f'results/trials/{experiment}.p')
    trials = read_trials(filename, default=Trials())

    space = {
        'epochs': 3000,
        'val_proportion': 0.30,
        'learning_rate': hp.loguniform('learning_rate', -6 * np.log(10), -2 * np.log(10)),  # between 1e-6 and 1e-2
        'weight_decay': hp.loguniform('weight_decay', -6 * np.log(10), -1 * np.log(10)),  # between 1e-6 and 1e-1
        'lambda1': hp.loguniform('lambda1', -2 * np.log(10), 0),  # between 1e-2 and 1
        'dropout': hp.quniform('dropout', 0.1, 0.25, 0.05),  # 10/15/20/25%

        # TCN pareameters
        # 2 temporal layers, resulting in a receptive field of (kernel_size-1)*((2^layers)-1)+1 = 7
        'kernel_size': 3,  # kernel size of 3
        'n_channel_layers': 2,
        # upsampling to 8, 16 channels, then to num_nodes with a linear layer
        'channel_dim': scope.int(hp.quniform('channel_dim', 4, 8, q=1))
    }

    best = fmin(
        fn=lambda p: eval_params(p, data, filename, max_evals, n_validation_sets),
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals,
        show_progressbar=False)

    # checkpoint
    write_trials(filename, trials)

    return space_eval(space, best)


def run_experiment(experiment, max_evals, n_validation_sets):
    print(f'Run experiment.py: {experiment}')

    print(f'\tLoad data..')
    data = get_causeme_data(experiment)  # torch.Size([200, 1, 3, 300])

    print(f"\tRun hyperopt: max_evals={max_evals}, n_validation_sets={n_validation_sets}")
    best_params = run_hyperopt(experiment, data, max_evals=max_evals, n_validation_sets=n_validation_sets)
    print('\tBest parameters found:', best_params)
    write_json(get_path(f'results/hyper_params/{experiment}.json'), best_params)

    print(f'\tSet val_proportion to 0..')
    best_params['val_proportion'] = 0

    print(f"\tTrain {len(data)} datasets with best parameters..")
    filename = get_path(f'results/causeme/{experiment}.json.bz2')
    result = {
        "method_sha": "e0ff32f63eca4587b49a644db871b9a3",
        "parameter_values": ", ".join([f"{k}={v}" for k, v in best_params.items()]),
        "model": experiment.split('_')[0],
        "experiment.py": experiment,
        "scores": []
    }
    result = read_bz2(filename, default=result)

    for i in range(len(result['scores']), len(data)):  # torch.Size([200, 1, 3, 300])
        print(f"\t\tRun dataset #{i+1}/{len(data)}")
        t0 = time()

        _, contributions = train_tcn_model(data[i], **best_params)
        result["scores"].append(contributions)
        write_bz2(filename, result)

        print(f'\t\t[{round((time()-t0)/60, 1)} min]')

    print(f'\tSuccessfully ran experiment.py: {experiment}')


for experiment_name in sys.argv[1:]:
    run_experiment(experiment_name, max_evals=1, n_validation_sets=5)
