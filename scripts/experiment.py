import os

import numpy as np
import torch
from hyperopt import hp, fmin, tpe, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import AdamW
from tqdm import trange

from definitions import DEVICE, RESULTS_DIR, SETTINGS_DIR
from src.data import get_causeme_data
from src.models.navar_tcn import NAVAR_TCN
from src.utils import pretty_number, ProgressWriter, load_json_file, trange_print, save_bz2_file




def run_hyperopt(data, max_evals, n_validation_sets, overwrite_hyper_params=None):
    print(f"Run hyperopt: max_evals={max_evals}, n_validation_sets={n_validation_sets}")

    progress_writer = ProgressWriter(max_evals, 'Eval')

    space = {
        'epochs': 6000,
        'val_proportion': 0.30,
        'learning_rate': hp.loguniform('learning_rate', -6 * np.log(10), -2 * np.log(10)),  # between 1e-6 and 1e-2
        'weight_decay': hp.loguniform('weight_decay', -6 * np.log(10), -1 * np.log(10)),  # between 1e-6 and 1e-1
        'lambda1': hp.loguniform('lambda1', -1 * np.log(10), 0),  # between 1e-1 and 1
        'dropout': hp.quniform('dropout', 0.05, 0.25, 0.05),  # 5/10/15/20/25%

        # TCN pareameters
        # 2 temporal layers, resulting in a receptive field of (kernel_size-1)*((2^layers)-1)+1 = 7
        'kernel_size': 3,  # kernel size of 3
        'n_layers': 1,
        'hidden_dim': scope.int(hp.quniform('channel_dim', 2, 6, q=1))  # 2^hidden_dim
    }
    if overwrite_hyper_params is not None:
        for k, v in overwrite_hyper_params.items():
            space[k] = v

    def eval_params(params):
        train_data = data[progress_writer.counter % min(n_validation_sets, len(data))]
        loss, epochs, _ = train_tcn_model(train_data, **params)
        progress_writer.update()
        return {'loss': loss, 'status': STATUS_OK, 'epochs': epochs}

    trials = Trials()
    best = fmin(
        fn=eval_params,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        verbose=False,
        show_progressbar=False,
        trials=trials)

    best_params = space_eval(space, best)
    best_params['epochs'] = trials.results[np.argmin([r['loss'] for r in trials.results])]['epochs']

    print(f'Best parameters found:', best_params)

    return best_params


def run_experiment(experiment):
    print(f'Run experiment: {experiment}')
    print(f"Results will be written to '{RESULTS_DIR}'")

    print(f'Load CauseMe data..')
    data = get_causeme_data(experiment)  # torch.Size([200, 1, 3, 300])

    settings = load_json_file(os.path.join(SETTINGS_DIR, 'experiments.json'))[experiment]

    if settings['max_evals'] < 1:
        print(f"Use default parameters..")
        best_params = load_json_file(os.path.join(SETTINGS_DIR, 'default_parameters.json'))
    else:
        best_params = run_hyperopt(data=data, **settings)

    print(f"Train {len(data)} datasets with best parameters..")
    result = {
        "method_sha": "e0ff32f63eca4587b49a644db871b9a3",
        "model": experiment.split('_')[0],
        "experiment": experiment,
        "scores": []
    }

    best_params['val_proportion'] = 0
    for i in trange_print(len(data), desc='run datasets'):  # torch.Size([200, 1, 3, 300])
        _, _, contributions = train_tcn_model(data[i], **best_params)
        result["scores"].append(contributions)

    best_params['hidden_dim'] = 2 ** best_params['hidden_dim']
    del best_params['val_proportion']
    result["parameter_values"] = ", ".join([f"{k}={pretty_number(v)}" for k, v in best_params.items()])

    save_bz2_file(os.path.join(RESULTS_DIR, f'{experiment}.json.bz2'), result)

    print(f'Successfully ran experiment: {experiment}')
