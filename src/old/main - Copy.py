import os
import sys
import subprocess

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('hyperopt')

import numpy as np
from hyperopt import hp, fmin, tpe, space_eval, Trials
from hyperopt.pyll import scope
from tqdm import trange

from src.data import get_causeme_data, set_root, read_trials, write_trials, write_bz2, read_bz2, get_path, write_json
from src.train_models import train_tcn_model


set_root(os.path.split(sys.argv[0])[0])
experiments_todo = sys.argv[1:]

counter = 0
trials = Trials()


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

    def eval_params(params):
        global counter, trials

        # checkpoint
        write_trials(filename, trials)
        write_json(get_path('results/progress.json'), {'experiment.py': experiment})

        train_data = data[counter % min(n_validation_sets, len(data))]
        counter += 1

        loss, _ = train_tcn_model(train_data, **params)
        return loss

    best = fmin(
        fn=eval_params,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=max_evals)

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

    for i in trange(len(result['scores']), len(data), desc='run all datasets'):  # torch.Size([200, 1, 3, 300])
        _, contributions = train_tcn_model(data[i], **best_params)
        result["scores"].append(contributions)
        write_bz2(filename, result)

    print(f'\tSuccessfully ran experiment.py: {experiment}')


if __name__ == '__main__':
    try:
        for experiment_name in experiments_todo:
            run_experiment(experiment_name, max_evals=200, n_validation_sets=5)
    except Exception as e:
        write_json(get_path('results/progress.json'), {'error': str(e)})
        raise e
