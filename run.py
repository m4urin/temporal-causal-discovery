import argparse

from hyperopt.pyll import scope

from src.utils.argparse_helper import valid_number
from hyperopt import hp

from src.utils.hyperopt_helper import loguniform_10


default_space = {
    'max_evals': 100,
    'hidden_dim': 64,
    'dropout': 0.0,
    'kernel_size': 5,
    'n_blocks': 1,
    'n_layers_per_block': 2,
    'weight_sharing': False,
    'recurrent': False,
    'aleatoric': False,
    'epistemic': False,
    'uncertainty_contributions': False,
    'n_heads': 1,
    'softmax_method': 'softmax',
    'lr': 1e-3,
    'epochs': 2000,
    'weight_decay': 1e-6,
    'test_size': 0.0
}

hyperopt_space = {
    'hidden_dim': hp.choice('hidden_dim', [8, 16, 32, 64, 128]),
    'dropout': hp.uniform('dropout', 0.0, 1.0),
    'kernel_size': hp.quniform('kernel_size', 2, 100, 1),
    'n_blocks': hp.quniform('n_blocks', 1, 10, 1),
    'n_layers_per_block': hp.quniform('n_layers_per_block', 1, 10, 1),
    'n_heads': hp.quniform('n_heads', 1, 10, 1),
    'softmax_method': hp.choice('softmax_method', ['softmax', 'softmax-1', 'normalized-sigmoid', 'gumbel-softmax', 'sparsemax']),
    'lr': hp.loguniform('lr', -5, -2),
    'weight_decay': hp.loguniform('weight_decay', -6, -2)
}


def parse_args():
    parser = argparse.ArgumentParser()

    # Procedure for hyperopt
    parser.add_argument("--hyperopt", action='store_true', default=False)
    parser.add_argument("--max_evals", type=valid_number(int, min_inclusive=1), default=None)
    # Dataset
    parser.add_argument("--dataset", type=str, required=True)

    # Model
    parser.add_argument("--model", type=str, choices=['TAMCaD', 'NAVAR'], required=True)

    parser.add_argument("--hidden_dim", type=valid_number(int, min_inclusive=8),
                        default=scope.pow(2, scope.int(hp.quniform('hidden_dim', 3, 7, 1))))  # [8, 16, 32, 64, 128]
    parser.add_argument("--dropout", type=valid_number(float, min_inclusive=0.0, max_exclusive=1.0), default=0.0)

    parser.add_argument("--receptive_field", type=valid_number(int, min_inclusive=2), default=5)
    parser.add_argument("--kernel_size", type=valid_number(int, min_inclusive=2), default=None)
    parser.add_argument("--n_blocks", type=valid_number(int, min_inclusive=1), default=None)
    parser.add_argument("--n_layers_per_block", type=valid_number(int, min_inclusive=1), default=None)

    parser.add_argument("--weight_sharing", action='store_true', default=hp.choice('weight_sharing', [True, False]))
    parser.add_argument("--recurrent", action='store_true', default=False)

    parser.add_argument("--aleatoric", action='store_true', default=False)
    parser.add_argument("--epistemic", action='store_true', default=False)
    parser.add_argument("--uncertainty_contributions", action='store_true', default=None)

    # TAMCaD specific
    parser.add_argument("--n_heads", type=valid_number(int, min_inclusive=1),
                        default=None)

    methods = ['softmax', 'softmax-1', 'normalized-sigmoid', 'gumbel-softmax', 'sparsemax']
    parser.add_argument("--softmax_method", type=str, choices=methods, default=None)

    # Training
    parser.add_argument("--lr", type=valid_number(float, min_exclusive=0.0),
                        default=loguniform_10('lr', -5, -2))
    parser.add_argument("--epochs", type=valid_number(int, min_inclusive=0), default=3000)
    parser.add_argument("--weight_decay", type=valid_number(float, min_inclusive=0.0),
                        default=loguniform_10('weight_decay', -6, -2))
    parser.add_argument("--test_size", type=valid_number(float, min_inclusive=0.0, max_exclusive=1.0), default=0.0)

    args = parser.parse_args()

    if args.model == "NAVAR" and args.n_heads is not None:
        parser.error("--n_heads must only be provided when chosen TAMCaD model.")
    if args.model == "NAVAR" and args.softmax_method is not None:
        parser.error("--softmax_method must only be provided when chosen TAMCaD model.")
    if args.model == "TAMCaD" and args.hidden_dim % args.n_heads != 0:
        parser.error("--hidden_dim should be divisible by --n_heads.")

    if args.hyperopt and args.max_evals is None:
        parser.error("--max_evals must be provided when running --hyperopt.")

    return args


def main():
    # Parse the arguments
    args = parse_args()

    # Determine which parameter space to use based on hyperopt flag
    if args.hyperopt:
        param_space = hyperopt_space
    else:
        param_space = default_space

    # Overwrite parameters with user-specified values
    for key, value in vars(args).items():
        if value is not None:
            param_space[key] = value

    # Call train_model() with the constructed parameter space
    train_model(param_space)

if __name__ == "__main__":
    args_ = parse_args()
    print(args_)
