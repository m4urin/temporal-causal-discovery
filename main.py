import argparse
from pprint import pprint

from hyperopt import hp

from src.run import run
from src.utils import hp_pow, generate_architecture_options, valid_number, hp_loguniform_10, receptive_field

default_space = {
    'data_dir': None,
    'dataset_name': None,
    'model': None,
    'max_evals': 0,
    'subset_evals': 1,
    'test_size': 0.2,
    'lr': 1e-4,
    'epochs': 3000,
    'weight_decay': 1e-5,
    'hidden_dim': 64,
    'dropout': 0.2,
    'lambda1': 0.2,
    'n_heads': 4,
    'softmax_method': 'softmax',
    'recurrent': False,
    'aleatoric': False,
    'epistemic': False,
    'weight_sharing': False
}


hyperopt_space = {
    'hidden_dim': hp_pow('hidden_dim', 3, 7, base=2),  # 2^4, .. 2^7
    'dropout': hp.uniform('dropout', 0.0, 0.3),
    'n_heads': hp_pow('n_heads', 1, 3, base=2),  # 2^1, .. 2^3
    'lr': hp_loguniform_10('lr', -6, -2),
    'weight_decay': hp_loguniform_10('weight_decay', -6, -2),
    'lambda1': hp_loguniform_10('lambda1', -3, 0),
    'start_coeff': hp.uniform('start_coeff', -4, 0),
    'delta_coeff': hp.uniform('delta_coeff', 0, 3)
}


def parse_args():
    parser = argparse.ArgumentParser()
    # Config
    parser.add_argument("model", type=str, choices=['TAMCaD', 'NAVAR'],
                        help="Use the TAMCaD or NAVAR model.")
    parser.add_argument("--causeme", type=str, default=None)
    parser.add_argument("--synthetic", type=str, default=None)

    # Procedure for hyperopt
    parser.add_argument("--hyperopt", action='store_true', default=False)
    parser.add_argument("--max_evals", type=valid_number(int, min_incl=1), default=None)
    parser.add_argument("--subset_evals", type=valid_number(int, min_incl=1), default=None)

    # Model
    parser.add_argument("--hidden_dim", type=valid_number(int, min_incl=8), default=None)
    parser.add_argument("--dropout", type=valid_number(float, min_incl=0.0, max_excl=1.0), default=None)
    parser.add_argument("--lambda1", type=valid_number(float, min_incl=0.0, max_incl=1.0), default=None)

    parser.add_argument("--max_lags", type=valid_number(int, min_incl=2), default=5)
    parser.add_argument("--kernel_size", type=valid_number(int, min_incl=2), default=None)
    parser.add_argument("--n_blocks", type=valid_number(int, min_incl=1), default=None)
    parser.add_argument("--n_layers_per_block", type=valid_number(int, min_incl=1), default=None)

    parser.add_argument("--weight_sharing", action='store_true', default=None)
    parser.add_argument("--recurrent", action='store_true', default=None)

    parser.add_argument("--aleatoric", action='store_true', default=None)
    parser.add_argument("--epistemic", action='store_true', default=None)

    # TAMCaD specific
    parser.add_argument("--n_heads", type=valid_number(int, min_incl=1), default=None)
    methods = ['softmax', 'softmax-1', 'normalized-sigmoid', 'gumbel-softmax', 'sparsemax', 'gumbel-softmax-1']
    parser.add_argument("--softmax_method", type=str, choices=methods, default=None)
    parser.add_argument("--start_coeff", type=valid_number(float), default=None)
    parser.add_argument("--delta_coeff", type=valid_number(float), default=None)

    # Training
    parser.add_argument("--lr", type=valid_number(float, min_excl=0.0), default=None)
    parser.add_argument("--epochs", type=valid_number(int, min_incl=0), default=None)
    parser.add_argument("--weight_decay", type=valid_number(float, min_incl=0.0), default=None)
    parser.add_argument("--test_size", type=valid_number(float, min_incl=0.0, max_excl=1.0), default=None)

    args = parser.parse_args()

    if args.model == "NAVAR" and args.n_heads is not None:
        parser.error("--n_heads must only be provided when chosen TAMCaD model.")
    if args.model == "NAVAR" and args.softmax_method is not None:
        parser.error("--softmax_method must only be provided when chosen TAMCaD model.")
    if args.model == "TAMCaD" and args.hidden_dim and args.n_heads and args.hidden_dim % args.n_heads != 0:
        parser.error("--hidden_dim should be divisible by --n_heads.")

    choice_causeme, choice_syn = args.causeme is not None, args.synthetic is not None
    if choice_causeme and choice_syn:
        parser.error("use one of the data types, not both: --causeme or --synthetic.")
    if not choice_causeme and not choice_syn:
        parser.error("use one of the data types: --causeme or --synthetic.")
    if args.hyperopt and args.causeme is not None and args.subset_evals is None:
        parser.error("--subset_evals must be provided when running --hyperopt on --causeme data.")
    if args.hyperopt and args.max_evals is None:
        parser.error("--max_evals must be provided when running --hyperopt.")

    return args


def get_param_space():
    # Parse the arguments
    args = parse_args()

    param_space = {k: v for k, v in default_space.items()}

    if args.hyperopt:
        for key, value in hyperopt_space.items():
            param_space[key] = value

    print('Finding architecture options..')
    architecture_options = generate_architecture_options(
        max_lags=args.max_lags + 1, marge=3,  minimum_num_options=2,
        n_blocks=args.n_blocks, n_layers_per_block=args.n_layers_per_block, kernel_size=args.kernel_size)

    print('Options found:')
    for a in architecture_options:
        b_ = a['n_blocks']
        n_ = a['n_layers_per_block']
        k_ = a['kernel_size']
        print(a, "receptive_field:", receptive_field(b_, n_, k_))

    if args.hyperopt and len(architecture_options) > 1:
        param_space['architecture'] = hp.choice('architecture', architecture_options)
    else:
        param_space['architecture'] = architecture_options[0]

    # Overwrite parameters with user-specified values
    for key, value in vars(args).items():
        if key in param_space and value is not None:
            # overwrite value
            param_space[key] = value

    if args.causeme is not None:
        param_space['data_dir'] = 'causeme'
        param_space['dataset_name'] = args.causeme
    elif args.synthetic is not None:
        param_space['data_dir'] = 'synthetic'
        param_space['dataset_name'] = args.synthetic
        param_space['test_size'] = 0.0

    if param_space['model'] == 'NAVAR':
        param_space['n_heads'] = None
        param_space['softmax_method'] = None
    if not param_space['epistemic']:
        param_space['start_coeff'] = 0
        param_space['delta_coeff'] = 0

    return param_space


if __name__ == "__main__":
    space = get_param_space()
    print('Parsed input:')
    pprint(space)
    run(**space)
