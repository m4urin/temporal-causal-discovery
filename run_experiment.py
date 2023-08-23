from hyperopt import hp

def hyperopt_space():
    space = {
        'hidden_dim': hp.choice('hidden_dim', [16, 32, 64, 128]),  # Multiples of 4 between 8 and 128 (inclusive)
        'kernel_size': hp.choice('kernel_size', [x for x in range(2, 11, 2)]),  # Multiples of 2 between 2 and 10 (inclusive)
        'n_blocks': hp.choice('n_blocks', [x for x in range(1, 11)]),  # Numbers between 1 and 10 (inclusive)
        'n_layers_per_block': hp.choice('n_layers_per_block', [x for x in range(1, 11)]),  # Numbers between 1 and 10 (inclusive)
        'dropout': hp.uniform('dropout', 0.0, 1.0),
        'weight_sharing': hp.choice('weight_sharing', [True, False]),
        'recurrent': hp.choice('recurrent', [True, False]),
        'aleatoric': hp.choice('aleatoric', [True, False]),
        'epistemic': hp.choice('epistemic', [True, False]),
        'uncertainty_contributions': hp.choice('uncertainty_contributions', [True, False]),

        # TAMCaD specific
        'n_heads': hp.choice('n_heads', [1, 2, 4, 8, 16]),  # Assuming you might want to use powers of 2
        'softmax_method': hp.choice('softmax_method', ['softmax', 'softmax-1', 'normalized-sigmoid', 'gumbel-softmax']),

        # Dataset
        'dataset': hp.choice('dataset', ['CauseMe', 'SynC5']),

        # Training
        'lr': hp.loguniform('lr', -6, -2),  # Log-uniform between 1e-6 and 1e-2
        'epochs': hp.choice('epochs', [x for x in range(100, 3000, 100)]),  # Numbers between 100 and 3000, with step 100
        'weight_decay': hp.loguniform('weight_decay', -6, -2),  # Log-uniform between 1e-6 and 1e-2
        'test_size': hp.uniform('test_size', 0.0, 1.0)
    }

    return space
