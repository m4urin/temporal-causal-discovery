import torch
from matplotlib import pyplot as plt
from src.models.NAVAR import NAVAR
from src.training.train_model import train_model, train_test_split

dataset = {
    'name': 'random_N-5_T-800',
    'data': torch.randn(1, 3, 500),
    'data_noise_adjusted': torch.zeros(1, 3, 500)
}

model_params = {
    'default': {'lambda1': 0.0, 'weight_sharing': False, 'recurrent': False, 'dropout': 0.4},
    'use_recurrent_layers': {'lambda1': 0.0, 'weight_sharing': False, 'recurrent': True, 'dropout': 0.2},
    'use_weight_sharing': {'lambda1': 0.0, 'weight_sharing': True, 'recurrent': False, 'dropout': 0.2},
    'high_lambda': {'lambda1': 0.5, 'weight_sharing': False, 'recurrent': False, 'dropout': 0.2},
    'high_dropout': {'lambda1': 0.0, 'weight_sharing': False, 'recurrent': False, 'dropout': 0.5},
    'all': {'lambda1': 0.5, 'weight_sharing': True, 'recurrent': True, 'dropout': 0.5},
}

for name, params in model_params.items():
    result = train_model(
        experiment_name='memorization',
        experiment_run=name,
        dataset=dataset,
        model_type=NAVAR,
        lr=3e-3,
        epochs=500,
        weight_decay=1e-10,
        test_size=0.3,
        hidden_dim=32,
        kernel_size=3,
        n_blocks=4,
        n_layers=2,
        **params
    )
    print(result['model_params'])
    assert False
    print('\nMatrix:\n', result['train_artifacts']['matrix'])

    for phase in ['train_metrics', 'test_metrics']:
        for metric in result[phase].keys():
            plt.plot(*result[phase][metric], label=f"{phase}/{metric}")
    plt.legend()
    plt.title('loss')
    plt.show()

    train, _, _ = train_test_split(**dataset, test_size=0.3)
    plt.plot(train['x'][0, 0, -250:].cpu(), label='train_data')
    plt.plot(result['train_artifacts']['prediction'][0, -251:-1], label='prediction', linestyle='--')  # for each model
    plt.legend()
    plt.title('prediction')

    plt.show()



