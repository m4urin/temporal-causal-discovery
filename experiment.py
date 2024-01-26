import os

import torch

from io import OUTPUT_DIR
from src.models.SimpleAttention_ignore import SimpleAttention, train_model
from src.utils import load_causeme_data, ConsoleProgressBar, write_bz2_file


EXP = 'TestWEATHnoise_N-10_T-1000'
MODEL = 'TestWEATHnoise'


def get_output_dir(dataset_name: str) -> str:
    """Generate output directory based on current date, time, dataset name, and GPULAB job ID."""
    dir_name = f"experiment_{dataset_name}"
    full_path = os.path.join(OUTPUT_DIR, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def make_causal_matrix(attentions, attentions_std, **kwargs):
    max_ = attentions.max(dim=-1, keepdim=True).values
    attn = attentions / max_
    attn_std = attentions_std / max_
    return (attn - attn_std).clamp(min=0).t().cpu()


def write_to_file(causal_matrices, output_dir):  # (200, 5, 5) tensor
    file_dict = {
        'experiment': EXP,
        'model': MODEL,
        'method_sha': '8fbf8af651eb4be7a3c25caeb267928a',
        'scores': causal_matrices.reshape(200, -1).numpy().tolist(),
        'parameter_values': 'max_lags=7, hidden_dim=32'
    }
    write_bz2_file(os.path.join(output_dir, f'result.json.bz2'), file_dict)


def run():
    data = load_causeme_data(EXP)['data'].cuda()  # (200, 1, 5, 300)
    output_dir = get_output_dir(EXP)

    pbar = ConsoleProgressBar(total=len(data), title=f'Run {EXP}')
    results = []
    for d in data:
        x, y = d[..., :-1], d[..., 1:]
        model = SimpleAttention(n_samples=20, n_nodes=x.size(1), n_layers=3, hidden_dim=24, dropout=0.1).cuda()
        result = train_model(model, x, y, epochs=300, lr=2e-2, disable_tqdm=False)
        cm = make_causal_matrix(**result)
        results.append(cm)
        pbar.update()

    results = torch.stack(results)
    write_to_file(results, output_dir)


if __name__ == '__main__':
    run()
