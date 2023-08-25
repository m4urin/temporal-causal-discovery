

command = "bash -c 'chmod +x /project_antwerp/run.sh; /project_antwerp/run.sh {}'"
base = "TAMCaD --synthetic synthetic_N-5_T-500_K-6 --hyperopt --max_evals 120 --subset_evals 1 --n_blocks 2 --n_layers_per_block 2 --kernel_size 2 --softmax_method {}"

diff = ['softmax', 'softmax-1', 'gumbel-softmax', 'gumbel-softmax-1', 'sparsemax', 'normalized-sigmoid']

for d in diff:
    print(command.format(base.format(d)))



