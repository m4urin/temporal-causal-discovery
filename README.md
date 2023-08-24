> **Note**
> This repository is a work in progress, and the current method is based on another method called NAVAR (Neural Additive Vector Autoregression). To learn more about NAVAR, please refer to the original repository [here](https://github.com/bartbussmann/NAVAR).

# Temporal Causal Discovery with Deep Learning

This repository contains code that is part of my thesis for my master's degree in Computer Science: Data Science and Artificial Intelligence at Antwerp University.
The objective of the thesis is to develop a method for robust causal discovery in time series data using machine learning.
Please feel free to contact me if you have any questions or suggestions.

The complete thesis is available for access and can be downloaded in PDF format [here](https://github.com/m4urin/temporal-causal-discovery/raw/refactor/thesis.pdf). This document presents a detailed overview of the research, methodology, and results obtained in this study.

## Command Line Arguments for `main.py`

The file `main.py` provides a collection of command line arguments for configuring and executing hyperparameter optimization tasks using `hyperopt`. 
Here is a breakdown of the arguments:

#### Model type:
- `model`: Chooses between the `NAVAR` and `TAMCaD` models. Use:
  - `NAVAR`: To choose the NAVAR model.
  - `TAMCaD`: To choose the TAMCaD model. Requires:
    - `--n_heads`: Specifies the number of attention heads.
    - `--softmax_method`: Choose from methods: softmax, softmax-1, normalized-sigmoid, gumbel-softmax, sparsemax, gumbel-softmax-1.
    - `--beta`: Regularization coefficient beta.

#### Model architecture:

- `--max_lags`: Specifies maximum lags.
- `--kernel_size`: Kernel size for the architecture.
- `--n_blocks`: Number of blocks in the architecture.
- `--n_layers_per_block`: Number of layers per block in the architecture.
- `--weight_sharing`: Flag for weight sharing (WS).
- `--recurrent`: Flag for recurrent layers (Rec).
- `--hidden_dim`: Specifies the dimension of the hidden layer. Must be divisible by `--n_heads` for TAMCaD.

It is possible to specify only `--max_lags` and a subset of `--kernel_size`, `--n_blocks` and`--n_layers_per_block`. 
A method will try to find a valid architecture and give an error if no structure can be found. 
If multiple architectures are possible, the first one will be chosen. 
If `--hyperopt` is enabled, the options will be included in the hyper-parameter optimization.

#### Uncertainty:
- `--aleatoric`: Flag for aleatoric uncertainty (A).
- `--epistemic`: Flag for epistemic uncertainty (E).
  - `--start_coeff`: Specifies the starting coefficient.
  - `--delta_coeff`: Specifies the delta coefficient.

#### Train data:
Use one of:
- `--causeme`: Use one of the causeme datasets. For example: `nonlinear-VAR_N-20_T-300`
- `--synthetic`: Use one of the synthetic datasets. For example: `synthetic_N-5_T-500_K-6`

#### Training Configuration:
- `--lr`: Specifies the learning rate.
- `--epochs`: Sets the number of epochs.
- `--weight_decay`: Regularization for weight decay.
- `--test_size`: Proportion of the dataset to include in the test split.
- `--dropout`: Sets the dropout rate.
- `--lambda1`: Regularization coefficient.

#### Hyperopt Procedure:
- `--hyperopt`: Flag to run hyperopt.
  - `--max_evals`: Specifies the maximum number of evaluations for hyperopt. A value must be provided when running hyperopt.
  - `--subset_evals`: The subset of evaluations. Required when running hyperopt on 'causeme' data.

### Usage Example:

```bash
python main.py TAMCaD --synthetic synthetic_N-5_T-500_K-6 --hyperopt --max_evals 50 --dropout 0.3
```

Note: Ensure that you meet the requirements before running, like providing `--max_evals` when running `--hyperopt`, or making sure `--hidden_dim` is divisible by `--n_heads` when using the TAMCaD model. The script contains specific validations to guide you.