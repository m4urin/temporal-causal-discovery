from time import time

import numpy as np
import torch

from src.data.temporal_causal_data import TemporalCausalData
from src.eval.auc_roc import calculate_AUROC
from src.models.model_outputs import EvaluationResult, TrainResult, ModelOutput
from src.models.model_config import ModelConfig, TrainConfig
from src.utils.progress_3 import Nested_trange
from src.utils.pytorch import exponential_scheduler_with_warmup


def train_model(causal_data: TemporalCausalData,
                model_config: ModelConfig,
                train_config: TrainConfig,
                progress_manager: Nested_trange) -> EvaluationResult:

    dataset = causal_data.timeseries_data
    # Record start time
    start_time = time()

    # Test the model every 30 epochs, or at the last epoch
    test_every = 25
    assert train_config.num_epochs % test_every == 0, 'num_epoch should be a multiple of 25'

    # Instantiate the model
    model = model_config.instantiate_model()

    # Move the model and data to the GPU if it's available
    if torch.cuda.is_available():
        model = model.cuda()
        dataset = dataset.cuda()

    # Split data into experiments and validation sets
    if train_config.test_size == 0:
        train_data, test_data = dataset, dataset
    else:
        train_data, test_data = dataset.train_test_split(train_config.test_size)

    # Define the optimizer
    optimizer = train_config.optimizer(model.parameters(),
                                       lr=train_config.learning_rate,
                                       weight_decay=train_config.weight_decay)
    scheduler = exponential_scheduler_with_warmup(optimizer, start_factor=0.01, end_factor=0.01,
                                                  warmup_iters=100, exp_iters=train_config.num_epochs // 2,
                                                  total_iters=train_config.num_epochs)

    # Define lists to store the experiments and validation losses
    train_losses, test_losses, train_losses_true, test_losses_true = [], [], [], []

    # AUCROC
    auroc_scores = []
    true_causal_matrix = causal_data.causal_graph.get_causal_matrix(exclude_max_lags=True)

    # Set the model to experiments mode
    model.train()

    # Train the model
    for epoch in progress_manager.iter(loop_index=2):
        # Forward pass and loss calculation
        model_output: ModelOutput = model(train_data.train_data)

        # Backward pass and optimization
        model_output.get_loss().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        # Store the experiments loss
        train_losses.append(model_output.get_regression_loss().item())

        # Test the model on the validation set
        if epoch % test_every == 0 or epoch == train_config.num_epochs - 1:
            with torch.no_grad():
                model_output = model.forward_eval(test_data.train_data)
                test_losses.append(model_output.get_regression_loss().item())
                train_losses_true.append(model.forward_eval(train_data.train_data, train_data.true_data).get_regression_loss().item())
                test_losses_true.append(model.forward_eval(test_data.train_data, test_data.true_data).get_regression_loss().item())
                cm_size = model_output.causal_matrix.size()
                auroc_scores.append({
                    'epoch': epoch,
                    **calculate_AUROC(model_output.causal_matrix, true_causal_matrix[:cm_size[0], :cm_size[1]])
                })
                progress_manager.set_info("train={:.4f}, test={:.4f}, AUCROC={:.3f}".format(
                    train_losses[-1], test_losses[-1], auroc_scores[-1]['score']), refresh=False)

                if epoch > train_config.num_epochs // 2 and auroc_scores[-1]['score'] < 0.5:
                    break

    # Evaluate the model on the experiments data
    with torch.no_grad():
        if model.use_variational_layer:
            model_result = model.forward_eval(dataset.train_data, dataset.true_data).cpu()
        else:
            model_result = model.monte_carlo(dataset.train_data, dataset.true_data, samples=200, max_batch_size=25).cpu()

    train_result = TrainResult(model,
                               train_losses, test_losses,
                               train_losses_true, test_losses_true,
                               auroc_scores,
                               test_every,
                               time() - start_time)

    # Return the evaluation result object
    return EvaluationResult(
        dataset=causal_data,
        model_config=model_config,
        model_output=model_result,
        train_config=train_config,
        train_result=train_result
    )
