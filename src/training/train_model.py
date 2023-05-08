import torch
from definitions import RESULTS_DIR
from src.data.dataset import Dataset
from src.models.temporal_causal_model import TemporalCausalModel
from src.utils.io import join_paths
from src.utils.model_outputs import EvaluationResult, TrainResult
from src.utils.progress import iter_with_progress
from src.utils.config import ModelConfig, TrainConfig
from src.utils.visualisations import plot_train_val_loss


def train_model(dataset: Dataset, model_config: ModelConfig,
                train_config: TrainConfig, show_progress=True) -> EvaluationResult:
    """
    Trains a temporal causal model using the given data and configuration.

    Args:
        data (torch.Tensor): The input data, of size (batch_size, num_variables, sequence_length).
        model_config (ModelConfig): The configuration object for the model.
        train_config (TrainConfig): The configuration object for the training process.
        show_progress: TODO

    Returns:
        EvaluationResult: The evaluation result object containing the trained model, its performance, and the training configuration.
    """

    # Split data into training and validation sets
    n_test = int(train_config.val_proportion * dataset.sequence_length)
    n_train = dataset.sequence_length - n_test

    # Test the model every 30 epochs, or at the last epoch
    test_every = 30

    # Instantiate the model
    model: TemporalCausalModel = model_config.instantiate_model()

    # Move the model and data to the GPU if it's available
    if torch.cuda.is_available():
        model = model.cuda()
        dataset = dataset.cuda()

    # Define the optimizer
    optimizer = train_config.optimizer(model.parameters(),
                                       lr=train_config.learning_rate,
                                       weight_decay=train_config.weight_decay)

    # Define lists to store the training and validation losses
    train_losses, val_losses = [], []

    # Split data into training and validation sets
    train_data = dataset[..., :n_train]
    test_data = dataset[..., n_train:]

    # Set the model to training mode
    model.train()

    # Train the model
    for epoch in iter_with_progress(train_config.num_epochs, show_progress=show_progress):
        # Forward pass and loss calculation
        loss = model(train_data.data).get_loss()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Store the training loss
        train_losses.append(loss.item())

        # Test the model on the validation set
        if n_test > 0 and (epoch % test_every == 0 or epoch == train_config.num_epochs - 1):
            # Set the model to evaluation mode
            model.eval()
            with torch.no_grad():
                val_losses.append(model(test_data.data).get_loss().item())
            model.train()

    # Plot the training and validation losses
    plot_train_val_loss(train_losses, val_losses, test_every,
                        path=join_paths(RESULTS_DIR, f'training_img/{model_config.url}.png',
                                        make_dirs=True))

    # Evaluate the model on the training data
    with torch.no_grad():
        model_result = model.get_result(train_data.data)
        train_result = TrainResult(model.cpu(), train_losses, val_losses)

    # Return the evaluation result object
    return EvaluationResult(
        dataset=dataset,
        model_config=model_config,
        model_result=model_result,
        train_config=train_config,
        train_result=train_result
    )
