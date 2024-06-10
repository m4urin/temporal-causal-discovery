from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt, gridspec
from sklearn.metrics import roc_curve, auc

from src.utils import smooth_line, to_numpy


def plot_contemporaneous_relationships(
        *temporal_matrices,
        causal_links: List[tuple[int, int]],
        names: List[str] = None,
        smooth: float = None):
    """
    Plot the contemporaneous (synchronous) causal relationships between variables over time.

    Args:
        temporal_matrices (array-like): Causal matrix changing over time.
            The shape of each matrix should be of size (n_vars, n_vars, seq_length).
        causal_links (List[tuple[int, int]]): A list of tuples representing directed edges (a, b) where
            'a' causally influences 'b'.
        names (List[str], optional): A list of names for the temporal matrices, e.g., ['Ground truth', 'Model 1'].
            Defaults to None.
        smooth (float, optional): Smoothing parameter for the line smoothing (>0). Defaults to None.

    Example:
        >>> data = np.random.rand(3, 4, 4, 200) *0.5
        >>> data[0] = (data[0] > 0.4).astype(float)
        >>> plot_contemporaneous_relationships(*data, \
                                               causal_links=[(0, 1), (1, 2), (2, 0)], \
                                               names=['Ground truth', 'Model 1', 'Model 2'])
        >>> plt.show()
    """
    # Handle the default case for 'names'
    if names is None:
        names = [None] * len(temporal_matrices)
    else:
        assert len(names) == len(temporal_matrices), "Length of names should match number of temporal_matrices"

    # Convert all matrices to NumPy format for consistent processing
    temporal_matrices = np.stack([to_numpy(m) for m in temporal_matrices])

    # Apply line smoothing if specified
    if smooth:
        temporal_matrices = smooth_line(temporal_matrices, sigma=smooth, axis=-1)

    k, n_vars, _, seq_length = temporal_matrices.shape

    # Initialize the figure
    fig = plt.figure(figsize=(3 * len(causal_links) + 0.5, 1 + 0.3 * k))

    # Define grid layout
    gs = gridspec.GridSpec(1, len(causal_links) + 1,
                           width_ratios=[1.0] * len(causal_links) + [0.05],
                           height_ratios=[1.0],
                           figure=fig)

    # Create subplots based on the grid layout
    axs = [plt.subplot(gs[i]) for i in range(len(causal_links))]
    cax = plt.subplot(gs[-1])  # Colorbar axes

    first_im = None  # Store the first image for the colorbar

    # Plot each causal link
    for i, (a, b) in enumerate(causal_links):
        im = axs[i].imshow(temporal_matrices[:, b, a], aspect='auto', cmap='viridis',
                           interpolation='nearest', origin='upper', vmin=0.0, vmax=1.0)
        axs[i].set_title(r"$X^{(" + str(a) + ")} \\;\\to \\; X^{(" + str(b) + ")}$")
        axs[i].set_xlabel('Time')
        axs[i].set_yticks([])
        if i == 0:
            first_im = im  # Capture the first image to sync the colorbar

    # Set the y-ticks and labels only for the first subplot
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels(names)

    # Add a colorbar to the figure
    plt.colorbar(first_im, cax=cax)

    # Ensure the layout fits without overlaps
    plt.tight_layout()


def plot_all_contemporaneous_relationships(temporal_matrix, name: str = None, smooth: float = None):
    """
    Plot the contemporaneous (synchronous) causal relationships between variables over time for a single model.

    Args:
        temporal_matrix (array-like): Causal matrix for a single model.
            The shape should be of size (n_vars, n_vars, seq_length).
        name (str, optional): Name of the temporal matrix, e.g., 'Model 1'.
            Defaults to None.
        smooth (float, optional): Smoothing parameter for the line smoothing (>0). Defaults to None.

    Example:
        >>> data = np.random.rand(4, 4, 100)
        >>> plot_all_contemporaneous_relationships(data, name='Model 1')
        >>> plt.show()
    """
    # Convert matrix to NumPy format
    temporal_matrix = np.asarray(temporal_matrix)

    # Apply line smoothing if specified
    if smooth:
        temporal_matrix = smooth_line(temporal_matrix, sigma=smooth, axis=-1)

    n_vars, _, seq_length = temporal_matrix.shape

    # Generate all possible causal links
    causal_links = [(i, j) for i in range(n_vars) for j in range(n_vars)]

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(7, len(causal_links) / 4))

    # Prepare data for plotting
    plot_data = np.array([temporal_matrix[j, i] for i, j in causal_links])

    # Create the image
    im = ax.imshow(plot_data, aspect='auto', cmap='viridis',
                   interpolation='nearest', origin='upper', vmin=0.0, vmax=1.0)

    # Set titles and labels
    ax.set_title(f'Contemporaneous Relationships in {name}' if name else 'Contemporaneous Relationships')
    ax.set_xlabel('Time')
    ax.set_ylabel('Causal Links')

    # Set the y-ticks and labels for causal links
    ax.set_yticks(range(len(causal_links)))
    ax.set_yticklabels([f'X{a} -> X{b}' for a, b in causal_links])

    # Add a colorbar to the figure
    plt.colorbar(im, ax=ax)

    # Ensure the layout fits without overlaps
    plt.tight_layout()


def plot_train_val_loss(
        train_losses,
        val_losses=None,
        interval: int = 20,
        smooth: float = None):
    """
    Plot training and validation losses over time.

    Args:
        train_losses (array-like): List of training losses.
        val_losses (array-like, optional): List of validation losses (if available). Default is None.
        interval (int, optional): Interval for x-axis ticks. Default is 20.
        smooth (float, optional): Gaussian smoothing strength.
    """
    plt.clf()
    train_losses = to_numpy(train_losses)
    if val_losses is not None:
        val_losses = to_numpy(val_losses)

    if smooth is not None:
        train_losses = smooth_line(train_losses, sigma=smooth)
        if val_losses is not None:
            val_losses = smooth_line(val_losses, sigma=smooth)

    x = interval * np.arange(1, len(train_losses) + 1)

    # Plot training losses
    plt.plot(x, train_losses, label='Training Loss')

    if val_losses is not None:
        # Plot validation losses
        plt.plot(x[:len(val_losses)], val_losses, label='Validation Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)


def plot_multiple_time_series(*data,
                              x_label: str = None,
                              y_labels: List[str] = None,
                              y_limit: Tuple[float, float] = None,
                              x_limit: Tuple[float, float] = None,
                              sequence_labels: List[str] = None,
                              smooth: float = None):
    """
    Plot multiple time series on separate subplots.

    Args:
        data (array-like): The expected shape is num_plots * (num_seq, sequence_length).
        x_label (str, optional): Label for the x-axis. Defaults to None.
        y_labels (List[str], optional): List of labels for the y-axis for each subplot.
            The list length should be equal to num_plots. Defaults to None.
        y_limit (Tuple[float, float], optional): Tuple (min_y, max_y) to set y-axis limits for all subplots.
            Defaults to None.
        x_limit (Tuple[float, float], optional): Tuple (min_x, max_x) to set x-axis limits for all subplots.
            Defaults to None.
        sequence_labels (List[str], optional): List of labels for each sequence (line) in the subplots.
            The list length should be equal to num_seq. Defaults to None.
        smooth (float, optional): Smoothing parameter for the line smoothing (>0). Defaults to None.

    Raises:
        ValueError: If the shape of the data is not 3-dimensional.

    Example:
        >>> data = np.random.randn(4, 2, 100)  # 4 plots, 2 sequences in each, 100 points for each sequence
        >>> plot_multiple_time_series(*data, x_label='Time', y_labels=['Plot1', 'Plot2', 'Plot3', 'Plot4'], \
                                      sequence_labels=['A', 'B'], smooth=2.0, y_limit=(-1, 1))
        >>> plt.show()
    """

    # Convert all data to NumPy format for consistent processing
    data = np.stack([to_numpy(x) for x in data])

    # Get the dimensions of the data
    num_plots, num_seq, sequence_length = data.shape

    if smooth:
        data = smooth_line(data, sigma=smooth)

    # Create subplots
    fig, axs = plt.subplots(num_plots, 1, figsize=(6, 1.0 * num_plots), sharex=True)

    # Default sequence labels if not provided
    if sequence_labels is None:
        sequence_labels = [None] * num_seq

    # If there's only one plot, axs should still be a list
    if num_plots < 2:
        axs = [axs]

    # Loop through each subplot to plot the data
    for i in range(num_plots):
        for j in range(num_seq):
            axs[i].plot(data[i, j, :], label=sequence_labels[j], linewidth=1.0)
        if y_labels is not None:
            axs[i].set_ylabel(y_labels[i])
        if y_limit is not None:
            axs[i].set_ylim(*y_limit)
        if x_limit is not None:
            axs[i].set_xlim(*x_limit)

    # Set x-axis label for the last subplot
    axs[-1].set_xlabel("Timesteps" if x_label is None else x_label)

    # Adjust subplot layout
    fig.subplots_adjust(hspace=0.2)

    # Create a legend for all subplots
    handles, sequence_labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, sequence_labels, loc='upper right')


def plot_heatmaps(*matrices, names: List[str] = None, use_ticks=True):
    """
    Plot a series of heatmaps.

    This function plots heatmaps from given matrices, ensuring that all heatmaps are
    of the same size, including the one with the colorbar. This function also allows
    for the input matrices to either be NumPy arrays or PyTorch tensors.

    Args:
        *matrices (array-like): The matrices to plot as heatmaps.
        names (List[str], optional): The titles for each heatmap. Defaults to None, which
                                     means no titles will be displayed.

    Example:
        >>> data = np.random.randn(3, 5, 5)
        >>> plot_heatmaps(*data, names=['Matrix 1', 'Matrix 2', "M3"])
        >>> plt.show()
    """

    # Convert matrices to NumPy arrays if they are PyTorch tensors
    matrices = [to_numpy(m) for m in matrices]

    # Create a figure and layout grid for the heatmaps and colorbar
    fig = plt.figure(figsize=(2 * len(matrices) + 0.9, 3))
    gs = gridspec.GridSpec(1, len(matrices) + 1, width_ratios=[1.0] * len(matrices) + [0.1])

    # If no names are provided, create a list of None names
    if names is None:
        names = [None] * len(matrices)

    first_im = None  # Initialize image object for colorbar

    # Iterate through each matrix to create its corresponding heatmap
    for i, matrix in enumerate(matrices):
        ax = plt.subplot(gs[i])
        im = ax.imshow(matrix, cmap='Blues', interpolation=None, vmin=0.0, vmax=1.0)
        ax.set_title(names[i])  # Set title for each heatmap if names are provided
        ax.set_yticks([])
        ax.set_xticks([])

        if use_ticks:
            ax.set_xticks(list(range(matrix.shape[1])))
            if i == 0:
                first_im = im  # Store the image object of the first heatmap for colorbar
                ax.set_yticks(list(range(matrix.shape[0])))

    # Create an axis for the colorbar
    cax = plt.subplot(gs[-1])

    # Add colorbar to the last axis
    fig.colorbar(first_im, cax=cax, orientation='vertical')

    # Ensure the layout fits within the figure size
    plt.tight_layout()


def plot_heatmaps_grid(matrices, names=None, use_ticks=True):
    """
    Plot a grid of heatmaps.

    This function plots heatmaps from a 2D grid of matrices, ensuring all heatmaps
    are of the same size, including those with the colorbar. This function also allows
    for the input matrices to either be NumPy arrays or PyTorch tensors.

    Args:
        matrices (list[list[array-like]]): The grid of matrices to plot as heatmaps.
        names (List[List[str]], optional): The titles for each heatmap. Defaults to None,
                                           which means no titles will be displayed.
        use_ticks (bool): Whether to display ticks on the axes.

    Example:
        >>> data = [np.random.randn(5, 5) for _ in range(3)]
        >>> plot_heatmaps([data, data], names=[['Matrix 1', 'Matrix 2', "M3"], ['Matrix 4', 'Matrix 5', "M6"]])
        >>> plt.show()
    """
    rows = len(matrices)
    cols = len(matrices[0])

    # Create a figure and layout grid for the heatmaps and colorbar
    fig = plt.figure(figsize=(2 * cols + 0.9, 3 * rows))
    gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1.0] * cols + [0.1], height_ratios=[1.3]*rows)

    first_im = None  # Initialize image object for colorbar

    # Iterate through each row and column in the grid
    for row in range(rows):
        for col in range(cols):
            ax = plt.subplot(gs[row, col])
            matrix = matrices[row][col]
            if matrix is None:
                continue
            im = ax.imshow(matrix, cmap='Blues', interpolation=None, vmin=0.0, vmax=1.0)

            # Set title if names are provided
            if names and names[row][col]:
                ax.set_title(names[row][col])

            # Hide ticks or set them based on use_ticks
            ax.set_xticks([])
            ax.set_yticks([])
            if use_ticks:
                ax.set_xticks(list(range(matrix.shape[1])))
                if col == 0:  # Y ticks only for the first column
                    ax.set_yticks(list(range(matrix.shape[0])))
                    first_im = first_im or im

    # Create an axis for the colorbar in the last column
    cax = plt.subplot(gs[:, -1])
    fig.colorbar(first_im, cax=cax, orientation='vertical')

    # Ensure the layout fits within the figure size
    plt.tight_layout()

if __name__ == '__main__':

    data = [[np.random.randn(5, 5) for _ in range(2)] for _ in range(2)]

    # Define titles for each matrix in a matching 2x2 grid structure
    names = [['Matrix 1-1', 'Matrix 1-2'], ['Matrix 2-1', 'Matrix 2-2']]

    # Call the plot_heatmaps function with the generated data and names
    plot_heatmaps_grid(data, names=names, use_ticks=True)

    # Show the plot
    plt.show()


def plot_learned_relationships(x, contributions, scores):
    """
    Plots the relationship between variables at adjacent time steps, overlaying learned contributions.
    The plots are organized in a grid where each row i and column j corresponds to the
    relationship from variable j at time t-1 to variable i at time t.

    Args:
        x (array-like): Input array of shape (n_var, k), where n_var is the number of variables
                        and k is the number of time steps.
        contributions (array-like): Array of shape (n_var, n_var, k) representing the learned
                        contributions between variables.
        scores (array-like): Array of shape (n_var, n_var) containing the scores that indicate
                        how well the learned relationships fit the data.

    Example:
        >>> n, t = 3, 200
        >>> plot_learned_relationships(x=np.random.randn(n, t), \
                                       contributions=np.zeros((n, n, t)), \
                                       scores=np.random.rand(n, n))
        >>> plt.show()
    """
    x = to_numpy(x)
    contributions = to_numpy(contributions)
    scores = to_numpy(scores)

    # Get the number of variables from the input array x
    n_var, time_steps = x.shape

    # Initialize the plot grid
    fig, axs = plt.subplots(n_var, n_var, figsize=(2 * n_var, 1.5 * n_var))

    # Iterate through each subplot in the grid
    for i in range(n_var):
        for j in range(n_var):
            # Get the axis object for the current subplot
            ax = axs[i, j]

            # Plot the actual data points (x[j, :-1] vs x[i, 1:])
            ax.scatter(x[j, :-1], x[i, 1:], color='blue', alpha=0.05, label='Data', edgecolors='none')

            # Plot the learned contributions (x[i] vs contributions[i, j])
            ax.scatter(x[i], contributions[i, j], color='purple', alpha=0.08, label='Learned relationship',
                       edgecolors='none')

            # Label the x-axis for the last row
            if i == n_var - 1:
                ax.set_xlabel(r"$X_{t-1}^{(" + str(j + 1) + r")}$")

            # Label the y-axis for the first column
            if j == 0:
                ax.set_ylabel(r"$X_t^{(" + str(i + 1) + r")}$")

            # Remove x and y ticks for a cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

            # Add the score to the title of the subplot
            ax.set_title(f'Score: {scores[i, j]:.2f}')

    # Adjust layout to prevent overlap
    plt.tight_layout()


def plot_multi_roc_curve(ground_truth, predictions, names: List[str]):
    """
    Plot Receiver Operating Characteristic (ROC) curves for multiple prediction methods.

    This function takes ground truth labels and predictions from multiple methods to plot
    their ROC curves on the same plot for comparison. The Area Under Curve (AUC) for each
    method is also displayed in the legend.

    Args:
        ground_truth (array-like): Ground truth labels, can be
            either torch.Tensor, numpy.ndarray, or any iterable object.
        predictions (array-like): A list containing
            arrays of prediction scores from different methods.
        names (List[str]): A list of names for each prediction method, used for labeling
            in the plot legend.

    Example:
        >>> ground_truth = (np.random.rand(50) > 0.5).astype(float)
        >>> predictions = np.random.rand(2, 50)
        >>> names = ['Method 1', 'Method 2']
        >>> plot_multi_roc_curve(ground_truth, predictions, names)
        >>> plt.show()
    """

    # Create a new figure for plotting
    plt.figure(figsize=(5, 4))

    # Convert ground truth and predictions to numpy arrays if they are not
    ground_truth = to_numpy(ground_truth)
    predictions = [to_numpy(p) for p in predictions]

    # Iterate through each set of predictions and names to plot their ROC curves
    for pred, name in zip(predictions, names):
        fpr, tpr, _ = roc_curve(ground_truth, pred)  # Calculate ROC curve points
        auc_score = auc(fpr, tpr)  # Calculate AUC score
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_score:.2f})')  # Plot the ROC curve for this method

    # Plot the diagonal line, representing random guessing
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    # Set x and y-axis limits
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # Label x and y axis
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # Place the legend in the lower right corner
    plt.legend(loc='lower right')

    plt.tight_layout()


def plot_3d_scatter_points(x1_values, x2_values, y_values, ax=None, label=None, color='red'):
    """
    Plots a 3D scatter plot of data points.

    Args:
        x1_values (array-like): Values for the x1-axis.
        x2_values (array-like): Values for the x2-axis.
        y_values (array-like): Values for the y-axis.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Pre-existing 3D axes for the scatter plot. Defaults to None.
        label (str, optional): Label for the scatter points. Defaults to None.
        color (str, optional): Color for the label. Defaults to red.

    Returns:
        matplotlib.axes._subplots.Axes3DSubplot: 3D subplot axes.

    Example:
        >>> plot_3d_scatter_points([1, 2, 3], [4, 5, 6], [7, 8, 9])
        >>> plt.show()
    """
    # Create a new 3D plot if no existing axes are provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot 3D scatter points
    ax.scatter(x1_values, x2_values, y_values, marker='o', color=color, zorder=2, label=label)

    # Label axes
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    return ax


def plot_3d_surface(x1_grid, x2_grid, y_grid, ax=None, cmap=None, label=None, color=None):
    """
    Plots a 3D surface based on meshgrid data.

    Args:
        x1_grid (array-like): Meshgrid for the x1-axis.
        x2_grid (array-like): Meshgrid for the x2-axis.
        y_grid (array-like): Meshgrid for the y-axis.
        ax (matplotlib.axes._subplots.Axes3DSubplot, optional): Pre-existing 3D axes for the surface plot. Defaults to None.
        cmap (str, optional): Colormap for the surface plot. Defaults to None.
        label (str, optional): Label for the surface plot. Defaults to None.
        color (array-like, optional): Color for the label. Defaults to None.

    Returns:
        matplotlib.axes._subplots.Axes3DSubplot: 3D subplot axes.

    Example:
        >>> x = np.linspace(0, 7, 30)
        >>> y = np.linspace(0, 7, 30)
        >>> X, Y = np.meshgrid(x, y)
        >>> print(X.shape)
        >>> Z = np.sin(X + 0.5 * Y)
        >>> plot_3d_surface(X, Y, Z)
        >>> plt.show()
    """
    # Create a new 3D plot if no existing axes are provided
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot 3D surface
    surf = ax.plot_surface(x1_grid, x2_grid, y_grid, cmap=cmap, zorder=-1, alpha=0.60, label=label)

    # Handle labels and legend colors
    if label is not None:
        surf._facecolors2d = color  # Use for legend
        surf._edgecolors2d = color

    # Label axes
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    return ax
