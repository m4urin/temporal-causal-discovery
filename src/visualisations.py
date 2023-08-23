import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
import matplotlib.lines as mlines


def plot_train_val_loss(train_losses: list[float], val_losses: list[float] = None,
                        test_every: int = 1, path=None, show_plot=False):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='train loss')
    if len(val_losses) > 0:
        ax.plot(np.arange(0, len(val_losses)) * test_every, val_losses, label='eval loss')
        ax.set_title('Train and Validation Loss')
    else:
        ax.set_title('Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train and Validation Loss')
    ax.legend()

    if path is not None:
        plt.savefig(path)

    if show_plot:
        plt.show()
    plt.close()


def plot_multiple_timeseries(y, x=None, title=None, y_labels=None, labels=None, colors=None, x_label=None, path=None, limit=None, view=False):
    """
    Plot multiple time series on separate subplots.

    Parameters:
        y (ndarray): 3-dimensional array representing the time series data.
                        The expected shape is (k, n, t), where k is the number of time series,
                        n represents the number of time instances to plot,
                        and t is the number of data points for each time series.
        title (str, optional): Title of the plot. Defaults to None.
        y_labels (list of str, optional): List of names for each time series (k in total). Defaults to None.
        labels (list of str, optional): List of labels for each sequence (n in total). Defaults to None.
        x_label (str, optional): Label for the x-axis. Defaults to None.
        path (str, optional): File path to save the plot as an image. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If the input data has an invalid shape.

    """
    #plt.clf()
    k, n, t = y.shape  # Get the dimensions of the input data

    if x is None:
        x = np.full((k, n, t), np.arange(0, t))

    # Check if the shape of data is valid
    if len(y.shape) != 3:
        raise ValueError("Invalid shape of 'data'. Expected shape: (k, n, t)")

    fig, axs = plt.subplots(k, 1, figsize=(8, 6), sharex=True)

    if labels is None:
        labels = [None] * n
    if colors is None:
        colors = [None] * n

    if k < 2:
        axs = [axs]

    for i in range(k):
        for j in range(n):
            axs[i].plot(x[i, j, :], y[i, j, :], label=labels[j], linewidth=1.0, color=colors[j])

        if y_labels is not None:
            axs[i].set_ylabel(y_labels[i])  # Set the y-axis label for each subplot

        if limit is not None:
            axs[i].set_ylim(*limit)  # Set y-axis limits

    axs[-1].set_xlabel("Timesteps" if x_label is None else x_label)  # Set the x-axis label

    if title is not None:
        fig.suptitle(title)  # Set the title of the plot

    fig.subplots_adjust(hspace=0.2)  # Adjust the spacing between subplots

    # Create a single legend for all plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    if path is not None:
        plt.savefig(path)  # Save the plot as an image if the path is provided

    if view:
        plt.show()  # Display the plot
    plt.close()


def plot_gaussian_curves(mu_list, std_list):
    #plt.clf()

    x = np.linspace((mu_list - 3*std_list).min().item(), (mu_list + 3*std_list).max().item(), 1000)  # Range of x-axis values
    mu_list = mu_list.tolist()
    std_list = std_list.tolist()

    colored = len(mu_list) - 1
    for i, (mu, std) in enumerate(zip(mu_list, std_list)):
        y = 1/(std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / std)**2)
        plt.plot(x, y, label=f"mu={mu}, std={std}", color='red' if i == colored else 'black')

    plt.xlabel('X-axis')
    plt.ylabel('Probability Density')
    plt.title('Gaussian Curves')
    plt.legend()
    plt.show()
    plt.close()


def plot_heatmap(true_matrix, matrices, view=False, path=None, names=None):
    #plt.clf()

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(5, 6))

    if len(matrices) == 1:
        ax = [ax]

    if names is None:
        names = [None] * len(matrices)

    im = ax[0][0].imshow(true_matrix, cmap='Reds', interpolation=None, vmin=0.0, vmax=0.5)
    ax[0][0].set_title("True")

    divider = make_axes_locatable(ax[0][0])
    cax = divider.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(im, cax=cax, orientation='vertical')

    ax[0][1].axis('off')

    ax[0][0].set_yticks(list(range(true_matrix.shape[0])))
    ax[0][0].set_xticks(list(range(true_matrix.shape[1])))
    ax[0][0].set_ylabel("Incoming")

    for i, array in enumerate(matrices):
        j1, j2 = 1 + (i // 2), i % 2
        _ax = ax[j1, j2]
        im = _ax.imshow(array, cmap='Reds', interpolation=None, vmin=0.0, vmax=0.5)
        _ax.set_title(names[i])

        divider = make_axes_locatable(_ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(im, cax=cax, orientation='vertical')

        if j1 == 2:
            _ax.set_xlabel("Outgoing")
        if j2 == 0:
            _ax.set_ylabel("Incoming")
        _ax.set_yticks(list(range(array.shape[0])))
        _ax.set_xticks(list(range(array.shape[1])))

    fig.tight_layout(pad=0.6)

    if path is not None:
        plt.savefig(path)

    if view:
        plt.show()

    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float) -> None:
    """
    Plot the Receiver Operating Characteristic Curve.
    """
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')


""" 3D plots """
def plot_3d_points(x1, x2, y, ax=None, title=None, label=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, marker='o', color='red', zorder=2, label=label)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')

    if title is not None:
        ax.set_title(title)

    return ax


def plot_mesh(x1, x2, y, ax=None, title=None, cmap=None, label=None, label_color=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x1, x2, y, cmap=cmap, zorder=-1, alpha=0.60, label=label)

    if label is not None:
        # create invisible line
        surf._facecolors2d = label_color  # np array with [[R, G, B, A]]
        surf._edgecolors2d = label_color

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    if title is not None:
        ax.set_title(title)
    return ax


if __name__ == '__main__':
    a = np.array([[0, 2, 3], [1, 1, 1], [0.1, 5, 0.2]])
    b = np.array([[0, 1, 1], [1, 2, 1], [0.9, 1, 0.1]])
    plot_heatmap(a, [a, b], view=True, names=["A", "BB"])

