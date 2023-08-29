import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils import load_synthetic_data


def plot_causal_changes(predicted_1, predicted_2, ground_truth, i):
    # Check if the tensors have the same dimensions
    assert predicted_1.size() == predicted_2.size() == ground_truth.size()

    N, _, T = predicted_1.size()
    assert 0 <= i < N, "i should be within the range [0, N-1]"

    # Extract the i-th row for each matrix at every time step
    row_predicted_1 = predicted_1[i, :, :].cpu().numpy()
    row_predicted_2 = predicted_2[i, :, :].cpu().numpy()
    row_ground_truth = ground_truth[i, :, :].cpu().numpy()

    # Set up the figure and axes
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    time_points = np.arange(T)

    # Plot for predicted_1
    cax1 = axs[0].imshow(row_predicted_1, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
    axs[0].set_title('NAVAR')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Incoming connections for the affected variable')
    fig.colorbar(cax1, ax=axs[0])

    # Plot for predicted_2
    cax2 = axs[1].imshow(row_predicted_2, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
    axs[1].set_title('TAMCaD')
    axs[1].set_xlabel('Time')
    #axs[1].set_ylabel('Incoming connections for the affected variable')
    fig.colorbar(cax2, ax=axs[1])

    # Plot for ground_truth
    cax3 = axs[2].imshow(row_ground_truth, aspect='auto', cmap='viridis', interpolation='nearest', origin='upper')
    axs[2].set_title('Ground Truth')
    axs[2].set_xlabel('Time')
    #axs[2].set_ylabel('Incoming connections for the affected variable')
    fig.colorbar(cax3, ax=axs[2])

    plt.tight_layout()
    plt.show()


def load_matrices(name):
    # Load data from .pt files
    d1_path = r"C:\Users\mauri\PycharmProjects\temporal-causal-discovery\results\synthetic\NAVAR_{}\results.pt".format(name)
    d1 = torch.load(d1_path)
    predicted_1 = d1[0]['train_phase']['causal_matrix_best']

    d2_path = r"C:\Users\mauri\PycharmProjects\temporal-causal-discovery\results\synthetic\TAMCaD-softmax_{}\results.pt".format(name)
    d2 = torch.load(d2_path)
    predicted_2 = d2[0]['train_phase']['causal_matrix_best']

    # Load ground_truth from the load_synthetic_data function
    g = load_synthetic_data(name)
    ground_truth = g['gt']

    return predicted_1[0], predicted_2[0], ground_truth[0, 0, :, :predicted_1.size(-2), 1:]

# Use the function
predicted_1, predicted_2, ground_truth = load_matrices('synthetic_N-12_T-1000_K-6')

print(predicted_1.size(), predicted_2.size(), ground_truth.size())
for k in range(6, 12):
    plot_causal_changes(predicted_1, predicted_2, ground_truth, k)
