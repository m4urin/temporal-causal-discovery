import os.path

import torch
from torch import nn

from src.data.causal_graph import TemporalCausalGraph
from src.data.timeseries_data import TimeSeriesData


class TemporalCausalData(nn.Module):
    def __init__(self, name: str, causal_graph: TemporalCausalGraph, timeseries_data: TimeSeriesData):
        super().__init__()
        self.name = name
        self.causal_graph = causal_graph
        self.timeseries_data = timeseries_data

    def cuda(self, device=None):
        self.timeseries_data = self.timeseries_data.cuda()
        return self.cuda(device)

    def plot(self, title=None, folder_path=None, view=False):
        graph_path = os.path.join(folder_path, 'causal_graph') if folder_path is not None else None
        plot_path = os.path.join(folder_path, 'timeseries.svg') if folder_path is not None else None

        self.causal_graph.plot(path=graph_path, view=view)

        if graph_path is not None and os.path.exists(graph_path):
            os.remove(graph_path)

        self.timeseries_data.plot(title=title, path=plot_path, view=view)

    def save(self, folder, plot_name=None):
        path = os.path.join(folder, 'causal_dataset.pt')
        torch.save(self, path)
        self.plot(plot_name, folder_path=folder)

    @staticmethod
    def load(folder) -> 'TemporalCausalData':
        path = os.path.join(folder, 'causal_dataset.pt')
        return torch.load(path)
