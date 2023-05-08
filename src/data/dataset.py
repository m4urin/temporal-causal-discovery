import torch


class Dataset:
    def __init__(self, data: torch.Tensor, name: str = None):
        self.data = data
        self.batch_size, self.num_variables, self.sequence_length = data.shape
        self.name = name

    def __str__(self):
        return f"Dataset(name={self.name + ', ' if self.name is not None else ''}batch_size={self.batch_size}, " \
               f"num_variables={self.num_variables}, sequence_length={self.sequence_length})"

    def __repr__(self):
        return str(self)

    def cuda(self):
        self.data = self.data.cuda()
        return self

    def cpu(self):
        self.data = self.data.detach().cpu()
        return self

    def __getitem__(self, index):
        data = self.data[index]
        if len(data.shape) == 3:
            return Dataset(data)
        return data


def normalize(x: torch.Tensor, dim: int):
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(dim=dim, keepdim=True)

