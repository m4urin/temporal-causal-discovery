import torch
import torch.optim as optim
from tqdm import trange


class AutoregressiveTrainer:
    def __init__(self, model, loss_fn, learning_rate=1e-4, optimizer_name='Adam'):
        self.model = model
        self.optimizer_name = optimizer_name
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

        self._create_optimizer()

    def _create_optimizer(self):
        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f'Invalid optimizer name: {self.optimizer_name}')

    def train(self, x, num_epochs, update_every=20):

        # Move data and model to device if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            x = x.cuda()

        # Split input data into two parts, current input and target
        current_input = x[:, :, :-1]
        target = x[:, :, 1:]

        self.model.train()
        pbar = trange(num_epochs)
        for epoch in pbar:
            # Perform a training step on the current batch of data
            self.optimizer.zero_grad()

            # Run the model on the current input to predict the target
            output = self.model(current_input)

            # Calculate the loss between predicted output and target
            loss = self.loss_fn(output, target)

            loss.backward()
            self.optimizer.step()

            if epoch % update_every == 0 or epoch == num_epochs - 1:
                pbar.set_description(
                    'Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
