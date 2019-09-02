import torch
from lib.network import LinearNet


class LinearModel:
    def __init__(self, nodes=[2, 4, 8, 4, 1]):
        self.network = LinearNet(nodes)

    def train(self, inputs, outputs, epochs=1000):
        # convert input to tensors
        # TODO: Do not assume float input type
        tensor_in = torch.tensor(inputs).float()
        expected = torch.tensor(outputs).float()

        # define loss function
        criterion = torch.nn.MSELoss()  # appropriate for continuous output numbers

        # define optimizer (lr = "learning rate")
        optimizer = torch.optim.SGD(self.network.parameters(), lr=0.001)

        for _ in range(epochs):
            self.network.zero_grad()  # Make sure each pass of loop has a clean network
            output = self.network(tensor_in)
            loss = criterion(output, expected)
            loss.backward()
            optimizer.step()

    def test(self, inputs):
        tensor_in = torch.tensor(inputs).float()
        return self.network(tensor_in).tolist()

