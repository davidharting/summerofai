import torch


class LinearNet(torch.nn.Module):
    def __init__(self, nodes=[2, 4, 8, 4, 1]):
        super(LinearNet, self).__init__()
        # Define layers we want in the network
        # This part is the "art" of neural nets
        # Too many neurons and we will over-fit
        # By default, we will go with 4 -> 8 -> 4 for the hidden layers, and assume 2 inputs and 1 output
        self.number_of_layers = len(nodes)
        for index, input_nodes in enumerate(nodes):
            if index < len(nodes) - 1:
                output_nodes = nodes[index + 1]
                property_name = f"layer{index + 1}"
                layer = torch.nn.Linear(input_nodes, output_nodes)
                setattr(self, property_name, layer)

    def forward(self, x):
        """
        All networks must have this function to define how data flows through the network.
        x -- The input tensor
        """
        F = torch.nn.functional

        for i in range(self.number_of_layers - 1):
            layer = getattr(self, f"layer{i + 1}")
            x = layer(x)
            if i < self.number_of_layers - 2:
                # No activation on last layer
                # Because we want to return a continuous real number. Not just positive numbers.
                x = F.relu(x)

        return x
