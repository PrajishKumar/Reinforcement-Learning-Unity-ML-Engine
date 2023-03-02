import torch
import torch.nn as nn


def init_weights(m):
    """
    Initialize the weights corresponding to a NN layer with ReLU activation function.
    :param m: The NN layer.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))


class Critic(nn.Module):
    """
    The neural network representing the action-value function.
    """

    def __init__(self, state_size, action_size, seed=0):
        """
        Set the neural network architecture.

        :param state_size: Dimension of the state space.
        :param action_size: Dimension of the action space.
        :param seed: For internal random number generations.
        """
        super(Critic, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Define NN architecture.
        self.stacked_layers = nn.Sequential(
            nn.Linear(state_size + action_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Initialize weights.
        self.stacked_layers.apply(init_weights)

        # The last layer has different weights since it not passed through ReLU activation.
        self.stacked_layers[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Returns the action-values for a given state-action pair.
        :param state: The state to get the values for.
        :param action: The action to be taken in the state.
        :return: The action-value estimate for the corresponding state-action pair.
        """
        return self.stacked_layers(torch.cat((state, action), dim=1))
