import torch
import torch.nn as nn


class QNetwork(nn.Module):
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
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.stacked_layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state):
        """
        Returns the Q-values for every possible action at a given state.
        :param state: The state to get the values for.
        :return: Vector of Q-values, corresponding to all possible actions.
        """
        return self.stacked_layers(state)
