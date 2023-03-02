import torch
import torch.nn as nn


def init_weights(m):
    """
    Initialize the weights corresponding to a NN layer with ReLU activation function.
    :param m: The NN layer.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))


class Actor(nn.Module):
    """
    The neural network representing the policy.
    """

    def __init__(self, state_size, action_size, seed=0):
        """
        Set the neural network architecture.

        :param state_size: Dimension of the state space.
        :param action_size: Dimension of the action space.
        :param seed: For internal random number generations.
        """
        super(Actor, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Define NN architecture.
        self.stacked_layers = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Tanh(),
        )

        # Initialize weights.
        self.stacked_layers.apply(init_weights)

    def forward(self, state):
        """
        Returns the action values at a given state.
        :param state: The state to get the actions for.
        :return: Vector of actions in range [-1, 1].
        """
        return self.stacked_layers(state)
