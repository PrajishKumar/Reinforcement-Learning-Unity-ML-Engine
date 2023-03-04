import torch.nn as nn


def update_target_network(local_model, target_model, tau):
    """
    Update the model parameters of the target Q network.

    θ_target = τ * θ_local + (1 - τ) * θ_target
    :param local_model: The model used as a reference to update.
    :param target_model: The model that is updated.
    :param tau: Interpolation parameter.
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def init_relu_weights(m):
    """
    Initialize the weights corresponding to a NN layer with ReLU activation function.
    :param m: The NN layer.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
