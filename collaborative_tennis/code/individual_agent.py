import random
import numpy as np
import torch

from actor_model import Actor
from critic_model import Critic
from learning_parameters import *


class IndividualAgent:
    """
    The agent that interacts with the environment and learns from the obtained rewards.
    """

    def __init__(self, num_agents, observation_size, action_size, device, seed=0):
        """
        Initializes the agent with the Q networks.
        :param num_agents: The total number of individual agents (including this one).
        :param observation_size: Dimension of the observation space.
        :param action_size: Dimension of the action space. 
        :param device: The device on which the Torch computations are running.
        :param seed: For random number generation.
        """
        self.observation_size = observation_size
        self.action_size = action_size
        self.device = device
        random.seed(seed)

        # The "local" policy network that is learnt over time.
        self.actor_local = Actor(observation_size, action_size, seed).to(device)

        # The "target" policy network that's updated less frequently to avoid oscillating updates.
        self.actor_target = Actor(observation_size, action_size, seed).to(device)

        # Optimizer for learning the parameters of the policy network.
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # The "local" action-value network that is learnt over time.
        self.critic_local = Critic(observation_size, action_size, num_agents, seed).to(device)

        # The "target" action-value network that's updated less frequently to avoid oscillating updates.
        self.critic_target = Critic(observation_size, action_size, num_agents, seed).to(device)

        # Optimizer for learning the parameters of the action-value network.
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,
                                                 weight_decay=WEIGHT_DECAY)

        # Loss function for the critic.
        self.critic_loss_function = torch.nn.MSELoss()

        # Set the parameters of the target networks to that of the local networks.
        for local_param, target_param in zip(self.actor_local.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(local_param.data)
        for local_param, target_param in zip(self.critic_local.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(local_param.data)

    def act_local(self, observations):
        """
        Returns the actions that the local agents should take at current observations.
        :param observations: The observations of the agent.
        :return: The actions to take.
        """
        return self.__act(self.actor_local, observations)

    def act_target(self, observations):
        """
        Returns the actions that the target agents should take at current observations.
        :param observations: The observations of the agent.
        :return: The actions to take.
        """
        return self.__act(self.actor_target, observations)

    def __act(self, actor, observations):
        """
        Returns the actions that the agents should take for the current observations.
        :param actor: The actor used.
        :param observations: The observations of the agent.
        :return: The actions to take.
        """
        if type(observations).__module__ == np.__name__:  # if type numpy, convert to torch.
            observations = torch.from_numpy(observations).float().to(self.device)

        actor.eval()
        with torch.no_grad():
            action = actor(observations).float().to(self.device).detach()
        actor.train()

        return action
