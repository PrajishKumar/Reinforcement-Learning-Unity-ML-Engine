import random
from replay_buffer import ReplayBuffer
from q_network import QNetwork
import numpy as np
import torch
from learning_parameters import *


class Agent:
    """
    The agent that interacts with the environment and learns from the obtained rewards.  
    
    Note: The agent learns with the DQN algorithm.
    """

    def __init__(self, state_size, action_size, device, seed=0):
        """
        Initializes the agent with the Q networks. 
        :param state_size: Dimension of the state space. 
        :param action_size: Dimension of the action space. 
        :param device: The device on which the Torch computations are running.
        :param seed: For random number generation.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        random.seed(seed)

        # The "local" action-value network that is learnt over time.
        self.q_network_local = QNetwork(state_size, action_size, seed).to(device)

        # The "target" action-value network that's updated less frequently to avoid oscillating updates to Q values.  
        self.q_network_target = QNetwork(state_size, action_size, seed).to(device)

        # Optimizer for learning the network parameters. 
        self.optimizer = torch.optim.Adam(self.q_network_local.parameters(), lr=LR)

        # Replay buffer. 
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step. 
        self.t_step = 0

        # Loss function used. 
        self.loss_function = torch.nn.MSELoss()

    def step(self, state, action, reward, next_state, done):
        """
        Adds experience tuple to the replay buffer and optionally update the local Q network as needed. 
        :param state: The state of the environment.
        :param action: The action the agent took at @p state.
        :param reward: The reward obtained by taking @p action at @p state.
        :param next_state:  The new state of the environment once agent @p action at @p state.
        :param done: Flag to know if the environment is now in a terminal state.
        """
        # Save experience in replay memory.
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_buffer) > BATCH_SIZE:
                experiences = self.replay_buffer.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """
        Returns the actions that the agent should take at a state. 
        :param state: The state of the environment. 
        :param eps: Epsilon, for epsilon-greedy action selection. 
        :return: The action to take. 
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        # Epsilon-greedy action selection.
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Updates the Q-networks.
        :param experiences: Experience (s, a, r, s', done) tuples.
        :param gamma: The discount factor.
        """
        states, actions, rewards, next_states, dones = experiences

        # Action value computed by the local Q network.
        action_value_q_local = torch.gather(self.q_network_local(states), dim=1, index=actions)

        # Expected action value, from the target Q network.
        value_of_next_state = torch.reshape(torch.max(self.q_network_target(next_states), dim=1)[0], (-1, 1))
        action_target_q_target = rewards + gamma * value_of_next_state * (1 - dones)

        # Define the loss function.
        loss = self.loss_function(action_value_q_local, action_target_q_target.detach())

        # Update the Q network parameters.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target Q network.
        self.update_target_q_network(TAU)

    def update_target_q_network(self, tau):
        """
        Update the model parameters of the target Q network.

        θ_target = τ * θ_local + (1 - τ) * θ_target
        :param tau: Interpolation parameter.
        """
        for target_param, local_param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
