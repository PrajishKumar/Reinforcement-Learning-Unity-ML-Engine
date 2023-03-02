import random
from replay_buffer import ReplayBuffer
from actor_model import Actor
from critic_model import Critic
import numpy as np
import torch
from learning_parameters import *


class Agent:
    """
    The agent that interacts with the environment and learns from the obtained rewards.  
    
    Note: The agent learns with the DDQN algorithm.
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

        # The "local" policy network that is learnt over timed.
        self.actor_local = Actor(state_size, action_size, seed).to(device)

        # The "target" policy network that's updated less frequently to avoid oscillating updates.
        self.actor_target = Actor(state_size, action_size, seed).to(device)

        # Optimizer for learning the parameters of the policy network.
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # The "local" action-value network that is learnt over timed.
        self.critic_local = Critic(state_size, action_size, seed).to(device)

        # The "target" action-value network that's updated less frequently to avoid oscillating updates.
        self.critic_target = Critic(state_size, action_size, seed).to(device)

        # Optimizer for learning the parameters of the action-value network.
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC,
                                                 weight_decay=WEIGHT_DECAY)

        # Loss function for the critic.
        self.critic_loss_function = torch.nn.MSELoss()

        # Replay buffer.
        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step.
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        """
        Adds experience tuple to the replay buffer and optionally updates the local networks (actor/critic) as needed.
        :param states: The state of the environment of all agents.
        :param actions: The action the agents took at their corresponding @p states.
        :param rewards: The reward obtained by taking @p action at @p state of all agents.
        :param next_states: The new state of the environment once agent @p action at @p state, for all agents.
        :param dones: Flag to know if any of the environments are now in a terminal state.
        """
        # Save experience in replay memory.
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.replay_buffer.add(state, action, reward, next_state, done)

        # Do not learn if we do not have enough samples to learn from.
        if len(self.replay_buffer) <= BATCH_SIZE:
            return

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Sample experiences several times and continue updating parameters.
            for _ in range(NUM_LEARNINGS_PER_UPDATE):
                experiences = self.replay_buffer.sample()
                self.__learn(experiences, GAMMA)

    def act(self, states):
        """
        Returns the actions that the agents should take at a state.
        :param states: The states of the environment.
        :return: The actions to take.
        """
        states_torch = torch.from_numpy(states).float().to(self.device)  # numpy -> torch.

        # Evaluate policy.
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states_torch).cpu().data.numpy()
        self.actor_local.train()

        # Clip the actions to [-1, 1].
        return np.clip(action, -1, 1)

    def __learn(self, experiences, gamma):
        """
        Updates the actor and critic networks.
        :param experiences: Experience (s, a, r, s', done) tuples.
        :param gamma: The discount factor.
        """
        # Update the local critic network.
        self.__update_critic_local(experiences, gamma)

        # Update the local actor network.
        self.__update_actor_local(experiences)

        # Update the target networks.
        self.__update_target_network(self.critic_local, self.critic_target, TAU)
        self.__update_target_network(self.actor_local, self.actor_target, TAU)

    def __update_critic_local(self, experiences, gamma):
        """
        Updates the critic network.
        :param experiences: Experience (s, a, r, s', done) tuples.
        :param gamma: The discount factor.
        """
        states, actions, rewards, next_states, dones = experiences

        # Predict the next set of actions.
        actions_next = self.actor_target(next_states)

        # Predict the action-value for the next timestep.
        critic_targets_next = self.critic_target(next_states, actions_next)

        # Based on the expected future action-value, compute the expected action-value for the current state.
        critic_target = rewards + (gamma * critic_targets_next * (1 - dones))

        # Compute loss.
        critic_expected = self.critic_local(states, actions)
        critic_loss = self.critic_loss_function(critic_expected, critic_target.detach())

        # Minimize the loss.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def __update_actor_local(self, experiences):
        """
        Updates the actor network.
        :param experiences: Experience (s, a, r, s', done) tuples.
        """
        states, actions, rewards, next_states, dones = experiences

        # Predict the set of actions.
        actions_expected = self.actor_local(states)

        # Compute loss. We want to favour actions that increase the action-value in that state.
        actor_loss = -self.critic_local(states, actions_expected).mean()

        # Minimize the loss.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    @staticmethod
    def __update_target_network(local_model, target_model, tau):
        """
        Update the model parameters of the target Q network.

        θ_target = τ * θ_local + (1 - τ) * θ_target
        :param local_model: The model used as a reference to update.
        :param target_model: The model that is updated.
        :param tau: Interpolation parameter.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
