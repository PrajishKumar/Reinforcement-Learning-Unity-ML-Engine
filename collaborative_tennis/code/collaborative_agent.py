import random
import numpy as np
import torch

from individual_agent import IndividualAgent
from replay_buffer import ReplayBuffer
from learning_parameters import *
from utils import update_target_network


class CollaborativeAgent:
    """
    Contains all the individual agents and trains them together.
    """

    def __init__(self, num_agents, observation_size, action_size, device, seed=0):
        """
        Initializes all the individual agents.
        :param num_agents: The number of individual agents.
        :param observation_size: Dimension of the observation space.
        :param action_size: Dimension of the action space. 
        :param device: The device on which the Torch computations are running.
        :param seed: For random number generation.
        """
        self.num_agents = num_agents
        self.observation_size = observation_size
        self.action_size = action_size
        self.device = device
        random.seed(seed)

        # Create the two agents that have to work together.
        self.agents = [IndividualAgent(num_agents, observation_size, action_size, device, seed) for _ in
                       range(num_agents)]

        self.replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, device, seed)

        self.t_step = 0

    def act(self, observations):
        """
        Returns the actions that the agents should take for the current observations.
        :param observations: The observations of the agents.
        :return: The actions to take by each agent.
        """
        actions = [agent.act(observation) for agent, observation in zip(self.agents, observations)]
        return np.array(actions)

    def step(self, observations, actions, rewards, next_observations, dones):
        experiences = (observations, actions, rewards, next_observations, dones)
        self.replay_buffer.add(experiences)

        # Do not learn if we do not have enough samples to learn from.
        if len(self.replay_buffer) <= BATCH_SIZE:
            return

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Sample experiences several times and continue updating parameters.
            for _ in range(NUM_LEARNINGS_PER_UPDATE):
                for experience in self.replay_buffer.sample():

                    (observations_, actions_, rewards_, next_observations_, dones_) = experience
                    observations_ = torch.from_numpy(observations_).float().to(self.device)
                    actions_ = torch.from_numpy(actions_).float().to(self.device)
                    next_observations_ = torch.from_numpy(next_observations_).float().to(self.device)

                    observations_stacked = torch.flatten(observations_)
                    actions_stacked = torch.flatten(actions_)
                    next_observations_stacked = torch.flatten(next_observations_)

                    next_actions_stacked = torch.tensor([], device=self.device)
                    for agent_idx, agent in enumerate(self.agents):
                        next_actions = agent.actor_target(next_observations_[agent_idx])
                        next_actions_stacked = torch.cat((next_actions_stacked, next_actions))
                    next_actions_stacked = torch.flatten(next_actions_stacked)

                    for agent_idx, agent in enumerate(self.agents):
                        next_critic_target = agent.critic_target(next_observations_stacked, next_actions_stacked)
                        critic_target = rewards_[agent_idx] + GAMMA * next_critic_target * (1 - dones_[agent_idx])
                        critic_expected = agent.critic_local(observations_stacked, actions_stacked)

                        critic_loss = agent.critic_loss_function(critic_expected, critic_target.detach())
                        agent.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
                        agent.critic_optimizer.step()

                    for agent_idx, agent in enumerate(self.agents):
                        expected_actions_stacked = torch.tensor([], device=self.device)
                        for agent_idx_, agent_ in enumerate(self.agents):
                            if agent_idx == agent_idx:
                                actions_expected = agent_.actor_local(observations_[agent_idx_])
                            else:
                                actions_expected = agent_.actor_local(observations_[agent_idx_]).detach()
                            expected_actions_stacked = torch.cat((expected_actions_stacked, actions_expected))
                        expected_actions_stacked = torch.flatten(expected_actions_stacked)

                        actor_loss = -agent.critic_local(observations_stacked, expected_actions_stacked).mean()
                        agent.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        agent.actor_optimizer.step()

                    for agent in self.agents:
                        update_target_network(agent.actor_local, agent.actor_target, TAU)
                        update_target_network(agent.critic_local, agent.critic_target, TAU)
