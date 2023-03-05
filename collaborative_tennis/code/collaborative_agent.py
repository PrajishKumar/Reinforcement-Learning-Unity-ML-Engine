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
        actions = [agent.act_local(observation).cpu().numpy() for agent, observation in zip(self.agents, observations)]
        return np.array(actions)

    def step(self, observations, actions, rewards, next_observations, dones):
        self.replay_buffer.add(observations, actions, rewards, next_observations, dones)

        # Do not learn if we do not have enough samples to learn from.
        if len(self.replay_buffer) <= BATCH_SIZE:
            return

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Sample experiences several times and continue updating parameters.
            for _ in range(NUM_LEARNINGS_PER_UPDATE):

                obs_1, action_1, reward_1, next_obs_1, done_1, obs_2, action_2, reward_2, next_obs_2, done_2 = \
                    self.replay_buffer.sample()

                observations_stacked = torch.hstack((obs_1, obs_2)).to(self.device).float()
                actions_stacked = torch.hstack((action_1, action_2)).to(self.device).float()
                next_observations_stacked = torch.hstack((next_obs_1, next_obs_2)).to(self.device).float()

                next_action_1 = self.agents[0].act_target(next_obs_1)
                next_action_2 = self.agents[1].act_target(next_obs_2)
                next_actions_stacked = torch.hstack((next_action_1, next_action_2)).to(self.device).detach().float()

                for agent_idx, agent in enumerate(self.agents):
                    agent.critic_target.eval()
                    with torch.no_grad():
                        next_critic_target = agent.critic_target(next_observations_stacked,
                                                                 next_actions_stacked).detach().to(self.device)
                    agent.critic_target.train()

                    if agent_idx == 0:
                        critic_target = reward_1 + GAMMA * next_critic_target * (1 - done_1)
                    else:
                        critic_target = reward_2 + GAMMA * next_critic_target * (1 - done_2)
                    critic_expected = agent.critic_local(observations_stacked, actions_stacked)

                    critic_loss = agent.critic_loss_function(critic_expected, critic_target.detach())  # todo: l1loss
                    agent.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
                    agent.critic_optimizer.step()

                for agent_idx, agent in enumerate(self.agents):
                    if agent_idx == 0:
                        other_agent = self.agents[1]
                        expected_actions_stacked = torch.hstack((agent.actor_local(obs_1),
                                                                 other_agent.act_local(obs_2)))
                    else:
                        other_agent = self.agents[0]
                        expected_actions_stacked = torch.hstack((other_agent.act_local(obs_1),
                                                                 agent.actor_local(obs_2)))

                    actor_loss = -agent.critic_local(observations_stacked, expected_actions_stacked).mean()
                    agent.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    agent.actor_optimizer.step()

                for agent in self.agents:
                    update_target_network(agent.actor_local, agent.actor_target, TAU)
                    update_target_network(agent.critic_local, agent.critic_target, TAU)
