from collections import deque, namedtuple
import random
import numpy as np
import torch


class ReplayBuffer:
    """
    Container for storing experience tuples. Accepts tuples only up to a maximum size.

    Note: The storing and sampling of tuples is not prioritized in this implementation.
    """

    def __init__(self, action_size, buffer_size, batch_size, device, seed=0):
        """
        Initialize the buffer.
        :param action_size: Dimension of the action space.
        :param buffer_size: The maximum number of experience tuples to store.
        :param batch_size: Number of experience tuples to sample at every instance of network training.
        :param device: The device on which the Torch computations are running.
        :param seed: For random number generation.
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience",
                                     field_names=["obs_1", "action_1", "reward_1", "next_obs_1", "done_1",
                                                  "obs_2", "action_2", "reward_2", "next_obs_2", "done_2"])
        random.seed(seed)

    def add(self, observations, actions, rewards, next_observations, dones) -> None:
        # """
        # Adds an experience tuple to the replay buffer.
        # :param experience_tuple: The (s, a, r, s', d) tuple for all agents.
        # """
        experience_tuple = self.experience(observations[0], actions[0], rewards[0], next_observations[0], dones[0],
                                           observations[1], actions[1], rewards[1], next_observations[1], dones[1])
        self.memory.append(experience_tuple)

    def sample(self):
        """
        Randomly sample a batch of experiences from the replay buffer.

        Note: The sampling is random, not prioritized in this implementation.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        obs_1 = torch.from_numpy(np.vstack([e.obs_1 for e in experiences if e is not None])).float().to(self.device)
        action_1 = torch.from_numpy(np.vstack([e.action_1 for e in experiences if e is not None])).float().to(
            self.device)
        reward_1 = torch.from_numpy(np.vstack([e.reward_1 for e in experiences if e is not None])).float().to(
            self.device)
        next_obs_1 = torch.from_numpy(np.vstack([e.next_obs_1 for e in experiences if e is not None])).float().to(
            self.device)
        done_1 = torch.from_numpy(np.vstack([e.done_1 for e in experiences if e is not None])).float().to(self.device)
        obs_2 = torch.from_numpy(np.vstack([e.obs_2 for e in experiences if e is not None])).float().to(self.device)
        action_2 = torch.from_numpy(np.vstack([e.action_2 for e in experiences if e is not None])).float().to(
            self.device)
        reward_2 = torch.from_numpy(np.vstack([e.reward_2 for e in experiences if e is not None])).float().to(
            self.device)
        next_obs_2 = torch.from_numpy(np.vstack([e.next_obs_2 for e in experiences if e is not None])).float().to(
            self.device)
        done_2 = torch.from_numpy(np.vstack([e.done_2 for e in experiences if e is not None])).float().to(self.device)

        return obs_1, action_1, reward_1, next_obs_1, done_1, obs_2, action_2, reward_2, next_obs_2, done_2

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
