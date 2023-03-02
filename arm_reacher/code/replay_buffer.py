from collections import deque, namedtuple
import numpy as np
import random
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
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done) -> None:
        """
        Adds an experience tuple to the replay buffer.
        :param state: The state of the environment.
        :param action: The action the agent took at @p state.
        :param reward: The reward obtained by taking @p action at @p state.
        :param next_state:  The new state of the environment once agent @p action at @p state.
        :param done: Flag to know if the environment is now in a terminal state.
        """
        experience_tuple = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience_tuple)

    def sample(self):
        """
        Randomly sample a batch of experiences from the replay buffer.

        Note: The sampling is random, not prioritized in this implementation.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
