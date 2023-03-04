from collections import deque
import random


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
        random.seed(seed)

    def add(self, experience_tuple) -> None:
        """
        Adds an experience tuple to the replay buffer.
        :param experience_tuple: The (s, a, r, s', d) tuple for all agents.
        """
        self.memory.append(experience_tuple)

    def sample(self):
        """
        Randomly sample a batch of experiences from the replay buffer.

        Note: The sampling is random, not prioritized in this implementation.
        """
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """
        Return the current size of internal memory.
        """
        return len(self.memory)
