# Experience replay buffer
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128  # minibatch size to sample from the replay buffer

# Returns
GAMMA = 0.99  # discount factor

# Training the actor
LR_ACTOR = 1e-4  # learning rate

# Training the critic
LR_CRITIC = 3e-4  # learning rate
WEIGHT_DECAY = 0

# Target networks
TAU = 1e-3  # for soft update of target parameters

# DDQN
UPDATE_EVERY = 20  # how often to update the network
NUM_LEARNINGS_PER_UPDATE = 10  # how many times we update the networks per UPDATE_EVERY

# Action noise
MAX_ACTION_STD_DEV = 0.1
MIN_ACTION_STD_DEV = 0.01
DECAY_RATE_ACTION_STD_DEV = 0.995
