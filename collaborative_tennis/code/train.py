from collections import deque
import numpy as np
import torch

from learning_parameters import *


def add_noise(signal, mean, sigma):
    """
    Adds a gaussian noise over a signal.
    :param signal: The signal (numpy array).
    :param mean: Mean of the noise. Keep it 0.
    :param sigma: The standard deviation of the noise
    :return: The noisified signal.
    """
    signal += np.random.normal(mean, sigma, signal.shape)
    return signal


def step_env(env, agent, brain_name, observations, sigma):
    """
    Helper function to step once in the environment.
    :param env: The unity environment.
    :param agent: The agent to take the actions and learn from it.
    :param brain_name: The name of the unity agent.
    :param observations: The observations of the agents.
    :param sigma: Standard deviation for the noise to be applied on the actions.
    :return: The (1) actions the agents take
                 (2) rewards obtained
                 (3) the next state of the environments
                 (4) flags indication terminal states
    """
    actions = agent.act(observations)
    actions = add_noise(actions, 0., sigma)
    np.clip(actions, -1, 1)
    env_info = env.step(actions)[brain_name]
    next_observations = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    return actions, rewards, next_observations, dones


def save_models(collaborative_agent):
    """
    Saves the trained models of all the agents' actor and critic networks.
    :param collaborative_agent: The collaborative agent.
    """
    for agent_idx, agent in enumerate(collaborative_agent.agents):
        torch.save(agent.actor_local.state_dict(), f'../model/trained_actor_{agent_idx}.pt')
        torch.save(agent.critic_local.state_dict(), f'../model/trained_critic_{agent_idx}.pt')


def train(env, brain_name, agent, num_episodes, max_time_per_episode, score_acceptance_threshold, print_every=100):
    """
    Run the DDPG algorithm to train the agent.
    :param env: The unity environment.
    :param brain_name: The name of the unity agent.
    :param agent: The agent to take the actions and learn from it.
    :param num_episodes: Maximum number of episodes.
    :param max_time_per_episode: Amount of timesteps to let the agent interact with the environment per episode.
    :param score_acceptance_threshold: The target score.
    :param print_every: The episodes after which the scores are printed.
    :return: The averaged scores and the episode at which the agent scores above the acceptance threshold.
    """
    # Track scores.
    averaged_scores = deque(maxlen=print_every)
    num_episodes_to_acceptance_threshold = None

    # Initialize standard deviation for the action noise.
    sigma = MAX_ACTION_STD_DEV

    for episode_idx in range(1, num_episodes + 1):
        # Reset environment and scores at the start of every episode.
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        score = np.zeros(num_agents)

        # Starting observations for both agents.
        observations = env_info.vector_observations

        for t in range(max_time_per_episode):
            # Step once in the environment.
            actions, rewards, next_observations, dones = step_env(env, agent, brain_name, observations, sigma)

            # Learn from the experiences.
            agent.step(observations, actions, rewards, next_observations, dones)

            # Update the state vector.
            observations = next_observations

            # Keep score.
            score += np.array(rewards)

            # Start the next episode if any of the agents in the current episode reach terminal states.
            # We assume that all environments would be terminated at the same time.
            if np.any(dones):
                break

        # Reduce the standard deviation for the actions as we progress through the episodes.
        sigma = max(MIN_ACTION_STD_DEV, sigma * DECAY_RATE_ACTION_STD_DEV)

        # Keep track of scores.
        averaged_scores.append(np.max(score))
        print('\rEpisode {}\tAverage Score: {:.3f}'.format(episode_idx, np.mean(averaged_scores)), end="")
        if np.mean(averaged_scores) >= score_acceptance_threshold and num_episodes_to_acceptance_threshold is None:
            num_episodes_to_acceptance_threshold = episode_idx
            save_models(agent)
            print(f"\nOur agent learnt to get a score of {score_acceptance_threshold} in "
                  f"{num_episodes_to_acceptance_threshold} episodes!")
            break
        if episode_idx % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.3f}'.format(episode_idx, np.mean(
                averaged_scores)))

    save_models(agent)

    return averaged_scores, num_episodes_to_acceptance_threshold
